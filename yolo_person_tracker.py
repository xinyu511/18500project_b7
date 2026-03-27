#!/usr/bin/env python3
"""
SafeFollow vision pipeline — person detection, stereo-depth distance estimation,
and horizontal offset output for motion control.

Hardware: Raspberry Pi 5 + eYs3D Stereo Camera + YOLO person detector.

Outputs per frame:
  - user_distance  (metres, median stereo depth within bbox)
  - x_offset       (normalised [-1, 1], left=-1, centre=0, right=+1)
  - track_id       (persistent person ID from YOLO tracker)
  - target_lost    (True when person absent for TARGET_LOSS_FRAMES consecutive frames)
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ── constants ──────────────────────────────────────────────────────────────────
PERSON_CLASS = 0          # COCO class index for "person"
TARGET_DIST_M = 1.25      # desired following distance (metres)
MIN_DIST_M    = 0.5       # minimum reliable stereo range
MAX_DIST_M    = 4.0       # maximum reliable stereo range
TARGET_LOSS_FRAMES = 10   # consecutive missed frames before target-lost signal


# ── eYs3D stereo depth helper ──────────────────────────────────────────────────
def open_stereo_camera(color_device: str, depth_device: str, width: int, height: int):
    """
    Open the eYs3D stereo camera colour stream and depth stream.

    The eYs3D camera exposes two V4L2 nodes under Linux:
      /dev/video0  – colour (left) image
      /dev/video2  – 16-bit depth map (Z16, same resolution)

    If your camera uses a different device index, pass --color-device / --depth-device.
    """
    color_cap = cv2.VideoCapture(color_device, cv2.CAP_V4L2)
    depth_cap = cv2.VideoCapture(depth_device, cv2.CAP_V4L2)

    for cap, label in ((color_cap, "colour"), (depth_cap, "depth")):
        if not cap.isOpened():
            raise RuntimeError(f"Could not open eYs3D {label} device: {cap}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # eYs3D depth stream is 16-bit grey (Z16)
    depth_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Z16 "))

    return color_cap, depth_cap


def read_depth_m(depth_cap) -> np.ndarray | None:
    """
    Read one depth frame and return a float32 array of distances in metres.
    eYs3D Z16 values are in millimetres; divide by 1000.
    Returns None on read failure.
    """
    ok, raw = depth_cap.read()
    if not ok:
        return None
    # raw is CV_16UC1 encoded as a single-channel 16-bit image
    depth_mm = raw.view(np.uint16)          # reinterpret bytes as uint16
    depth_m  = depth_mm.astype(np.float32) / 1000.0
    return depth_m


def median_depth_in_box(depth_m: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Return the median valid depth (metres) within a bounding box.
    Valid pixels are those within [MIN_DIST_M, MAX_DIST_M].
    Returns float('inf') if no valid pixels exist.
    """
    roi = depth_m[y1:y2, x1:x2]
    valid = roi[(roi >= MIN_DIST_M) & (roi <= MAX_DIST_M)]
    if valid.size == 0:
        return float("inf")
    return float(np.median(valid))


# ── tracking state ─────────────────────────────────────────────────────────────
class TargetLossTracker:
    """Counts consecutive frames with no person detected and emits a lost signal."""

    def __init__(self, threshold: int = TARGET_LOSS_FRAMES):
        self.threshold = threshold
        self._miss_count = 0
        self.lost = False

    def update(self, person_detected: bool) -> None:
        if person_detected:
            self._miss_count = 0
            self.lost = False
        else:
            self._miss_count += 1
            if self._miss_count >= self.threshold:
                self.lost = True


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SafeFollow vision pipeline (person tracking + stereo depth)."
    )
    parser.add_argument("--color-device", default="/dev/video0",
                        help="V4L2 device for eYs3D colour stream")
    parser.add_argument("--depth-device", default="/dev/video2",
                        help="V4L2 device for eYs3D Z16 depth stream")
    parser.add_argument("--model",      default="yolo11n.pt",
                        help="YOLO model path or name")
    parser.add_argument("--cam-width",  type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--imgsz",      type=int, default=320,
                        help="YOLO inference image size")
    parser.add_argument("--conf",       type=float, default=0.40,
                        help="Confidence threshold")
    parser.add_argument("--show",       action="store_true",
                        help="Show annotated preview window")
    return parser.parse_args()


# ── main loop ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    color_cap, depth_cap = open_stereo_camera(
        args.color_device, args.depth_device,
        args.cam_width, args.cam_height,
    )

    model = YOLO(args.model)
    loss_tracker = TargetLossTracker()

    frame_cx = args.cam_width / 2.0   # image centre x for offset calculation

    prev_time = time.time()
    last_log_sec = int(prev_time)
    fps = 0.0

    print("SafeFollow vision pipeline running. Press 'q' to quit.")

    while True:
        ok, color_frame = color_cap.read()
        depth_m = read_depth_m(depth_cap)

        if not ok or depth_m is None:
            print("Frame grab failed; stopping.")
            break

        # ── YOLO tracking — person class only ─────────────────────────────────
        # model.track() keeps persistent IDs across frames (needed for re-acquisition).
        # classes=[PERSON_CLASS] drops all non-person detections before NMS,
        # which both speeds up inference and removes false positives.
        results = model.track(
            source=color_frame,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=[PERSON_CLASS],
            persist=True,       # maintain tracker state between calls
            verbose=False,
            device="cpu",
        )

        boxes     = results[0].boxes
        n_persons = len(boxes) if boxes is not None else 0
        loss_tracker.update(n_persons > 0)

        # ── extract closest detected person ───────────────────────────────────
        # The motion controller needs one target; pick the nearest person.
        user_distance = float("inf")
        x_offset      = 0.0
        target_id     = -1

        if n_persons > 0:
            ids = boxes.id  # may be None before tracker initialises
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = (int(v) for v in box.tolist())

                # Stereo depth: median valid depth inside the bounding box
                dist = median_depth_in_box(depth_m, x1, y1, x2, y2)

                if dist < user_distance:        # keep closest person as target
                    user_distance = dist
                    # Normalised horizontal offset: [-1 = far left, +1 = far right]
                    x_offset  = ((x1 + x2) / 2.0 - frame_cx) / frame_cx
                    target_id = int(ids[i]) if ids is not None else -1

        # ── annotate preview ──────────────────────────────────────────────────
        annotated = results[0].plot()

        if n_persons > 0 and user_distance < float("inf"):
            cv2.putText(
                annotated,
                f"Target ID{target_id}  {user_distance:.2f}m  offset={x_offset:+.2f}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
            )

        if loss_tracker.lost:
            cv2.putText(
                annotated, "TARGET LOST — SEARCH",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )

        now = time.time()
        dt  = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt

        cv2.putText(
            annotated, f"FPS: {fps:.1f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

        # ── hand off to motion controller ─────────────────────────────────────
        # Replace this block with your IPC/queue to the motor control module.
        vision_output = {
            "user_distance": user_distance,   # metres
            "x_offset":      x_offset,        # normalised [-1, +1]
            "track_id":      target_id,
            "target_lost":   loss_tracker.lost,
        }
        # e.g. control_queue.put(vision_output)

        if args.show:
            cv2.imshow("SafeFollow — Person Tracker", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        else:
            now_sec = int(now)
            if now_sec != last_log_sec:
                print(
                    f"FPS={fps:.1f}  persons={n_persons}"
                    f"  dist={user_distance:.2f}m  x_off={x_offset:+.2f}"
                    f"  {'LOST' if loss_tracker.lost else 'OK'}"
                )
                last_log_sec = now_sec

    color_cap.release()
    depth_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
