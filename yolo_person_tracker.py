#!/usr/bin/env python3
"""
SafeFollow vision pipeline — person detection, stereo-depth distance estimation,
and horizontal offset output for motion control.

Hardware: Raspberry Pi 5 + eYs3D Stereo Camera + YOLO person detector.

Depth priority:
  1. eYs3D SDK stereo depth  (accurate, metric)      ← requires SDK installed
  2. V4L2 monocular fallback (bbox-height heuristic) ← always available

Outputs per frame:
  - user_distance  (metres)
  - x_offset       (normalised [-1, 1], left=-1, centre=0, right=+1)
  - track_id       (persistent person ID)
  - target_lost    (True after TARGET_LOSS_FRAMES consecutive missed frames)
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ── constants ──────────────────────────────────────────────────────────────────
PERSON_CLASS       = 0
MIN_DIST_M         = 0.5
MAX_DIST_M         = 4.0
TARGET_LOSS_FRAMES = 10

# Monocular fallback constants — calibrate MONO_FOCAL_PX for your camera:
#   stand at a known distance D, note bbox height H in pixels, then:
#   MONO_FOCAL_PX = D * H / PERSON_HEIGHT_M
PERSON_HEIGHT_M = 1.7
MONO_FOCAL_PX   = 462.0   # rough default — override with --focal-length


# ── eYs3D SDK camera ───────────────────────────────────────────────────────────
def try_open_eys3d(sdk_home: str, width: int, height: int, fps: int):
    """
    Try to open the eYs3D camera via the official Python SDK.
    Returns (pipeline, color_fn, depth_fn) on success, or None on failure.

    sdk_home: path to the cloned eys3d_python_wrapper repo root
              (set via --sdk-home or EYS3D_SDK_HOME env var)
    """
    if sdk_home:
        os.environ["EYS3D_SDK_HOME"] = sdk_home

    try:
        from eys3d import Pipeline, Config  # noqa: PLC0415
    except ImportError:
        print("[INFO] eYs3D SDK not found — falling back to monocular depth.")
        return None

    try:
        config = Config()
        # Format 0 = YUY2 colour stream
        config.set_color_stream(streamFormat=0, width=width, height=height, fps=fps)
        # Format 1 = GRAY_TRANSFER depth stream; 11-bit depth
        config.set_depth_stream(streamFormat=1, width=width, height=height, fps=fps)
        config.set_depth_data_type(11)

        pipe = Pipeline()
        pipe.start(config)
        print(f"[INFO] eYs3D SDK opened — {width}x{height} @ {fps}fps")

        def get_color():
            ok, frame = pipe.wait_color_frame(timeout=1600)
            if not ok:
                return None
            arr = frame.get_rgb_data().reshape(frame.get_height(), frame.get_width(), 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        def get_depth():
            ok, frame = pipe.wait_depth_frame(timeout=1600)
            if not ok:
                return None
            # ZD values are in mm — convert to metres
            return frame.get_depth_ZD_value().reshape(
                frame.get_height(), frame.get_width()
            ).astype(np.float32) / 1000.0

        return pipe, get_color, get_depth

    except Exception as e:
        print(f"[WARN] eYs3D SDK failed to open camera: {e}")
        print("[INFO] Falling back to monocular depth.")
        return None


# ── V4L2 fallback camera ───────────────────────────────────────────────────────
def open_v4l2_camera(device: str, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    return cap


# ── depth helpers ──────────────────────────────────────────────────────────────
def median_depth_in_box(depth_m: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Median valid stereo depth (metres) inside a bounding box."""
    roi   = depth_m[y1:y2, x1:x2]
    valid = roi[(roi >= MIN_DIST_M) & (roi <= MAX_DIST_M)]
    return float(np.median(valid)) if valid.size > 0 else float("inf")


def monocular_distance(bbox_h_px: float, focal_px: float) -> float:
    """Apparent-height monocular estimate. Calibrate focal_px for accuracy."""
    if bbox_h_px <= 0:
        return float("inf")
    return (PERSON_HEIGHT_M * focal_px) / bbox_h_px


# ── target-loss tracker ────────────────────────────────────────────────────────
class TargetLossTracker:
    def __init__(self, threshold: int = TARGET_LOSS_FRAMES):
        self.threshold   = threshold
        self._miss_count = 0
        self.lost        = False

    def update(self, detected: bool) -> None:
        if detected:
            self._miss_count = 0
            self.lost = False
        else:
            self._miss_count += 1
            if self._miss_count >= self.threshold:
                self.lost = True


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SafeFollow vision pipeline (person tracking + stereo/monocular depth)."
    )
    p.add_argument("--sdk-home",     default=os.environ.get("EYS3D_SDK_HOME", ""),
                   help="Path to cloned eys3d_python_wrapper repo (enables stereo depth)")
    p.add_argument("--left-device",  default="/dev/video0",
                   help="V4L2 left camera (used when SDK unavailable)")
    p.add_argument("--model",        default="yolo11n.pt")
    p.add_argument("--cam-width",    type=int,   default=640)
    p.add_argument("--cam-height",   type=int,   default=360)
    p.add_argument("--cam-fps",      type=int,   default=15)
    p.add_argument("--imgsz",        type=int,   default=320)
    p.add_argument("--conf",         type=float, default=0.40)
    p.add_argument("--focal-length", type=float, default=MONO_FOCAL_PX,
                   help="Monocular focal length in px (calibrate if SDK unavailable)")
    p.add_argument("--show",         action="store_true")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Try eYs3D SDK first; fall back to V4L2 monocular
    sdk_result = try_open_eys3d(args.sdk_home, args.cam_width, args.cam_height, args.cam_fps)

    if sdk_result is not None:
        pipe, get_color, get_depth = sdk_result
        v4l2_cap   = None
        depth_mode = "stereo"
    else:
        v4l2_cap   = open_v4l2_camera(args.left_device, args.cam_width, args.cam_height)
        get_color  = None
        get_depth  = None
        depth_mode = "monocular"

    print(f"Depth mode: {depth_mode}")

    model        = YOLO(args.model)
    loss_tracker = TargetLossTracker()
    frame_cx     = args.cam_width / 2.0
    focal_px     = args.focal_length

    prev_time    = time.time()
    last_log_sec = int(prev_time)
    fps          = 0.0

    print("SafeFollow running. Press q to quit.")

    while True:
        # ── grab frames ───────────────────────────────────────────────────────
        if depth_mode == "stereo":
            color_frame = get_color()
            depth_m     = get_depth()
            if color_frame is None:
                print("SDK color frame failed; stopping.")
                break
        else:
            ok, color_frame = v4l2_cap.read()
            if not ok:
                print("Camera frame grab failed; stopping.")
                break
            depth_m = None

        # ── YOLO person tracking ───────────────────────────────────────────────
        results = model.track(
            source=color_frame,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=[PERSON_CLASS],
            persist=True,
            tracker="bytetrack_safefollow.yaml",
            verbose=False,
            device="cpu",
        )

        boxes     = results[0].boxes
        n_persons = len(boxes) if boxes is not None else 0
        loss_tracker.update(n_persons > 0)

        # ── pick closest person as target ──────────────────────────────────────
        user_distance = float("inf")
        x_offset      = 0.0
        target_id     = -1

        if n_persons > 0:
            ids = boxes.id
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = (int(v) for v in box.tolist())

                if depth_m is not None:
                    dist = median_depth_in_box(depth_m, x1, y1, x2, y2)
                else:
                    dist = monocular_distance(y2 - y1, focal_px)

                if dist < user_distance:
                    user_distance = dist
                    x_offset  = ((x1 + x2) / 2.0 - frame_cx) / frame_cx
                    target_id = int(ids[i]) if ids is not None else -1

        # ── annotate ───────────────────────────────────────────────────────────
        annotated = results[0].plot()

        if user_distance < float("inf"):
            cv2.putText(annotated,
                f"ID{target_id}  {user_distance:.2f}m  offset={x_offset:+.2f}  [{depth_mode}]",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        if loss_tracker.lost:
            cv2.putText(annotated, "TARGET LOST — SEARCH",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        now = time.time()
        dt  = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt

        cv2.putText(annotated, f"FPS: {fps:.1f}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # ── motion controller output ───────────────────────────────────────────
        vision_output = {
            "user_distance": user_distance,   # metres
            "x_offset":      x_offset,        # normalised [-1, +1]
            "track_id":      target_id,
            "target_lost":   loss_tracker.lost,
        }
        # e.g. control_queue.put(vision_output)

        if args.show:
            cv2.imshow("SafeFollow", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        else:
            now_sec = int(now)
            if now_sec != last_log_sec:
                print(f"FPS={fps:.1f}  persons={n_persons}"
                      f"  dist={user_distance:.2f}m  x_off={x_offset:+.2f}"
                      f"  {'LOST' if loss_tracker.lost else 'OK'}")
                last_log_sec = now_sec

    if depth_mode == "stereo":
        pipe.stop()
    else:
        v4l2_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
