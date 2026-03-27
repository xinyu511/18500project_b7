#!/usr/bin/env python3
"""
SafeFollow vision pipeline — person detection, stereo-depth distance estimation,
and horizontal offset output for motion control.

Hardware: Raspberry Pi 5 + eYs3D Stereo Camera + YOLO person detector.

The eYs3D camera does NOT expose a Z16 depth stream over V4L2.
Depth is computed here using OpenCV StereoSGBM from the left (video0)
and right (video1) YUYV streams.

Outputs per frame:
  - user_distance  (metres, median stereo depth within person bbox)
  - x_offset       (normalised [-1, 1], left=-1, centre=0, right=+1)
  - track_id       (persistent person ID from YOLO tracker)
  - target_lost    (True when person absent for TARGET_LOSS_FRAMES frames)
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ── constants ──────────────────────────────────────────────────────────────────
PERSON_CLASS = 0
MIN_DIST_M   = 0.5
MAX_DIST_M   = 4.0
TARGET_LOSS_FRAMES = 10

# eYs3D stereo camera intrinsics (adjust after calibration if needed).
# Focal length in pixels at 640×480; baseline in metres.
FOCAL_LENGTH_PX = 462.0   # approximate for eYs3D at 640×480
BASELINE_M      = 0.060   # ~60 mm baseline (typical eYs3D)


# ── camera helpers ─────────────────────────────────────────────────────────────
def open_left_camera(device: str, width: int, height: int) -> cv2.VideoCapture:
    """Open the left (primary) eye — used for YOLO and colour display."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open left camera: {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    return cap


def open_right_camera(device: str, width: int, height: int):
    """
    Try to open the right eye / combined stereo stream.
    Uses 320x480 (30fps) — smaller than 640x720 (10fps) to avoid select() timeout.
    Returns None if the device cannot stream (falls back to monocular depth).
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[WARN] Right camera {device} not available — using monocular depth.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Do NOT force YUYV — let the driver pick the active format
    ok, _ = cap.read()
    if not ok:
        print(f"[WARN] Right camera {device} gave no frames — using monocular depth.")
        cap.release()
        return None
    return cap


def split_stereo_frame(combined: np.ndarray):
    """Split a vertically stacked combined frame into left and right images."""
    h = combined.shape[0] // 2
    return combined[:h, :], combined[h:, :]


# Monocular fallback: known person height + focal length → distance
PERSON_HEIGHT_M  = 1.7
MONO_FOCAL_PX    = 462.0   # calibrate: dist_known * bbox_h / 1.7

def monocular_distance(bbox_h_px: float) -> float:
    if bbox_h_px <= 0:
        return float("inf")
    return (PERSON_HEIGHT_M * MONO_FOCAL_PX) / bbox_h_px


# ── stereo depth ───────────────────────────────────────────────────────────────
def make_stereo_matcher() -> cv2.StereoSGBM:
    """
    StereoSGBM tuned for indoor person tracking (0.5–4 m range).
    minDisparity=0, numDisparities must be divisible by 16.
    """
    win  = 5
    ndisp = 96   # covers ~0.7 m at 4 m with 60 mm baseline / 462 px focal
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=ndisp,
        blockSize=win,
        P1=8  * 3 * win ** 2,
        P2=32 * 3 * win ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def disparity_to_depth(disp: np.ndarray) -> np.ndarray:
    """
    Convert SGBM disparity (fixed-point ×16) to float32 depth in metres.
    depth = focal_px * baseline_m / disparity_px
    """
    disp_f = disp.astype(np.float32) / 16.0
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(
            disp_f > 0,
            FOCAL_LENGTH_PX * BASELINE_M / disp_f,
            np.inf,
        )
    return depth


def median_depth_in_box(depth_m: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    roi   = depth_m[y1:y2, x1:x2]
    valid = roi[(roi >= MIN_DIST_M) & (roi <= MAX_DIST_M)]
    return float(np.median(valid)) if valid.size > 0 else float("inf")


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
        description="SafeFollow vision pipeline (person tracking + stereo depth)."
    )
    p.add_argument("--left-device",  default="/dev/video0", help="eYs3D left eye (colour + YOLO)")
    p.add_argument("--right-device", default="/dev/video2", help="eYs3D right eye (stereo depth)")
    p.add_argument("--model",        default="yolo11n.pt")
    p.add_argument("--cam-width",    type=int,   default=640)
    p.add_argument("--cam-height",   type=int,   default=360,
                   help="Single-eye frame height (320x480 combined → 320x240 each)")
    p.add_argument("--imgsz",        type=int,   default=320, help="YOLO inference size")
    p.add_argument("--conf",         type=float, default=0.40)
    p.add_argument("--show",         action="store_true", help="Show annotated preview")
    return p.parse_args()


# ── main loop ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Left eye always required (YOLO runs here)
    left_cap = open_left_camera(args.left_device, args.cam_width, args.cam_height)

    # Right eye optional — falls back to monocular if unavailable
    right_cap   = open_right_camera(args.right_device, args.cam_width, args.cam_height)
    use_stereo  = right_cap is not None
    stereo      = make_stereo_matcher() if use_stereo else None
    depth_mode  = "stereo" if use_stereo else "monocular"
    print(f"Depth mode: {depth_mode}")

    model        = YOLO(args.model)
    loss_tracker = TargetLossTracker()
    frame_cx     = args.cam_width / 2.0

    prev_time    = time.time()
    last_log_sec = int(prev_time)
    fps          = 0.0

    print(f"SafeFollow running — left={args.left_device} right={args.right_device}. Press q to quit.")

    while True:
        ok, left_frame = left_cap.read()
        if not ok:
            print("Left camera frame grab failed; stopping.")
            break

        # ── depth ─────────────────────────────────────────────────────────────
        if use_stereo:
            ok_r, right_frame = right_cap.read()
            if ok_r:
                gray_l  = cv2.cvtColor(left_frame,  cv2.COLOR_BGR2GRAY)
                gray_r  = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                disp    = stereo.compute(gray_l, gray_r)
                depth_m = disparity_to_depth(disp)
            else:
                depth_m = None   # stereo read failed this frame; use monocular below
        else:
            depth_m = None

        # ── YOLO person tracking ───────────────────────────────────────────────
        results = model.track(
            source=left_frame,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=[PERSON_CLASS],
            persist=True,
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
                    dist = monocular_distance(y2 - y1)

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

    left_cap.release()
    if right_cap is not None:
        right_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
