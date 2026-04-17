#!/usr/bin/env python3
import argparse
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO

H_BINS = 24
S_BINS = 16
OVERLAP_IOU_THRESHOLD = 0.20
TARGET_LOSS_FRAMES    = 10   # consecutive missing-target frames before SEARCH
FAST_RELOCK_FRAMES    = 8    # if locked ID missing this long AND people are visible,
                             # release the lock immediately (handles ID-change events
                             # e.g. person moved very close to the camera)
DEDUP_IOU_THRESHOLD   = 0.60 # merge duplicate YOLO boxes (same person, two detections)


class TargetLossTracker:
    """Declares the locked target 'lost' after N consecutive missed frames."""

    def __init__(self, threshold: int = TARGET_LOSS_FRAMES):
        self.threshold   = threshold
        self._miss_count = 0
        self.lost        = False

    def update(self, detected: bool) -> None:
        if detected:
            self._miss_count = 0
            self.lost        = False
        else:
            self._miss_count += 1
            if self._miss_count >= self.threshold:
                self.lost = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tiny YOLO on Raspberry Pi camera (/dev/video0).",
        add_help=False,   # allow use as parents=[...] in main.py
    )
    parser.add_argument("--device", default="/dev/video0", help="V4L2 camera device")
    parser.add_argument(
        "--right-device",
        default="/dev/video2",
        help="Right camera device when --stereo-source=dual",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path or name")
    parser.add_argument("--cam-width", type=int, default=424, help="Camera capture width")
    parser.add_argument("--cam-height", type=int, default=240, help="Camera capture height")
    parser.add_argument(
        "--distance-mode",
        choices=("bbox", "stereo"),
        default="bbox",
        help="Distance estimation mode",
    )
    parser.add_argument(
        "--stereo-source",
        choices=("sbs", "dual"),
        default="sbs",
        help="Stereo input type: side-by-side stream or dual device",
    )
    parser.add_argument(
        "--stereo-width",
        type=int,
        default=1280,
        help="Capture width for side-by-side stereo mode",
    )
    parser.add_argument(
        "--stereo-height",
        type=int,
        default=480,
        help="Capture height for side-by-side stereo mode",
    )
    parser.add_argument(
        "--baseline-m",
        type=float,
        default=0.06,
        help="Stereo baseline in meters (distance between left/right camera centers)",
    )
    parser.add_argument(
        "--stereo-proc-scale",
        type=float,
        default=0.35,
        help="Downscale factor for disparity computation",
    )
    parser.add_argument(
        "--disp-block-size",
        type=int,
        default=7,
        help="StereoSGBM block size (odd number)",
    )
    parser.add_argument(
        "--disp-num",
        type=int,
        default=48,
        help="Stereo matcher numDisparities (multiple of 16)",
    )
    parser.add_argument(
        "--stereo-algo",
        choices=("bm", "sgbm"),
        default="bm",
        help="Stereo matcher algorithm (bm is faster, sgbm can be denser)",
    )
    parser.add_argument(
        "--disp-every",
        type=int,
        default=2,
        help="Recompute disparity every N frames (reuse previous map in between)",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Multiplicative correction applied to stereo depth",
    )
    parser.add_argument(
        "--depth-offset",
        type=float,
        default=0.0,
        help="Subtractive correction in meters applied after depth scaling",
    )
    parser.add_argument(
        "--swap-lr",
        action="store_true",
        help="Swap left/right images before stereo matching if depth looks inverted",
    )
    parser.add_argument(
        "--min-depth-m",
        type=float,
        default=0.3,
        help="Minimum accepted stereo depth in meters",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=8.0,
        help="Maximum accepted stereo depth in meters",
    )
    parser.add_argument(
        "--sample-y-ratio",
        type=float,
        default=0.7,
        help="Vertical sample point inside bbox for depth (0=top, 1=bottom)",
    )
    parser.add_argument(
        "--show-disparity",
        action="store_true",
        help="Show disparity visualization window",
    )
    parser.add_argument("--imgsz", type=int, default=256, help="YOLO inference image size")
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="Camera pixel format FOURCC (e.g. MJPG, YUYV)",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument(
        "--hfov-deg",
        type=float,
        default=62.0,
        help="Camera horizontal field-of-view in degrees",
    )
    parser.add_argument(
        "--vfov-deg",
        type=float,
        default=49.0,
        help="Camera vertical field-of-view in degrees",
    )
    parser.add_argument(
        "--person-height-m",
        type=float,
        default=1.70,
        help="Assumed real person height in meters (for distance estimate)",
    )
    parser.add_argument("--show", action="store_true", help="Show annotated preview window")
    return parser


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        parents=[build_parser()],
        description="Run tiny YOLO on Raspberry Pi camera (/dev/video0).",
    ).parse_args()


def estimate_distance_and_angle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: int,
    frame_h: int,
    hfov_deg: float,
    vfov_deg: float,
    person_height_m: float,
) -> tuple[float, float]:
    bbox_h = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)

    focal_px_x = (frame_w * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
    focal_px_y = (frame_h * 0.5) / math.tan(math.radians(vfov_deg) * 0.5)

    distance_m = (person_height_m * focal_px_y) / bbox_h
    angle_deg = math.degrees(math.atan((cx - frame_w * 0.5) / focal_px_x))
    return distance_m, angle_deg


def make_stereo_matcher(
    algo: str, block_size: int, num_disparities: int
) -> cv2.StereoMatcher:
    if block_size % 2 == 0:
        block_size += 1
    if num_disparities < 16:
        num_disparities = 16
    if num_disparities % 16 != 0:
        num_disparities = (num_disparities // 16 + 1) * 16

    if algo == "bm":
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        matcher.setPreFilterSize(9)
        matcher.setPreFilterCap(31)
        matcher.setTextureThreshold(10)
        matcher.setUniquenessRatio(10)
        matcher.setSpeckleWindowSize(50)
        matcher.setSpeckleRange(2)
        return matcher

    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def get_stereo_frames(
    args: argparse.Namespace, cap_left: cv2.VideoCapture, cap_right: cv2.VideoCapture | None
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    if args.stereo_source == "sbs":
        ok, frame = cap_left.read()
        if not ok:
            return False, None, None
        _h, w = frame.shape[:2]
        half_w = w // 2
        left = frame[:, :half_w]
        right = frame[:, half_w:]
        if args.swap_lr:
            left, right = right, left
        return True, left, right

    ok_left, left = cap_left.read()
    ok_right, right = cap_right.read() if cap_right is not None else (False, None)
    if not ok_left or not ok_right:
        return False, None, None
    if args.swap_lr:
        left, right = right, left
    return True, left, right


def disparity_to_depth_m(disparity_px: float, focal_px: float, baseline_m: float) -> float:
    if disparity_px <= 0.5:
        return float("inf")
    return (focal_px * baseline_m) / disparity_px


def sample_depth_from_disparity(
    disparity_map: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    focal_px: float,
    baseline_m: float,
    min_depth_m: float,
    max_depth_m: float,
    patch_radius: int = 5,
    num_samples: int = 5,
) -> float | None:
    cx = int(0.5 * (x1 + x2))
    depths: list[float] = []
    h, w = disparity_map.shape[:2]

    for i in range(num_samples):
        t = 0.3 + 0.4 * (i / max(1, num_samples - 1))
        cy = int(y1 + t * (y2 - y1))

        py0 = max(0, cy - patch_radius)
        py1 = min(h, cy + patch_radius + 1)
        px0 = max(0, cx - patch_radius)
        px1 = min(w, cx + patch_radius + 1)

        patch = disparity_map[py0:py1, px0:px1]
        valid = patch[patch > 0.5]
        if valid.size == 0:
            continue

        disp = float(np.median(valid))
        depth_m = disparity_to_depth_m(disp, focal_px, baseline_m)
        if np.isfinite(depth_m) and min_depth_m <= depth_m <= max_depth_m:
            depths.append(depth_m)

    if not depths:
        return None
    return float(np.median(depths))


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def person_color_histogram(frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """Extract HSV appearance descriptor from torso region."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return None

    bh = y2 - y1
    torso_y1 = y1 + int(0.20 * bh)
    torso_y2 = y1 + int(0.75 * bh)
    torso_y1 = max(y1, min(torso_y1, y2 - 1))
    torso_y2 = max(torso_y1 + 1, min(torso_y2, y2))
    roi = frame_bgr[torso_y1:torso_y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


def color_similarity(hist_a, hist_b) -> float:
    if hist_a is None or hist_b is None:
        return 0.0
    dist = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)  # 0=best
    return float(max(0.0, min(1.0, 1.0 - dist)))


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def dedup_overlapping(detections: list[dict], iou_thresh: float = DEDUP_IOU_THRESHOLD) -> list[dict]:
    """Drop duplicate YOLO boxes (same person detected twice) via NMS-style filter.
    Keeps the higher-confidence detection when two boxes overlap above the threshold."""
    sorted_dets = sorted(detections, key=lambda d: -d["conf"])
    kept: list[dict] = []
    for det in sorted_dets:
        box = (det["x1"], det["y1"], det["x2"], det["y2"])
        if any(bbox_iou(box, (k["x1"], k["y1"], k["x2"], k["y2"])) > iou_thresh
               for k in kept):
            continue
        kept.append(det)
    return kept


class TrackIdMapper:
    """Hybrid tracker+appearance mapping to compact, stable display IDs."""

    def __init__(
        self,
        max_missing_frames: int = 45,
        hist_ema: float = 0.20,
        match_threshold: float = 0.55,
    ):
        self.max_missing_frames = max_missing_frames
        self.hist_ema = hist_ema
        self.match_threshold = match_threshold
        self.raw_to_display: dict[int, int] = {}
        self.display_models: dict[int, dict] = {}
        self.next_display_id = 1

    def _alloc_display_id(self) -> int:
        active_ids = set(self.display_models.keys())
        next_id = 1
        while next_id in active_ids:
            next_id += 1
        self.next_display_id = next_id + 1
        return next_id

    def _match_display_id(
        self, hist, cx: float, cy: float, frame_diag: float, used_ids: set[int], overlap: bool
    ):
        best_id = None
        best_score = float("inf")
        color_w = 0.90 if overlap else 0.75
        pos_w = 1.0 - color_w
        for disp_id, model in self.display_models.items():
            if disp_id in used_ids:
                continue
            sim = color_similarity(hist, model.get("hist"))
            color_term = 1.0 - sim
            pos_term = min(
                1.0, math.hypot(cx - model.get("cx", cx), cy - model.get("cy", cy)) / frame_diag
            )
            score = color_w * color_term + pos_w * pos_term
            if score < best_score:
                best_score = score
                best_id = disp_id
        if best_id is not None and best_score <= self.match_threshold:
            return best_id
        return None

    def update(self, detections: list[dict], frame_idx: int, frame_w: int, frame_h: int) -> list[int]:
        frame_diag = max(1.0, math.hypot(frame_w, frame_h))
        assigned_display_ids: list[int] = []
        used_ids: set[int] = set()
        overlap_flags = [False] * len(detections)
        for i in range(len(detections)):
            box_i = (
                detections[i]["x1"],
                detections[i]["y1"],
                detections[i]["x2"],
                detections[i]["y2"],
            )
            for j in range(i + 1, len(detections)):
                box_j = (
                    detections[j]["x1"],
                    detections[j]["y1"],
                    detections[j]["x2"],
                    detections[j]["y2"],
                )
                if bbox_iou(box_i, box_j) >= OVERLAP_IOU_THRESHOLD:
                    overlap_flags[i] = True
                    overlap_flags[j] = True

        for idx, det in enumerate(detections):
            raw_id = det["raw_track_id"]
            hist = det["hist"]
            cx = det["cx"]
            cy = det["cy"]
            overlap = overlap_flags[idx]

            display_id = None
            if raw_id is not None and raw_id in self.raw_to_display:
                candidate = self.raw_to_display[raw_id]
                model = self.display_models.get(candidate)
                sim_to_candidate = color_similarity(hist, model.get("hist")) if model is not None else 0.0
                if (
                    model is not None
                    and candidate not in used_ids
                    and frame_idx - model.get("last_seen", frame_idx) <= self.max_missing_frames
                    and (not overlap or sim_to_candidate >= 0.35)
                ):
                    display_id = candidate

            if display_id is None:
                display_id = self._match_display_id(hist, cx, cy, frame_diag, used_ids, overlap)

            if display_id is None:
                display_id = self._alloc_display_id()

            model = self.display_models.get(display_id)
            if model is None:
                self.display_models[display_id] = {
                    "hist": hist,
                    "cx": cx,
                    "cy": cy,
                    "last_seen": frame_idx,
                }
            else:
                if hist is not None:
                    if model.get("hist") is None:
                        model["hist"] = hist
                    else:
                        hist_ema = min(self.hist_ema, 0.08) if overlap else self.hist_ema
                        model["hist"] = cv2.addWeighted(
                            model["hist"], 1.0 - hist_ema, hist, hist_ema, 0.0
                        )
                        model["hist"] = cv2.normalize(
                            model["hist"], model["hist"], alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1
                        )
                model["cx"] = 0.7 * model.get("cx", cx) + 0.3 * cx
                model["cy"] = 0.7 * model.get("cy", cy) + 0.3 * cy
                model["last_seen"] = frame_idx

            if raw_id is not None:
                self.raw_to_display[raw_id] = display_id
            assigned_display_ids.append(display_id)
            used_ids.add(display_id)

        stale_displays = [
            disp_id
            for disp_id, model in self.display_models.items()
            if frame_idx - model.get("last_seen", frame_idx) > self.max_missing_frames
        ]
        for disp_id in stale_displays:
            self.display_models.pop(disp_id, None)

        valid_disp_ids = set(self.display_models.keys())
        stale_raw_ids = [
            raw_id for raw_id, disp_id in self.raw_to_display.items() if disp_id not in valid_disp_ids
        ]
        for raw_id in stale_raw_ids:
            self.raw_to_display.pop(raw_id, None)

        return assigned_display_ids


def _compute_distance_for_det(det: dict, args, disparity_for_sampling, focal_px_x,
                              frame_w: int, frame_h: int) -> tuple[float, bool]:
    """Return (distance_m, is_fallback). Tries stereo first, falls back to bbox."""
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

    if args.distance_mode == "stereo" and disparity_for_sampling is not None:
        dist = sample_depth_from_disparity(
            disparity_map=disparity_for_sampling,
            x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
            focal_px=focal_px_x, baseline_m=args.baseline_m,
            min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m,
        )
        if dist is not None:
            dist = args.depth_scale * dist - args.depth_offset
            if dist < args.min_depth_m or dist > args.max_depth_m:
                dist = None
        if dist is not None:
            return dist, False

    # bbox-height fallback
    dist, _ = estimate_distance_and_angle(
        x1=x1, y1=y1, x2=x2, y2=y2,
        frame_w=frame_w, frame_h=frame_h,
        hfov_deg=args.hfov_deg, vfov_deg=args.vfov_deg,
        person_height_m=args.person_height_m,
    )
    return dist, True


def run_pipeline(args, on_vision=None, status_provider=None) -> None:
    """
    Run the vision pipeline.

    on_vision: optional callable invoked each frame with keyword arguments:
                 on_vision(user_distance, x_offset, target_lost)
               user_distance is metres, x_offset is normalised [-1, +1]
               (left = -1, centre = 0, right = +1), target_lost is bool.

    status_provider: optional zero-arg callable that returns a dict of
               motor-controller state to overlay on the video (when --show).
               Expected keys: state, left_pwm, right_pwm, forward, turn,
               dist_err, json_cmd.

    Target selection: locks onto one person (by stable display_id from
    TrackIdMapper). On initial acquisition — or when the locked target has
    been purged from the mapper — picks the closest person as the new lock.
    """
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device: {args.device}")

    cap_right = None
    if args.distance_mode == "stereo" and args.stereo_source == "dual":
        cap_right = cv2.VideoCapture(args.right_device, cv2.CAP_V4L2)
        if not cap_right.isOpened():
            raise RuntimeError(f"Could not open right camera device: {args.right_device}")

    if args.distance_mode == "stereo" and args.stereo_source == "sbs":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.stereo_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.stereo_height)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        if cap_right is not None:
            cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc.upper()))
    if cap_right is not None:
        cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc.upper()))

    model = YOLO(args.model)
    stereo_matcher = make_stereo_matcher(args.stereo_algo, args.disp_block_size, args.disp_num)

    prev_time = time.time()
    last_logged_sec = int(prev_time)
    fps = 0.0
    frame_idx = 0
    disparity_for_sampling = None
    track_id_mapper = TrackIdMapper()

    # Target-following state
    loss_tracker        = TargetLossTracker()
    locked_id: int | None = None
    last_known_distance = 1.25
    last_known_offset   = 0.0
    frames_since_target = 0   # counter for fast-relock

    print("Running detection. Press 'q' in preview window to quit.")
    try:
        while True:
            frame_idx += 1
            if args.distance_mode == "stereo":
                ok, left_frame, right_frame = get_stereo_frames(args, cap, cap_right)
                frame = left_frame
            else:
                ok, frame = cap.read()
                right_frame = None
            if not ok:
                print("Frame grab failed; stopping.")
                break

            results = model.track(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[0],  # person only (COCO class id 0)
                verbose=False,
                device="cpu",
                persist=True,
                tracker="bytetrack.yaml",
            )

            annotated = frame.copy()
            frame_h, frame_w = annotated.shape[:2]
            focal_px_x = (frame_w * 0.5) / math.tan(math.radians(args.hfov_deg) * 0.5)
            half_hfov  = args.hfov_deg * 0.5
            disparity_vis = None
            if args.distance_mode == "stereo":
                scale = max(0.2, min(1.0, args.stereo_proc_scale))
                refresh_every = max(1, args.disp_every)
                if disparity_for_sampling is None or frame_idx % refresh_every == 0:
                    left_small = cv2.resize(
                        left_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                    right_small = cv2.resize(
                        right_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                    gray_l = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
                    gray_r = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)
                    if args.stereo_algo == "bm":
                        gray_l = cv2.equalizeHist(gray_l)
                        gray_r = cv2.equalizeHist(gray_r)
                    disparity_small = stereo_matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
                    disparity_small /= scale
                    disparity_small = cv2.medianBlur(disparity_small, 5)
                    disparity_for_sampling = cv2.resize(
                        disparity_small, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR
                    )
                    if args.show_disparity:
                        disp_norm = cv2.normalize(
                            disparity_small, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                        )
                        disparity_vis = disp_norm.astype(np.uint8)

            boxes = results[0].boxes
            detections: list[dict] = []
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    raw_track_id = int(box.id[0]) if box.id is not None else None
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    hist = person_color_histogram(frame, int(x1), int(y1), int(x2), int(y2))
                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf, "raw_track_id": raw_track_id,
                        "cx": cx, "cy": cy, "hist": hist,
                    })

            # Drop duplicate YOLO boxes (same person detected twice with high IoU)
            detections = dedup_overlapping(detections)

            display_ids = track_id_mapper.update(detections, frame_idx, frame_w, frame_h)

            # Enrich detections with distance + angle + display_id
            for det, did in zip(detections, display_ids):
                det["display_id"] = did
                det["angle_deg"]  = math.degrees(
                    math.atan(((0.5 * (det["x1"] + det["x2"])) - frame_w * 0.5) / focal_px_x)
                )
                dist, fb = _compute_distance_for_det(
                    det, args, disparity_for_sampling, focal_px_x, frame_w, frame_h
                )
                det["distance_m"]  = dist
                det["dist_is_fallback"] = fb

            # ── target selection ──────────────────────────────────────────
            target_det = None
            if locked_id is not None:
                target_det = next(
                    (d for d in detections if d["display_id"] == locked_id), None
                )
                # Release lock if the target has been purged by the mapper
                # (happens after TrackIdMapper.max_missing_frames ≈ 45 frames)
                if target_det is None and locked_id not in track_id_mapper.display_models:
                    print(f"[vision] Lost target ID {locked_id} — releasing lock (purged)")
                    locked_id = None

            # Update miss counter for fast-relock
            if target_det is not None:
                frames_since_target = 0
            else:
                frames_since_target += 1

            # Fast re-lock: if locked target has been missing for FAST_RELOCK_FRAMES
            # AND at least one person is visible, assume an ID-change event
            # (e.g. person moved very close, ByteTrack dropped them, new display_id
            # was allocated) and release the stale lock so we can re-acquire.
            if (locked_id is not None and target_det is None
                    and len(detections) > 0
                    and frames_since_target >= FAST_RELOCK_FRAMES):
                print(f"[vision] Fast re-lock: ID {locked_id} missed for "
                      f"{frames_since_target} frames, {len(detections)} person(s) "
                      f"visible — releasing lock")
                locked_id = None
                frames_since_target = 0

            if locked_id is None and len(detections) > 0:
                # Initial / re-acquisition: lock onto the closest person
                target_det = min(detections, key=lambda d: d["distance_m"])
                locked_id = target_det["display_id"]
                print(f"[vision] Locked onto target ID {locked_id}")

            loss_tracker.update(detected=(target_det is not None))

            if target_det is not None:
                last_known_distance = target_det["distance_m"]
                # Convert angle → normalised x_offset using half-HFOV
                last_known_offset = max(-1.0, min(1.0,
                    target_det["angle_deg"] / half_hfov
                ))

            # ── motor callback ────────────────────────────────────────────
            if on_vision is not None:
                on_vision(
                    user_distance=last_known_distance,
                    x_offset=last_known_offset,
                    target_lost=loss_tracker.lost,
                )

            # ── annotation ────────────────────────────────────────────────
            for det in detections:
                is_target = (det is target_det)
                color = (0, 255, 255) if is_target else (0, 255, 0)  # yellow / green
                p1 = (int(det["x1"]), int(det["y1"]))
                p2 = (int(det["x2"]), int(det["y2"]))
                cv2.rectangle(annotated, p1, p2, color, 2)
                dist_text = (f"~{det['distance_m']:.2f}m" if det["dist_is_fallback"]
                             else f"{det['distance_m']:.2f}m")
                person_name = (f"person {det['display_id']}"
                               if det["display_id"] is not None else "person")
                label = (f"{person_name} {det['conf']:.2f} "
                         f"{dist_text} {det['angle_deg']:+.1f}deg")
                cv2.putText(annotated, label,
                            (p1[0], max(20, p1[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            now = time.time()
            dt  = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            lock_str = f"LOCK: {locked_id}" if locked_id is not None else "LOCK: none"
            if loss_tracker.lost:
                lock_str += "  [LOST]"
            cv2.putText(annotated,
                        f"FPS: {fps:.1f}  persons: {len(detections)}  {lock_str}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # ── motor controller overlay ──────────────────────────────────
            if status_provider is not None:
                st = status_provider()
                state_color = {
                    "FOLLOW": (0, 255, 0),    # green
                    "AVOID":  (0, 255, 255),  # yellow
                    "SEARCH": (0, 200, 255),  # orange
                    "STOP":   (0, 0, 255),    # red
                }.get(st["state"], (200, 200, 200))

                # Stack overlay lines from bottom-left upward
                fh = annotated.shape[0]
                fw = annotated.shape[1]
                lines = [
                    f"STATE: {st['state']}",
                    f"JSON:  {{\"T\":1,\"L\":{st['left_pwm']:+d},\"R\":{st['right_pwm']:+d}}}",
                    f"fwd={st['forward']:+.2f}  turn={st['turn']:+.2f}  "
                    f"err={st['dist_err']:+.2f}m",
                ]
                y = fh - 12
                for line in reversed(lines):
                    cv2.putText(annotated, line, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)
                    y -= 22

                # ── obstacle radar (top-down view) ────────────────────────
                obs = st.get("obstacles", {})
                if obs:
                    from motor_ctrl import DANGER_DIST, CAUTION_DIST
                    radar_r  = 40              # radar circle radius in px
                    radar_cx = fw - radar_r - 10
                    radar_cy = fh - 160        # above wheel bars

                    # Background circle
                    cv2.circle(annotated, (radar_cx, radar_cy), radar_r,
                               (40, 40, 40), -1)
                    cv2.circle(annotated, (radar_cx, radar_cy), radar_r,
                               (100, 100, 100), 1)
                    # Robot dot
                    cv2.circle(annotated, (radar_cx, radar_cy), 4,
                               (255, 255, 255), -1)

                    # Draw obstacle dots in 4 directions
                    # direction → (dx, dy) in image coords (up = -y)
                    dirs = {
                        "front": (0, -1),
                        "back":  (0, +1),
                        "left":  (-1, 0),
                        "right": (+1, 0),
                    }
                    for name, (dx, dy) in dirs.items():
                        dist_m = obs.get(name, 2.0)
                        # Map distance to radar radius (0m = centre, 2m = edge)
                        frac = min(1.0, dist_m / 2.0)
                        px = int(radar_cx + dx * frac * (radar_r - 6))
                        py = int(radar_cy + dy * frac * (radar_r - 6))
                        # Colour by zone
                        if dist_m < DANGER_DIST:
                            dot_col = (0, 0, 255)      # red
                            dot_r = 6
                        elif dist_m < CAUTION_DIST:
                            dot_col = (0, 200, 255)    # orange
                            dot_r = 5
                        else:
                            dot_col = (0, 255, 0)      # green
                            dot_r = 4
                        cv2.circle(annotated, (px, py), dot_r, dot_col, -1)
                        # Distance label
                        cv2.putText(annotated, f"{dist_m:.1f}",
                                    (px - 12, py - dot_r - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    dot_col, 1, cv2.LINE_AA)

                # ── wheel-speed bars ──────────────────────────────────────
                bar_x     = fw - 130
                bar_top   = fh - 90
                bar_h     = 70
                bar_w     = 20
                gap       = 10
                cv2.rectangle(annotated, (bar_x, bar_top),
                              (bar_x + bar_w, bar_top + bar_h), (60, 60, 60), -1)
                cv2.rectangle(annotated, (bar_x + bar_w + gap, bar_top),
                              (bar_x + 2 * bar_w + gap, bar_top + bar_h), (60, 60, 60), -1)
                mid_y = bar_top + bar_h // 2
                for i, pwm in enumerate([st['left_pwm'], st['right_pwm']]):
                    col = state_color
                    frac = max(-1.0, min(1.0, pwm / 255.0))
                    fill_h = int(abs(frac) * (bar_h // 2))
                    bx = bar_x + i * (bar_w + gap)
                    if pwm >= 0:
                        cv2.rectangle(annotated, (bx, mid_y - fill_h),
                                      (bx + bar_w, mid_y), col, -1)
                    else:
                        cv2.rectangle(annotated, (bx, mid_y),
                                      (bx + bar_w, mid_y + fill_h), col, -1)
                cv2.putText(annotated, "L", (bar_x + 4, bar_top - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated, "R", (bar_x + bar_w + gap + 4, bar_top - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(annotated, (bar_x - 2, mid_y),
                         (bar_x + 2 * bar_w + gap + 2, mid_y), (150, 150, 150), 1)

            if args.show:
                cv2.imshow("YOLO (Raspberry Pi)", annotated)
                if args.show_disparity and disparity_vis is not None:
                    cv2.imshow("Disparity", disparity_vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                # Obstacle string (if motor controller provides readings)
                obs_str = ""
                if status_provider is not None:
                    obs = status_provider().get("obstacles", {})
                    if obs:
                        obs_str = (f"  obs=[F={obs.get('front',0):.2f} "
                                   f"L={obs.get('left',0):.2f} "
                                   f"R={obs.get('right',0):.2f} "
                                   f"B={obs.get('back',0):.2f}]")

                # Per-frame vision log (mirrors [ctrl] cadence from motor_ctrl)
                if target_det is not None:
                    mode = "bbox" if target_det["dist_is_fallback"] else "stereo"
                    bbox_w = int(target_det["x2"] - target_det["x1"])
                    bbox_h = int(target_det["y2"] - target_det["y1"])
                    others = [d for d in detections if d is not target_det]
                    others_str = (", ".join(
                        f"#{d['display_id']}@{d['distance_m']:.1f}m"
                        for d in others
                    ) if others else "-")
                    print(f"[vision] n={len(detections)} "
                          f"target=#{target_det['display_id']} "
                          f"dist={target_det['distance_m']:.2f}m[{mode}] "
                          f"angle={target_det['angle_deg']:+.1f}° "
                          f"off={last_known_offset:+.2f} "
                          f"bbox={bbox_w}x{bbox_h} "
                          f"conf={target_det['conf']:.2f} "
                          f"others=[{others_str}]"
                          f"{obs_str}")
                else:
                    status = "LOST" if loss_tracker.lost else "searching"
                    print(f"[vision] n={len(detections)} target=none "
                          f"[{status}]{obs_str}")

                # FPS heartbeat once per second
                now_sec = int(now)
                if now_sec != last_logged_sec:
                    print(f"[fps  ] {fps:.1f} Hz  frame={frame_idx}  "
                          f"{lock_str}{obs_str}")
                    last_logged_sec = now_sec
    finally:
        cap.release()
        if cap_right is not None:
            cap_right.release()
        cv2.destroyAllWindows()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
