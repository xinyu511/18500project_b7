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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiny YOLO on Raspberry Pi camera (/dev/video0)."
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
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

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

    print("Running detection. Press 'q' in preview window to quit.")
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
                # Disparity is measured in the downscaled image pixel units.
                # Convert to full-resolution pixel disparity before depth conversion.
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
        person_count = 0
        detections: list[dict] = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                raw_track_id = int(box.id[0]) if box.id is not None else None
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                hist = person_color_histogram(frame, int(x1), int(y1), int(x2), int(y2))
                detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": conf,
                        "raw_track_id": raw_track_id,
                        "cx": cx,
                        "cy": cy,
                        "hist": hist,
                    }
                )

        display_ids = track_id_mapper.update(detections, frame_idx, frame_w, frame_h)

        for det, track_id in zip(detections, display_ids):
                x1 = det["x1"]
                y1 = det["y1"]
                x2 = det["x2"]
                y2 = det["y2"]
                conf = det["conf"]
                angle_deg = math.degrees(
                    math.atan(((0.5 * (x1 + x2)) - frame_w * 0.5) / focal_px_x)
                )
                if args.distance_mode == "stereo" and disparity_for_sampling is not None:
                    distance_m = sample_depth_from_disparity(
                        disparity_map=disparity_for_sampling,
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        focal_px=focal_px_x,
                        baseline_m=args.baseline_m,
                        min_depth_m=args.min_depth_m,
                        max_depth_m=args.max_depth_m,
                    )

                    if distance_m is not None:
                        distance_m = args.depth_scale * distance_m - args.depth_offset
                        if distance_m < args.min_depth_m or distance_m > args.max_depth_m:
                            distance_m = None

                    if distance_m is None:
                        distance_m, _ = estimate_distance_and_angle(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            frame_w=frame_w,
                            frame_h=frame_h,
                            hfov_deg=args.hfov_deg,
                            vfov_deg=args.vfov_deg,
                            person_height_m=args.person_height_m,
                        )
                        dist_text = f"~{distance_m:.2f}m"
                    else:
                        dist_text = f"{distance_m:.2f}m"
                else:
                    distance_m, _ = estimate_distance_and_angle(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        hfov_deg=args.hfov_deg,
                        vfov_deg=args.vfov_deg,
                        person_height_m=args.person_height_m,
                    )
                    dist_text = f"{distance_m:.2f}m"
                person_count += 1

                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                person_name = f"person {track_id}" if track_id is not None else "person"
                label = f"{person_name} {conf:.2f} {dist_text} {angle_deg:+.1f}deg"
                text_org = (p1[0], max(20, p1[1] - 8))
                cv2.putText(
                    annotated,
                    label,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}  persons: {person_count}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if args.show:
            cv2.imshow("YOLO (Raspberry Pi)", annotated)
            if args.show_disparity and disparity_vis is not None:
                cv2.imshow("Disparity", disparity_vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        else:
            # In headless mode, print a heartbeat every ~1 second.
            now_sec = int(now)
            if now_sec != last_logged_sec:
                print(f"FPS: {fps:.1f}")
                last_logged_sec = now_sec

    cap.release()
    if cap_right is not None:
        cap_right.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
