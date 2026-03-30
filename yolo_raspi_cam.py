#!/usr/bin/env python3
import argparse
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO


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

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=[0],  # person only (COCO class id 0)
            verbose=False,
            device="cpu",
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
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
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
                label = f"person {conf:.2f} {dist_text} {angle_deg:+.1f}deg"
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
