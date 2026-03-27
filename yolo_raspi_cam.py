#!/usr/bin/env python3
import argparse
import math
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiny YOLO on Raspberry Pi camera (/dev/video0)."
    )
    parser.add_argument("--device", default="/dev/video0", help="V4L2 camera device")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path or name")
    parser.add_argument("--cam-width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--cam-height", type=int, default=360, help="Camera capture height")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference image size")
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


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device: {args.device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    # YUYV is often hardware-friendly on Pi camera adapters.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    model = YOLO(args.model)

    prev_time = time.time()
    last_logged_sec = int(prev_time)
    fps = 0.0

    print("Running detection. Press 'q' in preview window to quit.")
    while True:
        ok, frame = cap.read()
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
        boxes = results[0].boxes
        person_count = 0
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                distance_m, angle_deg = estimate_distance_and_angle(
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
                person_count += 1

                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                label = f"person {conf:.2f} {distance_m:.2f}m {angle_deg:+.1f}deg"
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
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        else:
            # In headless mode, print a heartbeat every ~1 second.
            now_sec = int(now)
            if now_sec != last_logged_sec:
                print(f"FPS: {fps:.1f}")
                last_logged_sec = now_sec

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
