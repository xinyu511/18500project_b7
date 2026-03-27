#!/usr/bin/env python3
import argparse
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
    parser.add_argument("--show", action="store_true", help="Show annotated preview window")
    return parser.parse_args()


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
            verbose=False,
            device="cpu",
        )

        annotated = results[0].plot()

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
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
