#!/usr/bin/env python3
"""
SafeFollow entry point.

Wires the YOLO vision pipeline to the UGV02 motor controller:
  yolo_person_tracker.run_pipeline()
      → on_vision callback
          → FollowController.update()
              → UGV02._send()  (non-blocking HTTP, background thread)

Usage:
  python main.py --ip 192.168.4.1 [--show] [vision options...]
"""

import argparse
import signal
import sys

from motor_ctrl import FollowController, UGV02
from yolo_person_tracker import parse_args as vision_parse_args, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SafeFollow — combined vision + motor controller",
        # Pass through unknown args to the vision arg parser
        parents=[vision_parse_args.__wrapped__]
        if hasattr(vision_parse_args, "__wrapped__") else [],
        add_help=False,
    )
    p.add_argument("--ip", default="192.168.4.1",
                   help="Robot IP (AP mode: 192.168.4.1; STA mode: see OLED)")
    p.add_argument("-h", "--help", action="help")
    # Re-declare all vision args so they appear in --help and are parsed together
    p.add_argument("--sdk-home",     default="")
    p.add_argument("--left-device",  default="/dev/video0")
    p.add_argument("--model",        default="yolo11n.pt")
    p.add_argument("--cam-width",    type=int,   default=640)
    p.add_argument("--cam-height",   type=int,   default=360)
    p.add_argument("--cam-fps",      type=int,   default=15)
    p.add_argument("--imgsz",        type=int,   default=320)
    p.add_argument("--conf",         type=float, default=0.40)
    p.add_argument("--focal-length", type=float, default=462.0)
    p.add_argument("--show",         action="store_true",
                   help="Display annotated camera feed")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    robot = UGV02(args.ip)
    ctrl  = FollowController(robot)

    # Graceful shutdown on Ctrl-C or SIGTERM
    def _shutdown(sig, frame):
        print("\n[main] Shutting down — stopping robot.")
        robot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[main] Robot IP: {args.ip}")
    print("[main] Starting vision pipeline...")

    run_pipeline(args, on_vision=ctrl.update)


if __name__ == "__main__":
    main()
