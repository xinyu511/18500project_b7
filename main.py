#!/usr/bin/env python3
"""
SafeFollow entry point.

Wires the YOLO vision pipeline to the UGV02 motor controller:
  yolo_raspi_cam.run_pipeline()
      → on_vision callback (called each frame)
          → FollowController.update()
              → UGV02._send()   (non-blocking HTTP, background thread)

The vision pipeline locks onto one person by their stable display_id and
follows them; the motor controller handles PID distance control + steering,
enters SEARCH mode when the locked target is lost.

Usage:
  python main.py --ip 192.168.4.1 [--show] [vision options...]

Run with --help to see all forwarded vision arguments (camera, distance mode,
stereo baseline, etc.).
"""

import argparse
import signal
import sys

from motor_ctrl import FollowController, UGV02
from yolo_raspi_cam import build_parser as vision_build_parser, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SafeFollow — combined vision + motor controller",
        parents=[vision_build_parser()],   # inherit every vision argument
    )
    p.add_argument("--ip", default="192.168.4.1",
                   help="Robot IP (AP mode: 192.168.4.1; STA mode: see OLED)")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    robot = UGV02(args.ip)
    ctrl  = FollowController(robot)

    # Graceful shutdown: stop the robot on Ctrl-C or SIGTERM
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
