#!/usr/bin/env python3
"""
SafeFollow entry point.

Wires the YOLO vision pipeline, ultrasonic obstacle sensors, and the UGV02
motor controller together:

  yolo_raspi_cam.run_pipeline()
      → on_vision(user_distance, x_offset, target_lost)
          → FollowController.update()  (reads obstacle sensors internally)
              → UGV02._send()          (non-blocking HTTP)

Usage:
  python main.py --ip 192.168.4.1 [--show] [--no-sensors] [vision options...]

Run with --help to see all forwarded vision arguments.
"""

import argparse
import signal
import sys

from motor_ctrl import FollowController, UGV02
from yolo_raspi_cam import build_parser as vision_build_parser, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SafeFollow — vision + obstacle avoidance + motor controller",
        parents=[vision_build_parser()],
    )
    p.add_argument("--ip", default="192.168.4.1",
                   help="Robot IP (AP mode: 192.168.4.1; STA mode: see OLED)")

    # Obstacle sensors
    p.add_argument("--no-sensors", action="store_true",
                   help="Disable ultrasonic obstacle sensors (follow-only mode)")
    p.add_argument("--front-pins", type=int, nargs=2, default=[23, 24],
                   metavar=("TRIG", "ECHO"), help="Front sensor GPIO pins")
    p.add_argument("--left-pins",  type=int, nargs=2, default=[17, 27],
                   metavar=("TRIG", "ECHO"), help="Left sensor GPIO pins")
    p.add_argument("--right-pins", type=int, nargs=2, default=[22, 10],
                   metavar=("TRIG", "ECHO"), help="Right sensor GPIO pins")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    robot = UGV02(args.ip)

    # Obstacle sensors (optional)
    sensors = None
    if not args.no_sensors:
        try:
            from obstacle import ObstacleSensors
            sensors = ObstacleSensors(
                front_pins=tuple(args.front_pins),
                left_pins=tuple(args.left_pins),
                right_pins=tuple(args.right_pins),
            )
        except RuntimeError as e:
            print(f"[main] WARNING: {e}")
            print("[main] Running without obstacle sensors (--no-sensors)")

    ctrl = FollowController(robot, obstacle_sensors=sensors)

    def _shutdown(sig, frame):
        print("\n[main] Shutting down — stopping robot.")
        robot.stop()
        if sensors is not None:
            sensors.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    mode = "with obstacle avoidance" if sensors else "follow-only (no sensors)"
    print(f"[main] Robot IP: {args.ip}  |  Mode: {mode}")
    print("[main] Starting vision pipeline...")
    run_pipeline(args, on_vision=ctrl.update, status_provider=ctrl.get_status)


if __name__ == "__main__":
    main()
