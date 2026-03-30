#!/usr/bin/env python3
"""
SafeFollow motor controller — UGV02 serial driver.

Reads vision pipeline output and sends JSON commands to the UGV02 ESP32 over UART/USB serial.

JSON command reference (Waveshare UGV02):
  {"T":1,"L":<left>,"R":<right>}       direct wheel speeds, range -0.5 to +0.5
  {"T":13,"X":<linear>,"Z":<angular>}  linear m/s + angular rad/s
  {"T":0}                               emergency stop (gimbal stop — also stops base)

Serial port:
  UART via 40-pin:  /dev/ttyAMA0  (default)
  USB cable:        /dev/ttyUSB0
"""

import argparse
import json
import time
import threading

import serial

# ── following controller constants ────────────────────────────────────────────
TARGET_DIST_M   = 1.25    # desired following distance (metres)
DIST_TOLERANCE  = 0.10    # dead-band: no correction if error < this (metres)

# PID gains for forward/backward speed
KP_DIST = 0.6
KI_DIST = 0.05
KD_DIST = 0.1

# Proportional gain for steering (x_offset → angular velocity)
KP_STEER = 0.5

# Output clamps
MAX_LINEAR  = 0.35   # m/s  — stay well below 0.5 (max) for safety
MAX_ANGULAR = 0.60   # rad/s

# If target is lost, rotate slowly to search
SEARCH_ANGULAR = 0.25


class PID:
    def __init__(self, kp: float, ki: float, kd: float, output_limit: float):
        self.kp    = kp
        self.ki    = ki
        self.kd    = kd
        self.limit = output_limit
        self._integral  = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def compute(self, error: float) -> float:
        now = time.time()
        dt  = max(now - self._prev_time, 1e-3)

        self._integral  += error * dt
        derivative       = (error - self._prev_error) / dt

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = max(-self.limit, min(self.limit, output))

        self._prev_error = error
        self._prev_time  = now
        return output


class UGV02:
    """Thin serial wrapper for the UGV02 ESP32 sub-controller."""

    def __init__(self, port: str, baud: int = 115200):
        self._ser = serial.Serial(port, baudrate=baud, dsrdtr=None, timeout=1.0)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()

        # Start background reader so ESP32 feedback doesn't block writes
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        print(f"[motor] Serial opened: {port} @ {baud}")

    def _send(self, cmd: dict) -> None:
        payload = json.dumps(cmd, separators=(",", ":")) + "\n"
        with self._lock:
            self._ser.write(payload.encode())

    def _read_loop(self) -> None:
        while True:
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"[ugv02] {line}")
            except Exception:
                break

    def move(self, linear: float, angular: float) -> None:
        """
        Send T:13 velocity command.
        linear:  forward speed in m/s  (positive = forward)
        angular: rotation in rad/s     (positive = turn left)
        """
        linear  = max(-MAX_LINEAR,  min(MAX_LINEAR,  linear))
        angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, angular))
        self._send({"T": 13, "X": round(linear, 4), "Z": round(angular, 4)})

    def set_wheels(self, left: float, right: float) -> None:
        """
        Send T:1 direct wheel speed command.
        Range -0.5 to +0.5 per wheel.
        """
        left  = max(-0.5, min(0.5, left))
        right = max(-0.5, min(0.5, right))
        self._send({"T": 1, "L": round(left, 4), "R": round(right, 4)})

    def stop(self) -> None:
        """Immediate stop."""
        self._send({"T": 1, "L": 0, "R": 0})

    def close(self) -> None:
        self.stop()
        time.sleep(0.1)
        self._ser.close()


class FollowController:
    """
    Converts vision output → UGV02 motion commands.

    State machine matches the design report:
      FOLLOW : user visible, maintain 1.25 m distance and alignment
      SEARCH : user lost, rotate slowly to re-acquire
      STOP   : obstacle detected or button pressed (set externally)
    """

    def __init__(self, robot: UGV02):
        self.robot   = robot
        self.pid     = PID(KP_DIST, KI_DIST, KD_DIST, MAX_LINEAR)
        self.state   = "FOLLOW"
        self.stopped = False   # set True externally for safety stop

    def update(self, user_distance: float, x_offset: float, target_lost: bool) -> None:
        if self.stopped:
            self.robot.stop()
            return

        if target_lost:
            self._enter_search()
            return

        self.state = "FOLLOW"

        dist_error = user_distance - TARGET_DIST_M

        # Dead-band: don't creep forward/back for small errors
        if abs(dist_error) < DIST_TOLERANCE:
            dist_error = 0.0
            self.pid.reset()

        linear  = self.pid.compute(dist_error)
        angular = -KP_STEER * x_offset   # negative: positive offset → turn right

        self.robot.move(linear, angular)
        print(f"[ctrl] FOLLOW  dist_err={dist_error:+.2f}m  "
              f"linear={linear:+.3f}  angular={angular:+.3f}")

    def _enter_search(self) -> None:
        if self.state != "SEARCH":
            self.state = "SEARCH"
            self.pid.reset()
            print("[ctrl] SEARCH — rotating to find user")
        self.robot.move(0.0, SEARCH_ANGULAR)

    def safety_stop(self) -> None:
        """Call this when an obstacle is detected or stop button pressed."""
        self.stopped = True
        self.robot.stop()
        print("[ctrl] SAFETY STOP")

    def resume(self) -> None:
        self.stopped = False
        print("[ctrl] Resumed")


# ── CLI & standalone test ──────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafeFollow UGV02 motor controller")
    p.add_argument("--port",  default="/dev/ttyAMA0",
                   help="Serial port (UART: /dev/ttyAMA0, USB: /dev/ttyUSB0)")
    p.add_argument("--baud",  type=int, default=115200)
    p.add_argument("--test",  action="store_true",
                   help="Run a quick movement test sequence instead of follow mode")
    return p.parse_args()


def run_test(robot: UGV02) -> None:
    """Quick sanity-check: forward → stop → turn → stop."""
    print("Test: forward 1 s")
    robot.move(0.2, 0.0);  time.sleep(1.0)
    print("Test: stop 0.5 s")
    robot.stop();           time.sleep(0.5)
    print("Test: turn left 1 s")
    robot.move(0.0, 0.3);  time.sleep(1.0)
    print("Test: stop")
    robot.stop()


def main() -> None:
    args = parse_args()
    robot = UGV02(args.port, args.baud)

    if args.test:
        try:
            run_test(robot)
        finally:
            robot.close()
        return

    # ── integrate with vision pipeline ────────────────────────────────────────
    # Import here so motor_ctrl.py can also run standalone (--test mode).
    # In the full system, run both scripts and connect via a queue or shared state.
    ctrl = FollowController(robot)

    print("Motor controller ready. Ctrl-C to stop.")
    try:
        # Placeholder loop — replace this block with your IPC mechanism.
        # Example using a multiprocessing.Queue shared with yolo_person_tracker.py:
        #
        #   while True:
        #       vision = vision_queue.get()
        #       ctrl.update(
        #           user_distance = vision["user_distance"],
        #           x_offset      = vision["x_offset"],
        #           target_lost   = vision["target_lost"],
        #       )
        #
        # For now, simulate a static target at 2 m, centred, for 5 s:
        print("Simulating: target at 2.0 m for 5 s (robot should move forward)")
        for _ in range(25):
            ctrl.update(user_distance=2.0, x_offset=0.0, target_lost=False)
            time.sleep(0.2)
        robot.stop()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
