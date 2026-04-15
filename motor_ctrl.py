#!/usr/bin/env python3
"""
SafeFollow motor controller — UGV02 HTTP driver.

Reads vision pipeline output (user_distance, x_offset, target_lost) and sends
JSON commands to the UGV02 ESP32 over HTTP Wi-Fi.

JSON command reference (Waveshare UGV02):
  {"T":1,"L":<left>,"R":<right>}   direct wheel PWM, range -255 to +255
                                    positive = forward, negative = backward

HTTP endpoint:
  GET http://<robot_ip>/js?json=<command>

Robot IP:
  AP mode (default): 192.168.4.1
  STA mode:          shown on OLED screen (ST line)
"""

import argparse
import json
import queue
import time
import threading

import requests

# ── following controller constants ────────────────────────────────────────────
TARGET_DIST_M  = 1.25   # desired following distance (metres)
DIST_TOLERANCE = 0.10   # dead-band: no correction if |error| < this (metres)

# PID gains for forward/backward (output in normalised [-1, +1])
KP_DIST = 0.6
KI_DIST = 0.05
KD_DIST = 0.1

# Proportional gain for steering (x_offset → turn contribution, normalised)
KP_STEER = 0.5

# PWM limits
MAX_PWM  = 255   # full PWM range
MIN_PWM  = 60    # minimum effective PWM — below this DC motors may stall

# If target is lost, rotate in place to search
SEARCH_PWM = 80  # rotation PWM during search (each wheel opposite sign)

# Depth noise filtering
DIST_EMA_ALPHA = 0.2   # EMA smoothing factor: lower = smoother but slower to react
                        # 0.2 means each new reading contributes 20% of the update


# ── helpers ───────────────────────────────────────────────────────────────────
def _deadband(pwm: float) -> int:
    """
    Convert a normalised [-1,+1] value to an integer PWM in [-MAX_PWM, +MAX_PWM].
    Values whose absolute magnitude would fall below MIN_PWM are snapped up to
    MIN_PWM so the motors actually turn (DC geared motors stall at very low PWM).
    Zero is preserved as zero (robot stays stopped).
    """
    raw = pwm * MAX_PWM
    if raw == 0.0:
        return 0
    magnitude = max(MIN_PWM, min(MAX_PWM, abs(raw)))
    return int(magnitude if raw > 0 else -magnitude)


class PID:
    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = 1.0):
        self.kp    = kp
        self.ki    = ki
        self.kd    = kd
        self.limit = output_limit
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def compute(self, error: float) -> float:
        now = time.time()
        dt  = max(now - self._prev_time, 1e-3)

        self._integral += error * dt
        derivative      = (error - self._prev_error) / dt

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = max(-self.limit, min(self.limit, output))

        self._prev_error = error
        self._prev_time  = now
        return output


class UGV02:
    """HTTP wrapper for the UGV02 ESP32 sub-controller."""

    def __init__(self, ip: str, timeout: float = 0.1):
        self._base_url  = f"http://{ip}/js"
        self._timeout   = timeout
        # Queue depth 1: always send the latest command, drop stale ones.
        # This prevents the vision loop from stalling behind a slow HTTP response.
        self._send_queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._send_loop, daemon=True)
        self._worker.start()
        print(f"[motor] HTTP target: {self._base_url}")

    def _send_loop(self) -> None:
        """Background thread — drains the send queue and issues HTTP requests."""
        while True:
            cmd = self._send_queue.get()
            payload = json.dumps(cmd, separators=(",", ":"))
            # Use params= so requests URL-encodes the JSON value, matching:
            #   curl -G --data-urlencode 'json=...' http://<ip>/js
            try:
                requests.get(self._base_url, params={"json": payload},
                             timeout=self._timeout)
            except Exception:
                pass  # fire-and-forget — robot sends no response

    def _send(self, cmd: dict) -> None:
        """Enqueue a command; if the queue is full, replace the pending item."""
        try:
            self._send_queue.put_nowait(cmd)
        except queue.Full:
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                pass
            self._send_queue.put_nowait(cmd)

    def set_wheels(self, left: int, right: int) -> None:
        """
        Send T:1 direct wheel PWM command.
        left, right: integer PWM in [-255, +255].
        """
        left  = max(-255, min(255, int(left)))
        right = max(-255, min(255, int(right)))
        self._send({"T": 1, "L": left, "R": right})

    def stop(self) -> None:
        """Immediate stop."""
        self._send({"T": 1, "L": 0, "R": 0})

    def close(self) -> None:
        self.stop()


class FollowController:
    """
    Converts vision output → UGV02 wheel PWM commands.

    State machine:
      FOLLOW : user visible — maintain TARGET_DIST_M and centre alignment
      SEARCH : user lost    — rotate slowly to re-acquire
      STOP   : obstacle / button (set externally via safety_stop())

    Normalised control scheme:
      forward ∈ [-1, +1]  →  scaled to PWM by _deadband()
      turn    ∈ [-1, +1]  →  differential added/subtracted per wheel

      left_pwm  = _deadband(forward - turn)
      right_pwm = _deadband(forward + turn)
    """

    def __init__(self, robot: UGV02):
        self.robot        = robot
        self.pid          = PID(KP_DIST, KI_DIST, KD_DIST, output_limit=1.0)
        self.state        = "FOLLOW"
        self.stopped      = False
        self._smooth_dist = None   # EMA-filtered distance; None until first reading

    def update(self, user_distance: float, x_offset: float, target_lost: bool) -> None:
        if self.stopped:
            self.robot.stop()
            return

        if target_lost:
            self._enter_search()
            return

        self.state = "FOLLOW"

        # EMA filter: dampen sudden depth spikes while tracking gradual movement.
        # On first reading after (re-)acquisition, seed the filter directly so
        # the robot doesn't lurch toward a stale smoothed value.
        if self._smooth_dist is None:
            self._smooth_dist = user_distance
        else:
            self._smooth_dist = (DIST_EMA_ALPHA * user_distance
                                 + (1.0 - DIST_EMA_ALPHA) * self._smooth_dist)

        dist_error = self._smooth_dist - TARGET_DIST_M

        # Dead-band: ignore tiny distance errors
        if abs(dist_error) < DIST_TOLERANCE:
            dist_error = 0.0
            self.pid.reset()

        # forward: positive → move toward target (target is farther than desired)
        forward = self.pid.compute(dist_error)

        # turn: negative x_offset (target left) → positive turn (turn left)
        turn = -KP_STEER * x_offset
        turn = max(-1.0, min(1.0, turn))

        left_pwm  = _deadband(forward - turn)
        right_pwm = _deadband(forward + turn)

        self.robot.set_wheels(left_pwm, right_pwm)
        print(f"[ctrl] FOLLOW  raw={user_distance:.2f}m  "
              f"smooth={self._smooth_dist:.2f}m  dist_err={dist_error:+.2f}m  "
              f"fwd={forward:+.3f}  turn={turn:+.3f}  "
              f"L={left_pwm:+d}  R={right_pwm:+d}")

    def _enter_search(self) -> None:
        if self.state != "SEARCH":
            self.state        = "SEARCH"
            self._smooth_dist = None   # reset filter so reacquisition seeds fresh
            self.pid.reset()
            print("[ctrl] SEARCH — rotating to find user")
        # Rotate in place: left wheel forward, right wheel backward
        self.robot.set_wheels(SEARCH_PWM, -SEARCH_PWM)

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
    p = argparse.ArgumentParser(description="SafeFollow UGV02 motor controller (HTTP)")
    p.add_argument("--ip",   default="192.168.4.1",
                   help="Robot IP address (AP mode: 192.168.4.1; STA mode: see OLED)")
    p.add_argument("--test", action="store_true",
                   help="Run a quick movement test sequence instead of follow mode")
    return p.parse_args()


def run_test(robot: UGV02) -> None:
    """Quick sanity-check: forward → stop → turn → stop."""
    print("Test: forward 1 s")
    robot.set_wheels(100, 100);         time.sleep(1.0)
    print("Test: stop 0.5 s")
    robot.stop();                        time.sleep(0.5)
    print("Test: rotate left 1 s")
    robot.set_wheels(-SEARCH_PWM, SEARCH_PWM); time.sleep(1.0)
    print("Test: stop")
    robot.stop()


def main() -> None:
    args  = parse_args()
    robot = UGV02(args.ip)

    if args.test:
        try:
            run_test(robot)
        finally:
            robot.close()
        return

    # ── integrate with vision pipeline ────────────────────────────────────────
    # In the full system connect via a multiprocessing.Queue shared with
    # yolo_person_tracker.py, e.g.:
    #
    #   while True:
    #       vision = vision_queue.get()
    #       ctrl.update(
    #           user_distance = vision["user_distance"],
    #           x_offset      = vision["x_offset"],
    #           target_lost   = vision["target_lost"],
    #       )
    #
    ctrl = FollowController(robot)

    print("Motor controller ready. Ctrl-C to stop.")
    try:
        # Simulate: target at 2.0 m, centred, for 5 s
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
