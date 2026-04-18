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

# ── obstacle avoidance constants ─────────────────────────────────────────────
DANGER_DIST   = 0.40  # metres — emergency: full stop / hard turn
CAUTION_DIST  = 0.80  # metres — start slowing + steering around
REPULSION_K   = 0.6   # side-repulsion gain (0 = ignore sides, 1 = very strong)
AVOID_BIAS    = 0.7   # steering bias magnitude when front is blocked

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
    Converts vision output → UGV02 wheel PWM commands with obstacle avoidance.

    State machine:
      FOLLOW  : user visible, no dangerous obstacle — PID + steering
      AVOID   : user visible, obstacle in path — modified steering to go around
      SEARCH  : user lost — rotate slowly to re-acquire
      STOP    : boxed in on 3 sides, or external safety_stop()

    Obstacle avoidance (3 priority layers applied per frame):
      Layer 1  EMERGENCY   front < DANGER_DIST → hard turn or full stop
      Layer 2  AVOIDANCE   front < CAUTION_DIST → slow down + steer around
                           side  < CAUTION_DIST → repulsive steering bias
      Layer 3  BASE        PID distance + proportional steering (unmodified)
    """

    def __init__(self, robot: UGV02, obstacle_sensors=None):
        self.robot        = robot
        self.obstacles    = obstacle_sensors   # None = no avoidance
        self.pid          = PID(KP_DIST, KI_DIST, KD_DIST, output_limit=1.0)
        self.state        = "FOLLOW"
        self.stopped      = False
        self._smooth_dist = None
        self._search_dir  = 1   # +1 = left, -1 = right (flipped by obstacles)

        # Latest-command snapshot (overlay / logging)
        self.last_raw_dist    = 0.0
        self.last_smooth_dist = 0.0
        self.last_dist_err    = 0.0
        self.last_forward     = 0.0
        self.last_turn        = 0.0
        self.last_left_pwm    = 0
        self.last_right_pwm   = 0
        self.last_obs         = {"front": 2.0, "left": 2.0, "right": 2.0}

    # ── internal helpers ──────────────────────────────────────────────────

    def _record(self, left: int, right: int) -> None:
        self.last_left_pwm  = left
        self.last_right_pwm = right

    def _read_obstacles(self) -> dict[str, float]:
        if self.obstacles is None:
            return {"front": 2.0, "left": 2.0, "right": 2.0}
        readings = self.obstacles.get_readings()
        self.last_obs = readings
        return readings

    def _apply_avoidance(self, forward: float, turn: float,
                         x_offset: float, obs: dict) -> tuple[float, float, str]:
        """
        Modify forward/turn based on obstacle readings.
        Returns (forward, turn, state_label).

        Priority:
          1. EMERGENCY — front danger zone
          2. FRONT CAUTION — slow down + bias around
          3. SIDE REPULSION — push away from close walls
        """
        front = obs["front"]
        left  = obs["left"]
        right = obs["right"]
        state = "FOLLOW"

        # ── Layer 1: EMERGENCY ────────────────────────────────────────────
        if front < DANGER_DIST:
            if left < DANGER_DIST and right < DANGER_DIST:
                # Boxed in from three sides → full stop
                return 0.0, 0.0, "STOP"

            # Hard turn toward the clearest side
            forward = 0.0
            if left >= right:
                turn = +1.0
            else:
                turn = -1.0
            return forward, turn, "AVOID"

        # ── Layer 2: FRONT CAUTION — slow down + steer around ─────────────
        if front < CAUTION_DIST:
            state = "AVOID"
            # Scale forward proportionally: full at CAUTION, zero at DANGER
            scale = (front - DANGER_DIST) / (CAUTION_DIST - DANGER_DIST)
            forward *= max(0.0, scale)

            # Steer bias: prefer the side closest to the person so the robot
            # arcs TOWARD the target, not away from it.
            if x_offset < 0 and left > DANGER_DIST:
                # Person is left, left is clear → go left
                bias = +AVOID_BIAS
            elif x_offset >= 0 and right > DANGER_DIST:
                # Person is right, right is clear → go right
                bias = -AVOID_BIAS
            elif left >= right:
                bias = +AVOID_BIAS
            else:
                bias = -AVOID_BIAS
            turn += bias

        # ── Layer 3: SIDE REPULSION — always active ───────────────────────
        if right < CAUTION_DIST:
            # Right wall close → push left
            repulsion = REPULSION_K * (1.0 - right / CAUTION_DIST)
            turn += repulsion
            if state == "FOLLOW":
                state = "AVOID"

        if left < CAUTION_DIST:
            # Left wall close → push right
            repulsion = REPULSION_K * (1.0 - left / CAUTION_DIST)
            turn -= repulsion
            if state == "FOLLOW":
                state = "AVOID"

        # Clamp turn to [-1, +1]
        turn = max(-1.0, min(1.0, turn))

        return forward, turn, state

    # ── public API ────────────────────────────────────────────────────────

    def update(self, user_distance: float, x_offset: float, target_lost: bool) -> None:
        self.last_raw_dist = user_distance
        obs = self._read_obstacles()

        if self.stopped:
            self.state = "STOP"
            self.robot.stop()
            self._record(0, 0)
            return

        if target_lost:
            self._enter_search(obs)
            return

        # ── base follow logic (unchanged) ─────────────────────────────────
        if self._smooth_dist is None:
            self._smooth_dist = user_distance
        else:
            self._smooth_dist = (DIST_EMA_ALPHA * user_distance
                                 + (1.0 - DIST_EMA_ALPHA) * self._smooth_dist)

        dist_error = self._smooth_dist - TARGET_DIST_M
        if abs(dist_error) < DIST_TOLERANCE:
            dist_error = 0.0
            self.pid.reset()

        forward = self.pid.compute(dist_error)
        if forward < 0.0:
            forward = 0.0                    # never reverse — no rear camera
            self.pid.reset()                 # clear negative integral wind-up
        turn    = -KP_STEER * x_offset
        turn    = max(-1.0, min(1.0, turn))

        # ── apply obstacle avoidance ──────────────────────────────────────
        forward, turn, state = self._apply_avoidance(forward, turn, x_offset, obs)

        if state == "STOP":
            # Boxed in — emergency stop
            self.state = "STOP"
            self.robot.stop()
            self.last_smooth_dist = self._smooth_dist or 0.0
            self.last_dist_err    = dist_error
            self.last_forward     = 0.0
            self.last_turn        = 0.0
            self._record(0, 0)
            print(f"[ctrl] STOP   boxed in  F={obs['front']:.2f} "
                  f"L={obs['left']:.2f} R={obs['right']:.2f}")
            return

        self.state = state   # "FOLLOW" or "AVOID"

        left_pwm  = _deadband(forward - turn)
        right_pwm = _deadband(forward + turn)

        # ── final safety gate: refuse net-forward if front blocked ────────
        if obs["front"] < DANGER_DIST:
            # Allow pure rotation (one positive, one negative) but block
            # net-forward motion (both wheels positive).
            if left_pwm > 0 and right_pwm > 0:
                left_pwm = 0
                right_pwm = 0
            # Also block if one wheel drives strongly forward
            elif left_pwm > 0:
                left_pwm = 0
            elif right_pwm > 0:
                right_pwm = 0

        self.robot.set_wheels(left_pwm, right_pwm)
        self.last_smooth_dist = self._smooth_dist
        self.last_dist_err    = dist_error
        self.last_forward     = forward
        self.last_turn        = turn
        self._record(left_pwm, right_pwm)

        obs_str = (f"F={obs['front']:.2f} L={obs['left']:.2f} "
                   f"R={obs['right']:.2f}")
        print(f"[ctrl] {state:6s} raw={user_distance:.2f}m  "
              f"smooth={self._smooth_dist:.2f}m  err={dist_error:+.2f}m  "
              f"fwd={forward:+.3f}  turn={turn:+.3f}  "
              f"L={left_pwm:+d}  R={right_pwm:+d}  obs=[{obs_str}]")

    def _enter_search(self, obs: dict) -> None:
        if self.state != "SEARCH":
            self.state        = "SEARCH"
            self._smooth_dist = None
            self.pid.reset()
            print("[ctrl] SEARCH — rotating to find user")

        # Flip search direction if rotating toward an obstacle
        if self._search_dir > 0 and obs["left"] < DANGER_DIST:
            self._search_dir = -1
            print("[ctrl] SEARCH — flipped to clockwise (left blocked)")
        elif self._search_dir < 0 and obs["right"] < DANGER_DIST:
            self._search_dir = +1
            print("[ctrl] SEARCH — flipped to counter-clockwise (right blocked)")

        # If front is blocked during search, stop rotating and wait
        if obs["front"] < DANGER_DIST and obs["left"] < DANGER_DIST and obs["right"] < DANGER_DIST:
            self.robot.stop()
            self._record(0, 0)
            print(f"[ctrl] SEARCH STOP — boxed in  F={obs['front']:.2f} "
                  f"L={obs['left']:.2f} R={obs['right']:.2f}")
            return

        spd = SEARCH_PWM * self._search_dir
        self.robot.set_wheels(spd, -spd)
        self.last_forward  = 0.0
        self.last_turn     = 0.0
        self.last_dist_err = 0.0
        self._record(spd, -spd)

        obs_str = (f"F={obs['front']:.2f} L={obs['left']:.2f} "
                   f"R={obs['right']:.2f}")
        dir_label = "CCW" if self._search_dir > 0 else "CW"
        print(f"[ctrl] SEARCH {dir_label}  L={spd:+d} R={-spd:+d}  obs=[{obs_str}]")

    def safety_stop(self) -> None:
        self.stopped = True
        self.state   = "STOP"
        self.robot.stop()
        self._record(0, 0)
        print("[ctrl] SAFETY STOP")

    def resume(self) -> None:
        self.stopped = False
        print("[ctrl] Resumed")

    def get_status(self) -> dict:
        return {
            "state":      self.state,
            "stopped":    self.stopped,
            "raw_dist":   self.last_raw_dist,
            "smooth":     self.last_smooth_dist,
            "dist_err":   self.last_dist_err,
            "forward":    self.last_forward,
            "turn":       self.last_turn,
            "left_pwm":   self.last_left_pwm,
            "right_pwm":  self.last_right_pwm,
            "json_cmd":   {"T": 1, "L": self.last_left_pwm, "R": self.last_right_pwm},
            "obstacles":  dict(self.last_obs),
        }


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
