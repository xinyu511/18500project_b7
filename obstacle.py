#!/usr/bin/env python3
"""
Ultrasonic obstacle sensor wrapper.

Reads three HC-SR04 sensors (front / left / right) in a background
thread and exposes the latest cached readings via get_readings().

GPIO pins (BCM numbering):
  Front:  trigger=23, echo=24
  Left:   trigger=17, echo=27
  Right:  trigger=22, echo=10
"""

import threading
import time


# Sensor polling interval — 50 ms gives ~20 Hz per sensor,
# fast enough for 8-15 FPS vision loop without hogging the CPU.
POLL_INTERVAL = 0.05

# Default max range in metres (matches HC-SR04 practical limit).
MAX_RANGE_M = 2.0


class ObstacleSensors:
    """Thread-safe wrapper around three ultrasonic distance sensors."""

    def __init__(
        self,
        front_pins: tuple[int, int] = (23, 24),
        left_pins:  tuple[int, int] = (17, 27),
        right_pins: tuple[int, int] = (22, 10),
        max_range_m: float = MAX_RANGE_M,
    ):
        try:
            from gpiozero import DistanceSensor  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "gpiozero is required for ultrasonic sensors. "
                "Install with:  pip install gpiozero"
            )

        self._sensors = {
            "front": DistanceSensor(
                trigger=front_pins[0], echo=front_pins[1],
                max_distance=max_range_m,
            ),
            "left": DistanceSensor(
                trigger=left_pins[0], echo=left_pins[1],
                max_distance=max_range_m,
            ),
            "right": DistanceSensor(
                trigger=right_pins[0], echo=right_pins[1],
                max_distance=max_range_m,
            ),
        }
        self._max = max_range_m

        # Cached readings (metres).  Initialised to max range (= clear).
        self._lock = threading.Lock()
        self._readings = {k: max_range_m for k in self._sensors}

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"[obstacle] 3 sensors initialised (max {max_range_m:.1f} m)")

    def _poll_loop(self) -> None:
        """Continuously read all sensors and cache the results."""
        while True:
            for name, sensor in self._sensors.items():
                try:
                    dist = sensor.distance  # metres, blocks ~5-12 ms
                except Exception:
                    dist = self._max  # treat read errors as "clear"
                with self._lock:
                    self._readings[name] = dist
            time.sleep(POLL_INTERVAL)

    def get_readings(self) -> dict[str, float]:
        """Return a snapshot of all three sensor distances in metres."""
        with self._lock:
            return dict(self._readings)

    def close(self) -> None:
        for sensor in self._sensors.values():
            sensor.close()
