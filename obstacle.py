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
from time import sleep

from gpiozero import DistanceSensor


POLL_INTERVAL = 0.05   # seconds between polling cycles
MAX_RANGE_M   = 2      # max_distance passed to DistanceSensor


class ObstacleSensors:
    """Thread-safe wrapper around three ultrasonic distance sensors."""

    def __init__(
        self,
        front_pins: tuple[int, int] = (23, 24),
        left_pins:  tuple[int, int] = (17, 27),
        right_pins: tuple[int, int] = (22, 10),
    ):
        # Create sensors exactly like sensor_test_all.py
        self._sensors = {
            "front": DistanceSensor(trigger=front_pins[0], echo=front_pins[1], max_distance=MAX_RANGE_M),
            "left":  DistanceSensor(trigger=left_pins[0],  echo=left_pins[1],  max_distance=MAX_RANGE_M),
            "right": DistanceSensor(trigger=right_pins[0], echo=right_pins[1], max_distance=MAX_RANGE_M),
        }

        # Read each sensor once to verify they work
        self._lock = threading.Lock()
        self._readings = {}
        for name, sensor in self._sensors.items():
            dist = sensor.distance
            self._readings[name] = dist
            print(f"[obstacle] {name}: {dist*100:.1f} cm")

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print("[obstacle] 3 sensors ready")

    def _poll_loop(self) -> None:
        while self._running:
            for name, sensor in self._sensors.items():
                dist = sensor.distance
                with self._lock:
                    self._readings[name] = dist
            sleep(POLL_INTERVAL)

    def get_readings(self) -> dict[str, float]:
        with self._lock:
            return dict(self._readings)

    def close(self) -> None:
        self._running = False
        for sensor in self._sensors.values():
            sensor.close()
