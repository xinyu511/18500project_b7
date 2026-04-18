#!/usr/bin/env python3
"""
Ultrasonic obstacle sensor wrapper.

Reads three HC-SR04 sensors (front / left / right) directly
when get_readings() is called.

GPIO pins (BCM numbering):
  Front:  trigger=23, echo=24
  Left:   trigger=17, echo=27
  Right:  trigger=22, echo=10
"""

from gpiozero import DistanceSensor


MAX_RANGE_M = 2


class ObstacleSensors:
    """Wrapper around three ultrasonic distance sensors."""

    def __init__(
        self,
        front_pins: tuple[int, int] = (23, 24),
        left_pins:  tuple[int, int] = (17, 27),
        right_pins: tuple[int, int] = (22, 10),
    ):
        self._sensors = {
            "front": DistanceSensor(trigger=front_pins[0], echo=front_pins[1], max_distance=MAX_RANGE_M),
            "left":  DistanceSensor(trigger=left_pins[0],  echo=left_pins[1],  max_distance=MAX_RANGE_M),
            "right": DistanceSensor(trigger=right_pins[0], echo=right_pins[1], max_distance=MAX_RANGE_M),
        }

        # Verify sensors work
        for name, sensor in self._sensors.items():
            dist = sensor.distance
            print(f"[obstacle] {name}: {dist*100:.1f} cm")
        print("[obstacle] 3 sensors ready")

    def get_readings(self) -> dict[str, float]:
        return {name: sensor.distance for name, sensor in self._sensors.items()}

    def close(self) -> None:
        for sensor in self._sensors.values():
            sensor.close()
