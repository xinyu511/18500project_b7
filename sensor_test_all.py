#!/usr/bin/env python3
"""Test all three ultrasonic sensors simultaneously. Prints readings every 0.3s."""

from gpiozero import DistanceSensor
from time import sleep

sensors = {
    "Front": DistanceSensor(trigger=23, echo=24, max_distance=2),
    "Left":  DistanceSensor(trigger=17, echo=27, max_distance=2),
    "Right": DistanceSensor(trigger=22, echo=10, max_distance=2),
}

print("Reading all 3 sensors. Ctrl-C to stop.\n")
try:
    while True:
        parts = []
        for name, sensor in sensors.items():
            cm = round(sensor.distance * 100, 1)
            parts.append(f"{name}: {cm:6.1f} cm")
        print("  |  ".join(parts))
        sleep(0.3)
except KeyboardInterrupt:
    print("\nDone.")
