from gpiozero import DistanceSensor
from time import sleep

sensor = DistanceSensor(
    echo=24,     # GPIO24
    trigger=23,  # GPIO23
    max_distance=2  # meters
)

while True:
    print("Distance:", round(sensor.distance * 100, 2), "cm")
    sleep(0.5)