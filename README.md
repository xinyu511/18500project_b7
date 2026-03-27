# 18500project_b7

Tiny YOLO real-time detection for Raspberry Pi camera input (`/dev/video0`).

## What this uses
- `ultralytics` YOLO model
- OpenCV capture from V4L2
- Low default resolution for better FPS

## Quick start
1. Create and activate a venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run with tiny model + small input:
```bash
python3 yolo_raspi_cam.py \
  --device /dev/video0 \
  --model yolo11n.pt \
  --cam-width 640 \
  --cam-height 360 \
  --imgsz 320 \
  --conf 0.35 \
  --show
```

Press `q` to quit.

## Notes for Raspberry Pi performance
- Start with `--imgsz 320` (or `256` for more FPS).
- Keep camera capture size small (`640x360` or `424x240`).
- Use `yolo11n.pt` (nano) first; if too slow, lower `--imgsz`.
- If your camera feed fails, test first with:
```bash
ffplay -f v4l2 -input_format yuyv422 -video_size 1280x720 /dev/video0
```
