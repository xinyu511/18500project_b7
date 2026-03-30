#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi
exec python3 yolo_raspi_cam.py \
  --distance-mode stereo \
  --stereo-source sbs \
  --device /dev/video0 \
  --stereo-width 1280 \
  --stereo-height 480 \
  --baseline-m 0.06 \
  --hfov-deg 100 \
  --vfov-deg 67.5 \
  --stereo-algo sgbm \
  --disp-num 48 \
  --disp-block-size 5 \
  --stereo-proc-scale 0.5 \
  --disp-every 2 \
  --min-depth-m 0.20 \
  --max-depth-m 3.5 \
  --depth-scale 1.875 \
  --depth-offset 0.4375 \
  --imgsz 320 \
  --conf 0.35 \
  --show \
  --show-disparity \
  "$@"
