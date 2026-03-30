#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${HOME}/screen_recordings"
FPS="${FPS:-20}"
CRF="${CRF:-23}"
PRESET="${PRESET:-ultrafast}"
DURATION="${DURATION:-}"   # optional, e.g. DURATION=60 ./record_kms.sh

mkdir -p "$OUTDIR"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTFILE="${OUTDIR}/screen_${STAMP}.mp4"

if [[ -e /dev/dri/card1 ]]; then
    CARD="/dev/dri/card1"
elif [[ -e /dev/dri/card0 ]]; then
    CARD="/dev/dri/card0"
else
    echo "No DRM device found under /dev/dri."
    exit 1
fi

echo "Recording full screen with kmsgrab"
echo "DRM device: $CARD"
echo "Output: $OUTFILE"

CMD=(
    sudo ffmpeg
    -y
    -framerate "$FPS"
    -device "$CARD"
    -f kmsgrab -i -
    -vf 'hwdownload,format=bgr0'
    -c:v libx264
    -preset "$PRESET"
    -crf "$CRF"
)

if [[ -n "$DURATION" ]]; then
    CMD+=( -t "$DURATION" )
fi

CMD+=( "$OUTFILE" )

"${CMD[@]}"