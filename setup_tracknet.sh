#!/usr/bin/env bash
# Clone TrackNetV3 and fetch its pretrained weights (Stage 3 shuttle tracker).
# Run from the repo root. On Kaggle, run this in a notebook cell with leading '!'.
set -e

DEST="${1:-third_party/TrackNetV3}"
mkdir -p "$(dirname "$DEST")"

if [ ! -d "$DEST" ]; then
  echo "[setup] cloning TrackNetV3 -> $DEST"
  git clone https://github.com/qaz812345/TrackNetV3 "$DEST"
else
  echo "[setup] $DEST already exists, skipping clone"
fi

pip -q install gdown || true

# Pretrained weights are hosted on the TrackNetV3 GitHub Releases / Google Drive.
# Their README lists the current download links — check it if these IDs change:
#   https://github.com/qaz812345/TrackNetV3#-pretrained-weights
# Example (replace FILE_ID with the IDs from their README):
#   gdown --id <TRACKNET_FILE_ID> -O "$DEST/TrackNet_best.pt"
#   gdown --id <INPAINTNET_FILE_ID> -O "$DEST/InpaintNet_best.pt"

echo "[setup] clone done. Now place TrackNet_best.pt and InpaintNet_best.pt in $DEST"
echo "        (see weight links in $DEST/README.md), then run src/shuttle_tracker.py"
