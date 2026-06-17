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

# Pretrained weights: a single zip on Google Drive -> unzips to ckpts/.
# (id from the TrackNetV3 README; update if upstream changes the link.)
CKPT_DIR="$DEST/ckpts"
if [ ! -f "$CKPT_DIR/TrackNet_best.pt" ]; then
  echo "[setup] downloading TrackNetV3 weights zip"
  gdown 1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA -O "$DEST/TrackNetV3_ckpts.zip"
  ( cd "$DEST" && unzip -o TrackNetV3_ckpts.zip )
fi

if [ -f "$CKPT_DIR/TrackNet_best.pt" ]; then
  echo "[setup] ready: $CKPT_DIR/{TrackNet_best.pt,InpaintNet_best.pt}"
  echo "        run: python src/shuttle_tracker.py --source match.mp4 \\"
  echo "             --repo $DEST --tracknet-ckpt ckpts/TrackNet_best.pt \\"
  echo "             --inpaint-ckpt ckpts/InpaintNet_best.pt --out shuttle.json"
else
  echo "[setup] WARN: weights not found. Download manually from the link in"
  echo "        $DEST/README.md, unzip into $CKPT_DIR/, then run shuttle_tracker.py"
fi
