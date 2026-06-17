# Badminton Scorekeeper

Reconstruction of the "Girsta" AI badminton demo: track players + shuttlecock,
overlay tracking, measure shot speed, keep score. Full plan in [`docs/PLAN.md`](docs/PLAN.md).

Compute: **Kaggle** free GPU (T4/P100). Input: an **online badminton match** pulled
with `yt-dlp` (no manual upload). Local CPU works only for short tests.

## Layout
```
badminton-scorekeeper/
  requirements.txt
  setup_tracknet.sh        # clone TrackNetV3 + fetch weights (Stage 3)
  src/pipeline.py          # Stage 1: players + tracking + overlay
  src/calibrate_court.py   # Stage 2: court homography (pixels -> metres)
  src/shuttle_tracker.py   # Stage 3: shuttlecock tracking via TrackNetV3
  docs/PLAN.md             # full build plan
```

## Status
- [x] Stage 1: player detection + tracking + overlay
- [x] Stage 2: court calibration (homography) — `CourtMapper` ready for speed
- [x] Stage 3: shuttle tracking (TrackNetV3 adapter) — `ShuttleTracker` / `shuttle.json`
- [ ] Stage 4: shot speed (uses `CourtMapper.speed_kmh`)
- [ ] Stage 5: scoring

## Run on Kaggle

1. **New Notebook** → Settings → Accelerator = **GPU T4**, and **Internet = ON**
   (Internet is required for `yt-dlp` and weight downloads).

```python
# Cell 1 — deps
!pip -q install "ultralytics>=8.3.0" "supervision>=0.25.0" yt-dlp
```

```python
# Cell 2 — grab an online badminton match and trim a short clip to iterate fast
# Replace URL with any public badminton match (search YouTube "badminton full match").
URL = "https://www.youtube.com/watch?v=XXXXXXXXXXX"
!yt-dlp -f "mp4[height<=720]" -o /kaggle/working/match_full.mp4 "$URL"
# 30s clip from 1:00 keeps detection fast while we build:
!ffmpeg -y -ss 60 -t 30 -i /kaggle/working/match_full.mp4 \
        -an /kaggle/working/match.mp4 -loglevel error
print("clip ready: /kaggle/working/match.mp4")
```

Use a **fixed broadcast/side camera** clip (whole court visible, camera doesn't pan) —
calibration assumes the court doesn't move.

```python
# Cell 3 — clone the repo and work from inside it
!git clone https://github.com/sheikhBasit/badminton-scorekeeper
%cd badminton-scorekeeper
# all later cells assume the repo root as the working dir
```

### Stage 1 — players + overlay
```python
!python src/pipeline.py --source /kaggle/working/match.mp4 \
        --output /kaggle/working/out.mp4 --model yolo11n.pt
from IPython.display import Video; Video("/kaggle/working/out.mp4", embed=True, width=480)
```

### Stage 2 — court calibration
Headless on Kaggle, so first save a reference frame, read off the 4 outer-corner
pixel coords (TL, TR, BR, BL), then pass them:

```python
import cv2, matplotlib.pyplot as plt
cap = cv2.VideoCapture("/kaggle/working/match.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok, frame = cap.read(); cap.release()
plt.figure(figsize=(12, 7)); plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.grid(True); plt.title("read the 4 OUTER court corners: TL, TR, BR, BL"); plt.show()
```
```python
# plug the coords you read into --corners, order TL TR BR BL
!python src/calibrate_court.py --source /kaggle/working/match.mp4 --frame 0 \
        --corners "315,210 980,205 1180,690 110,700" --out /kaggle/working/court.npz
from IPython.display import Image
display(Image("court_corners.jpg"))    # corners on the frame
display(Image("court_topdown.jpg"))    # should be a clean rectangle if correct
```
If `court_topdown.jpg` looks like a skewed/curved court, re-read the corners and redo.
`court.npz` then feeds Stage 4 speed via `CourtMapper.load("court.npz")`.

### Stage 3 — shuttle tracking (TrackNetV3)
TrackNetV3 needs its repo + pretrained weights (one-time setup):
```python
!bash setup_tracknet.sh third_party/TrackNetV3
# then place TrackNet_best.pt + InpaintNet_best.pt in third_party/TrackNetV3/
# (weight links are in that repo's README; use gdown as shown in setup_tracknet.sh)
```
```python
!python src/shuttle_tracker.py --source /kaggle/working/match.mp4 \
        --repo third_party/TrackNetV3 --out /kaggle/working/shuttle.json
# -> shuttle.json: {frame: {x, y, visible}} with short gaps interpolated
```
`shuttle.json` + `court.npz` are the two inputs Stage 4 combines into shot speed
(`CourtMapper.speed_kmh(prev_xy, xy, dt)` per frame).

## Push this repo to GitHub (for `git clone` on Kaggle)
```bash
# from the repo root, after committing:
gh repo create badminton-scorekeeper --public --source . --remote origin --push
# or manually:
git remote add origin https://github.com/sheikhBasit/badminton-scorekeeper.git
git push -u origin main
```
Then in the Kaggle notebook: `!git clone https://github.com/sheikhBasit/badminton-scorekeeper`.

## Run locally (CPU test / interactive calibration)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision        # CPU build
python src/pipeline.py --source match.mp4 --output out.mp4
python src/calibrate_court.py --source match.mp4 --interactive --out court.npz
```
`--interactive` opens a window to click the 4 corners (desktop only, not Kaggle).
