# Full Kaggle run — all stages on GPU

Copy these cells into a Kaggle notebook **in order**.
Settings: **Accelerator = GPU T4**, **Internet = ON**.

The only manual step is reading the 4 court corners off a frame (Cell 5).

---

### Cell 1 — dependencies
```python
!pip -q install "ultralytics>=8.3.0" "supervision>=0.25.0" yt-dlp gdown
```

### Cell 2 — clone this repo
```python
!git clone https://github.com/sheikhBasit/badminton-scorekeeper
%cd badminton-scorekeeper
```

### Cell 3 — get a badminton clip (fixed full-court camera works best)
```python
# pull ~12s from a YouTube match via search; replace with a specific URL if you prefer
!yt-dlp "ytsearch1:badminton singles full match broadcast" \
    -f "mp4[height<=720]/best[height<=720]" \
    --download-sections "*60-72" --force-keyframes-at-cuts \
    -o /kaggle/working/raw.mp4
# IMPORTANT: re-encode to clean H.264 yuv420p. yt-dlp section-cuts can produce a
# container that cv2.VideoCapture / TrackNetV3 read as 0 frames. This fixes it.
!ffmpeg -y -i /kaggle/working/raw.mp4 -c:v libx264 -pix_fmt yuv420p -r 30 -an \
    /kaggle/working/match.mp4 -loglevel error
import cv2; c=cv2.VideoCapture("/kaggle/working/match.mp4"); n=0
while c.read()[0]: n+=1
print("match.mp4 frames readable by cv2:", n)   # must be > 0
```

### Cell 4 — Stage 1: players + overlay (sanity check)
```python
!python src/pipeline.py --source /kaggle/working/match.mp4 \
    --output /kaggle/working/players.mp4 --model yolo11n.pt
from IPython.display import Video; Video("/kaggle/working/players.mp4", embed=True, width=480)
```

### Cell 5 — read the 4 court corners (manual)
```python
import cv2, matplotlib.pyplot as plt
cap = cv2.VideoCapture("/kaggle/working/match.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok, frame = cap.read(); cap.release()
plt.figure(figsize=(13, 8)); plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.grid(True); plt.title("read OUTER court corners as x,y: TL, TR, BR, BL"); plt.show()
```

### Cell 6 — Stage 2: calibrate court (paste your corners)
```python
# order: top-left top-right bottom-right bottom-left, read from Cell 5
CORNERS = "315,210 980,205 1180,690 110,700"   # <-- EDIT to your frame
!python src/calibrate_court.py --source /kaggle/working/match.mp4 --frame 0 \
    --corners "{CORNERS}" --out /kaggle/working/court.npz
from IPython.display import Image
display(Image("court_topdown.jpg"))   # must look like a clean rectangle
```

### Cell 7 — Stage 3: shuttle tracking (TrackNetV3)
```python
# clone TrackNetV3 + download weights (single Google-Drive zip -> ckpts/)
!bash setup_tracknet.sh third_party/TrackNetV3
# TrackNetV3 imports pycocotools + parse but doesn't list them as pip names.
# DO NOT `pip install -r third_party/TrackNetV3/requirements.txt` -- it pins
# torch==1.10.0 / numpy==1.22.4 and will downgrade & break Kaggle's GPU stack.
# Install ONLY the two missing modules on top of Kaggle's existing torch:
!pip -q install pycocotools parse
```
```python
!python src/shuttle_tracker.py --source /kaggle/working/match.mp4 \
    --repo third_party/TrackNetV3 \
    --tracknet-ckpt ckpts/TrackNet_best.pt --inpaint-ckpt ckpts/InpaintNet_best.pt \
    --out /kaggle/working/shuttle.json
```

### Cell 8 — Stage 4: shot speed
```python
!python src/speed.py --source /kaggle/working/match.mp4 \
    --shuttle /kaggle/working/shuttle.json --court /kaggle/working/court.npz \
    --output /kaggle/working/speed.mp4 --unit mps
from IPython.display import Video; Video("/kaggle/working/speed.mp4", embed=True, width=480)
```

### Cell 9 — Stage 5: scoring (pass 1, auto-guess winners)
```python
!python src/scoring.py --source /kaggle/working/match.mp4 \
    --shuttle /kaggle/working/shuttle.json --court /kaggle/working/court.npz \
    --output /kaggle/working/scored.mp4
# inspect/edit rallies.json winners, then optionally rerun with --winners-file rallies.json
```

### Cell 10 — Stage 7: full combined demo
```python
!python src/demo.py --source /kaggle/working/match.mp4 \
    --shuttle /kaggle/working/shuttle.json --court /kaggle/working/court.npz \
    --output /kaggle/working/demo.mp4 --unit mps
from IPython.display import Video; Video("/kaggle/working/demo.mp4", embed=True, width=480)
```

---

## If something breaks
- **Cell 7 weights fail to download** (gdown quota): open the link in
  `third_party/TrackNetV3/README.md`, download the zip, upload it as a Kaggle
  dataset, and `unzip` it into `third_party/TrackNetV3/ckpts/`.
- **Cell 7 `predict.py` errors on an arg / CSV name** — TrackNetV3 may have changed.
  Check `python third_party/TrackNetV3/predict.py --help`; the adapter's CSV finder
  (`src/shuttle_tracker.py` `_find_csv`) accepts `<stem>_ball.csv`, `<stem>.csv`, or
  any `.csv` in the save dir — adjust if needed.
- **Crowd/umpire get boxed as players** — broadcast footage. The court filter is
  built in: `demo.py` filters to the court by default (it has `--court`), and
  `pipeline.py` filters when you pass `--court /kaggle/working/court.npz`. Tune
  `--court-margin` (metres) if real players get cut or crowd slips through.
- **`predict.py` CUDA error on a CPU box** — only if you run without a GPU.
  TrackNetV3 hard-codes `.cuda()`; on Kaggle's GPU it's fine. For CPU, patch the
  clone: `sed -i 's/\.cuda()/.cpu()/g; s/torch.load(\(args[^)]*\))/torch.load(\1, map_location="cpu")/' third_party/TrackNetV3/predict.py`.
- **`court_topdown.jpg` looks skewed** — re-read corners in Cell 5, redo Cell 6.
