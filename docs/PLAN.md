# AI Badminton Scorekeeper — Build Plan

Goal: recreate the "Girsta" demo — from a match video, track the shuttlecock + players,
overlay tracking, keep score, and measure shot speed.

Source inspiration: Girsta / AI NEWS (Instagram @girsta). No public repo is linked to the
clip; it's built on the same open-source CV stack below (Roboflow Supervision + trackers).

## Stack

| Job | Tool | Notes |
|---|---|---|
| Player detection + tracking | YOLOv11 (Ultralytics) + ByteTrack | Pretrained on people, works out of the box |
| Shuttlecock tracking | TrackNetV3 — https://github.com/qaz812345/TrackNetV3 | Shuttle is tiny + motion-blurred; generic YOLO misses it. Needs GPU. |
| Annotation / tracking glue / speed | Roboflow Supervision — https://github.com/roboflow/supervision | Boxes, traces, labels + speed-estimation tutorial |
| Court → real-world mapping | OpenCV homography | Required for real speed (km/h) vs pixels/sec |

## Pipeline (4 stages)

1. **Detect + track**
   - YOLOv11 per frame → player boxes → ByteTrack for stable IDs.
   - TrackNetV3 → shuttle (x, y) per frame.

2. **Calibrate the court** (key trick for speed)
   - Click the 4 court corners once. Court is fixed 13.4 m × 6.1 m → build homography pixels→meters.
   ```python
   import cv2, numpy as np
   SRC = np.float32([tl, tr, br, bl])              # 4 court corners in the image
   DST = np.float32([[0,0],[6.1,0],[6.1,13.4],[0,13.4]])  # meters
   H, _ = cv2.findHomography(SRC, DST)
   ```

3. **Measure shot speed**
   ```python
   real = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0][0]
   speed_mps = np.linalg.norm(real - prev_real) * fps   # distance/frame * fps
   kmh = speed_mps * 3.6
   ```
   - Smooth over a few frames (shuttle estimate is noisy); report peak per rally.

4. **Keep score** (hardest part)
   - Detect rally end: shuttle stops / hits floor / leaves court polygon for N frames.
   - Decide winner from which half the shuttle landed in; increment a rules-aware
     state machine (rally-point to 21, win by 2).

## Minimal starting skeleton

```python
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack()
box, trace = sv.BoxAnnotator(), sv.TraceAnnotator()

def process(frame):
    det = sv.Detections.from_ultralytics(model(frame)[0])
    det = det[det.class_id == 0]          # people only
    det = tracker.update_with_detections(det)
    # + TrackNet shuttle inference + homography speed + score state here
    out = box.annotate(frame.copy(), det)
    return trace.annotate(out, det)

sv.process_video("match.mp4", "out.mp4", callback=lambda f, i: process(f))
```

## Difficulty / honesty

- 🟢 Player tracking + overlays — an afternoon (Supervision does it).
- 🟡 Shuttle tracking — TrackNetV3 works but needs setup/GPU; shuttle is the hardest object in sports CV.
- 🟡 Shot speed — easy once homography exists; calibration is the fiddly bit.
- 🔴 Automatic scoring — research-grade. Viral demos usually approximate (fixed cam,
  manual point triggers, simple heuristics). Not umpire-grade.

## Pragmatic shortcut

Do players + shuttle + speed properly; make scoring semi-automatic (auto rally reset,
tap a key for who won). ~90% of the visual wow for ~10% of the effort.

## Reference repos

- https://github.com/roboflow/supervision — core CV toolkit (start here)
- https://github.com/qaz812345/TrackNetV3 — shuttlecock tracking
- https://github.com/muhammadyasin79/Badminton_Analytics_Project — YOLO-pose shuttle + trajectory + speed
- https://github.com/ToanNguyenKhanh/Badminton-Analysis — player + shuttle detection

## Build order (start here)

1. [x] Scaffold repo: `requirements.txt`, `src/pipeline.py`, `src/calibrate_court.py`
2. [x] Get YOLOv11 + ByteTrack + Supervision running (players + overlay) — `src/pipeline.py`
3. [x] Court calibration script (4 corners → save homography) — `src/calibrate_court.py`
4. [x] Integrate TrackNetV3 for shuttle tracking — `src/shuttle_tracker.py` + `setup_tracknet.sh`
5. [x] Add speed estimation via homography — `src/speed.py` (per-frame speed, shot segmentation, overlay)
6. [ ] Add semi-automatic scoring state machine
7. [ ] Polish overlay (score box, speed readout, traces)

Repo: `/home/aoi/Desktop/mnt/muaaz/badminton-scorekeeper/`
Compute: Kaggle GPU. Source: online badminton match via `yt-dlp` (see README), not a local file.
