# PRD — Real-Time Badminton Scorekeeper

## Vision
A coach or player mounts a phone at the back of a badminton court. A second phone
(or tablet, or laptop browser) shows a live radar minimap, running scoreboard, and
shot speed — updated in real time as the rally unfolds. No manual tagging, no
post-processing, no installation. Works on any court with WiFi.

---

## Architecture

```
[Camera phone]
  │  MJPEG stream (HTTP multipart or WebSocket binary)
  ▼
[Kaggle / Colab notebook — GPU server]
  │  Frame buffer → YOLO (players) + TrackNetV3 (shuttle) + scoring/speed
  │  ngrok tunnel exposes the server to the internet
  ▼  WebSocket JSON stream  {players, shuttle, score, speed, rally}
[Display — mobile browser PWA]
  │  Canvas radar + scoreboard + speed HUD
  └─ No app store needed; works in Safari / Chrome on any phone
```

Using Kaggle/Colab as the GPU server means:
- Free T4/P100 GPU for ML inference
- ngrok (or similar tunnel) exposes a public URL from inside the notebook
- Camera phone and display phone only need internet; no local network required
- Zero infrastructure cost

---

## Phases

### Phase 1 — Streaming pipeline (server-side, pure Python)
Convert the existing batch pipeline into a stateful, frame-by-frame pipeline.

- `src/stream_pipeline.py` — rolling frame buffer, feeds YOLO + TrackNet3 frame-by-frame
- TrackNetV3 needs 3 consecutive frames: maintain a `deque(maxlen=3)` fed at 30fps
- Scoring state machine already stateful — wire `BadmintonMatch.award()` on rally end
- Output per frame: `{"frame": N, "players": [...], "shuttle": {x,y,vis}, "score": {...}, "speed_kmh": F, "rally_active": bool}`
- **No video file needed** — processes PIL/numpy frames directly from the stream

### Phase 2 — Server / API (Kaggle or Colab notebook)
FastAPI app + WebSocket, launched from inside the notebook, exposed via ngrok.

- `server/app.py` — FastAPI with two endpoints:
  - `POST /frame` — receives a JPEG frame, runs pipeline, returns JSON (HTTP polling fallback)
  - `WS /stream` — bidirectional WebSocket: receives frames, pushes JSON results
- `server/tunnel.py` — starts ngrok tunnel, prints public URL to notebook output
- Court calibration: first 30 frames run Hough line detection to auto-find court corners;
  fallback: client sends 4 tap coordinates on a still frame
- Notebook cell: `!pip install fastapi uvicorn pyngrok && python server/app.py`

### Phase 3 — Web app (mobile browser PWA)
Single HTML file served from the same FastAPI server. No build step, no app store.

- `webapp/index.html` — self-contained (vanilla JS + Canvas API)
- **Camera tab**: `getUserMedia({ video: { facingMode: 'environment' } })` → captures frames
  at 15–30fps → sends to server WebSocket
- **Display tab** (or same tab, different device): connects to server WS → renders:
  - Canvas radar (court lines + player dots + shuttle + trails) — port of `radar.py` logic to JS
  - Scoreboard bar (top): player names, score, server marker `*`
  - Speed HUD (bottom-left): km/h, fades after 2s — matches current radar.py behaviour
- PWA manifest + service worker so it installs on phone home screen
- Responsive: works portrait (score + radar stacked) or landscape (side-by-side)

### Phase 4 — Court auto-calibration
Replace manual corner-clicking with automatic detection at session start.

- Server grabs 10 frames on connect, runs `cv2.HoughLinesP` to detect court boundary lines
- Fits 4 corners from line intersections → passes to `CourtMapper`
- If confidence low: server sends a still frame to the web app; user taps 4 corners on screen
- Calibration saved per session (in-memory); re-calibrate button in the app

---

## What's Already Done

| Component | Status |
|---|---|
| YOLO + ByteTrack player detection + court filter | ✅ Done (`pipeline.py`) |
| Court homography (`CourtMapper`) | ✅ Done (`calibrate_court.py`) |
| TrackNetV3 shuttle tracking | ✅ Done (`shuttle_tracker.py`) |
| Speed calculation | ✅ Done (`speed.py`) |
| Scoring state machine | ✅ Done + verified (`scoring.py`) |
| Radar rendering logic | ✅ Done (`radar.py`) |
| Kaggle kernel infra + ngrok knowledge | ✅ Done |

---

## What Needs Building

| Item | Phase | Effort |
|---|---|---|
| `stream_pipeline.py` — frame-by-frame stateful pipeline | 1 | Medium |
| `server/app.py` — FastAPI + WebSocket | 2 | Medium |
| `server/tunnel.py` — ngrok setup | 2 | Small |
| `webapp/index.html` — camera + display + radar canvas | 3 | Large |
| Court Hough auto-calibration | 4 | Medium |

---

## Constraints & Assumptions

- **Camera**: fixed mount, full court in frame, 30fps. Handheld / zoomed won't work.
- **Latency target**: <800ms from real-world event to display update (1–2 rally frames of lag is acceptable; score updates are not latency-critical).
- **Shuttle at 30fps**: TrackNetV3 was trained on 30fps footage — same as our clip. Adequate for scoring; speed readings at 30fps are estimates (shuttle moves ~1m/frame at 200 km/h).
- **Kaggle/Colab GPU**: T4 can run yolo11n + TrackNetV3 at ~25–30fps. yolo11x is too slow for real-time; switch to yolo11n for streaming.
- **No offline mode in Phase 1–3**: always needs the Kaggle/Colab server running.
- **Singles only**: one-per-half player filter already implemented.

---

## Build Order

```
Phase 1 (stream_pipeline.py)
  → Phase 2 (server/app.py + tunnel)  ← user provides Kaggle/Colab setup method
    → Phase 3 (webapp/index.html)
      → Phase 4 (auto-calibration)
```

Start: Phase 1 — `src/stream_pipeline.py`.
