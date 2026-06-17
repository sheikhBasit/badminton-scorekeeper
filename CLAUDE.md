# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A computer-vision pipeline that turns a badminton match video into an annotated
render: tracked players, shuttlecock trail, real-world shot speed, and a running
scoreboard. Built as 6 stages (numbered 1–5 + 7) that chain through a small set
of on-disk artifacts.

## Compute model (important)

Designed to run on **Kaggle GPU notebooks**, not locally. The canonical end-to-end
run is `docs/KAGGLE_RUN.md` (ordered, copy-paste cells). Local CPU works only for
short tests and numeric checks. There is **no test framework** — validation is done
with inline `python3 - <<'PY' ... PY` scripts that import from `src/` and assert on
synthetic inputs (the scoring rules and homography math are validated this way).

`numpy`, `cv2`, `supervision`, `torch`, `ultralytics` are expected to be importable.

## Architecture: artifacts are the contract between stages

The whole system is organized so that three things are computed **once** from the
video, and everything downstream is a pure function of those:

1. **players** — detected live during render (not persisted), `src/pipeline.py`
2. **`court.npz`** — court homography (Stage 2), `src/calibrate_court.py`
3. **`shuttle.json`** — per-frame shuttle (x,y,visible) (Stage 3), `src/shuttle_tracker.py`

Then:
- **Stage 4 speed** (`src/speed.py`) = f(`shuttle.json`, `court.npz`)
- **Stage 5 scoring** (`src/scoring.py`) = f(`shuttle.json`, `court.npz`)
- **Stage 7 demo** (`src/demo.py`) fuses players + speed + scoreboard into one render,
  **reusing functions imported from `speed.py` and `scoring.py`** — do not duplicate
  that logic; extend the source module and `demo.py` picks it up.

`src/calibrate_court.py` defines **`CourtMapper`**, the central abstraction used
everywhere downstream: `to_metres(px)`, `contains(px, margin_m)` (the court-region
player filter), `speed_kmh(a,b,dt)`, and `save`/`load` to the `.npz`. Court reference
is the doubles outer boundary, 6.10 m × 13.40 m (constants `COURT_W_M`, `COURT_L_M`).

The player court-filter lives in `pipeline.py` as `filter_to_court(det, mapper, margin)`
and is imported by `demo.py` — keep it there so both share one implementation.

## Per-stage commands

```bash
# Stage 1 — players + overlay (add --court court.npz to filter out crowd/officials)
python src/pipeline.py --source match.mp4 --output players.mp4 [--court court.npz --court-margin 2.0]

# Stage 2 — court homography. Headless (Kaggle): pass corners TL TR BR BL.
python src/calibrate_court.py --source match.mp4 --frame 0 --corners "x1,y1 x2,y2 x3,y3 x4,y4" --out court.npz
python src/calibrate_court.py --source match.mp4 --interactive --out court.npz   # desktop click mode
# verify court_topdown.jpg looks like a clean rectangle

# Stage 3 — shuttle (needs TrackNetV3 vendored first: bash setup_tracknet.sh third_party/TrackNetV3)
python src/shuttle_tracker.py --source match.mp4 --repo third_party/TrackNetV3 \
    --tracknet-ckpt ckpts/TrackNet_best.pt --inpaint-ckpt ckpts/InpaintNet_best.pt --out shuttle.json

# Stage 4 — speed (--unit mps|kmh)
python src/speed.py --source match.mp4 --shuttle shuttle.json --court court.npz --output speed.mp4

# Stage 5 — scoring (pass 1 auto-guesses winners -> edit rallies.json -> rerun with --winners-file)
python src/scoring.py --source match.mp4 --shuttle shuttle.json --court court.npz --output scored.mp4

# Stage 7 — full combined render (court filter on by default; --no-court-filter to disable)
python src/demo.py --source match.mp4 --shuttle shuttle.json --court court.npz --output demo.mp4
```

## TrackNetV3 integration (Stage 3)

`shuttle_tracker.py` does **not** reimplement the model — it shells out to the
vendored upstream `predict.py` (qaz812345/TrackNetV3) and parses its CSV
(`Frame, Visibility, X, Y`) into `{frame: ShuttlePoint}`. The repo is vendored into
`third_party/TrackNetV3/` (gitignored) by `setup_tracknet.sh`, which also downloads
the single Google-Drive weights zip into `ckpts/`. `ShuttleTracker` is an interface
so the detector is swappable.

Known integration gotchas (already handled in `docs/KAGGLE_RUN.md`):
- TrackNetV3 imports `pycocotools` and `parse`; install those, but **do not** install
  its `requirements.txt` (pins torch==1.10/numpy==1.22 and breaks a modern stack).
- It hard-codes `.cuda()`; fine on GPU, needs a `map_location='cpu'`/`.cpu()` patch on CPU.
- Re-encode source clips to clean H.264 yuv420p — `yt-dlp` section-cuts can read as 0 frames.

## Accuracy caveats to preserve (don't "fix" by removing the disclaimers)

- **Speed** uses a ground-plane homography but the shuttle flies above the plane, so
  it's a live-readout estimate, not certified. Outliers >~470 km/h are auto-rejected.
- **Scoring is semi-automatic**: rallies are detected automatically, but the per-rally
  winner is a heuristic (which court half the shuttle landed in). The intended workflow
  is editing `rallies.json` then re-rendering with `--winners-file`.

## Conventions

Stage scripts are standalone `argparse` CLIs reused as importable modules by later
stages. Tuning constants are module-level UPPERCASE near the top of each file
(e.g. `SMASH_MPS`, `MERGE_GAP`, `MIN_RALLY_FRAMES` in speed.py/scoring.py) — adjust
those for real-footage tuning rather than threading new flags everywhere.
