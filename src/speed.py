"""
Stage 4: shot speed.

Combines the shuttle track (Stage 3, shuttle.json) with the court homography
(Stage 2, court.npz) to estimate real-world shuttle speed per frame, segment the
rally into individual shots, and render an overlay (live speed + per-shot peak,
plus a "SMASH!" flag on fast shots) like the original demo.

    python src/speed.py --source match.mp4 --shuttle shuttle.json \
        --court court.npz --output speed.mp4

Outputs:
    speed.mp4     annotated video (shuttle marker + trail + speed readout)
    speeds.csv    per-frame: frame, x_px, y_px, speed_mps, speed_kmh
    shots.json    detected shots with peak speed each

IMPORTANT accuracy caveat: a homography maps points that lie ON the court plane.
The shuttle flies ABOVE it, so its image point is back-projected to where the line
of sight meets the floor -- this over/under-estimates speed depending on shuttle
height and camera angle. Good enough for a live readout (this is what the demos do),
not for certified measurements. A side-on, zoomed, fixed camera minimises the error.
"""
import argparse
import csv
import json
from collections import deque

import cv2
import numpy as np
import supervision as sv

from calibrate_court import CourtMapper

# tuning
POS_SMOOTH_WIN = 3        # moving-average window on pixel positions
SPEED_SMOOTH_WIN = 5      # median window on per-step speed (odd)
MAX_PLAUSIBLE_MPS = 130   # ~470 km/h; above this is treated as detection noise
TRAIL_LEN = 20            # shuttle trail length (frames)
MIN_SHOT_TRAVEL_M = 1.0   # min court-length travel before a reversal counts as a new shot
SMASH_MPS = 50.0          # ~180 km/h; flashes "SMASH!" above this


def load_shuttle(path):
    """shuttle.json -> sorted list of (frame, x, y) for visible points only."""
    with open(path) as f:
        raw = json.load(f)
    pts = []
    for k, v in raw.items():
        if v.get("visible", True):
            pts.append((int(k), float(v["x"]), float(v["y"])))
    pts.sort()
    return pts


def moving_average(arr, win):
    if win <= 1 or len(arr) < win:
        return arr
    kernel = np.ones(win) / win
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def median_filter(arr, win):
    if win <= 1 or len(arr) < win:
        return arr
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.array([np.median(padded[i : i + win]) for i in range(len(arr))])


def compute_speeds(pts, mapper, fps):
    """Return per-point dict: frame -> (px, py, speed_mps, court_y_m)."""
    frames = np.array([p[0] for p in pts])
    xs = moving_average(np.array([p[1] for p in pts]), POS_SMOOTH_WIN)
    ys = moving_average(np.array([p[2] for p in pts]), POS_SMOOTH_WIN)

    metres = mapper.to_metres(np.column_stack([xs, ys]))  # (N,2) in court metres

    speeds = np.zeros(len(pts))
    for i in range(1, len(pts)):
        dt = (frames[i] - frames[i - 1]) / fps
        if dt <= 0:
            continue
        dist = float(np.linalg.norm(metres[i] - metres[i - 1]))
        v = dist / dt
        speeds[i] = v if v <= MAX_PLAUSIBLE_MPS else speeds[i - 1]
    speeds = median_filter(speeds, SPEED_SMOOTH_WIN)

    out = {}
    for i, fr in enumerate(frames):
        out[int(fr)] = (float(xs[i]), float(ys[i]), float(speeds[i]), float(metres[i, 1]))
    return out, frames, metres[:, 1], speeds


def segment_shots(frames, court_y, speeds):
    """Split the track into shots at direction reversals along court length."""
    shots, start = [], 0
    direction = 0  # +1 moving down-court, -1 up-court
    travel = 0.0
    for i in range(1, len(frames)):
        dy = court_y[i] - court_y[i - 1]
        step_dir = 1 if dy > 0 else (-1 if dy < 0 else direction)
        if direction == 0:
            direction = step_dir
        travel += abs(dy)
        if step_dir != direction and travel >= MIN_SHOT_TRAVEL_M:
            shots.append((start, i))
            start, direction, travel = i, step_dir, 0.0
    shots.append((start, len(frames) - 1))

    summary = []
    for a, b in shots:
        if b <= a:
            continue
        seg = speeds[a : b + 1]
        peak_i = int(a + np.argmax(seg))
        summary.append({
            "start_frame": int(frames[a]),
            "end_frame": int(frames[b]),
            "peak_frame": int(frames[peak_i]),
            "peak_mps": round(float(speeds[peak_i]), 1),
            "peak_kmh": round(float(speeds[peak_i]) * 3.6, 1),
        })
    return summary


def shot_peak_lookup(shots):
    """frame -> the running peak (mps) of the shot that frame belongs to."""
    lut = {}
    for s in shots:
        for f in range(s["start_frame"], s["end_frame"] + 1):
            lut[f] = s["peak_mps"]
    return lut


def render(source, output, per_frame, shots, unit):
    info = sv.VideoInfo.from_video_path(source)
    peak_lut = shot_peak_lookup(shots)
    trail = deque(maxlen=TRAIL_LEN)
    run_max = [0.0]
    to_unit = (lambda v: v) if unit == "mps" else (lambda v: v * 3.6)
    label = "m/s" if unit == "mps" else "km/h"

    def cb(frame, idx):
        out = frame
        if idx in per_frame:
            x, y, v, _ = per_frame[idx]
            p = (int(x), int(y))
            trail.append(p)
            run_max[0] = max(run_max[0], v)
            for j in range(1, len(trail)):
                cv2.line(out, trail[j - 1], trail[j], (0, 220, 255), 2)
            cv2.circle(out, p, 7, (0, 0, 255), -1)
            cv2.circle(out, p, 9, (255, 255, 255), 2)
            cv2.putText(out, f"{to_unit(v):.1f} {label}", (p[0] + 12, p[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if v >= SMASH_MPS and abs(v - peak_lut.get(idx, 0.0)) < 1e-6:
                cv2.putText(out, "SMASH!", (p[0] - 30, p[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            trail.clear()

        cv2.rectangle(out, (15, 15), (300, 95), (0, 0, 0), -1)
        cur = per_frame.get(idx, (0, 0, 0.0, 0))[2]
        cv2.putText(out, f"shuttle: {to_unit(cur):5.1f} {label}", (25, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, f"max:     {to_unit(run_max[0]):5.1f} {label}", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out

    sv.process_video(source_path=source, target_path=output, callback=cb)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input video")
    ap.add_argument("--shuttle", required=True, help="shuttle.json (Stage 3)")
    ap.add_argument("--court", required=True, help="court.npz (Stage 2)")
    ap.add_argument("--output", default="speed.mp4")
    ap.add_argument("--unit", choices=["mps", "kmh"], default="mps")
    ap.add_argument("--fps", type=float, default=0.0, help="override video fps")
    args = ap.parse_args()

    info = sv.VideoInfo.from_video_path(args.source)
    fps = args.fps or info.fps
    print(f"[info] {info.width}x{info.height} @ {fps}fps")

    pts = load_shuttle(args.shuttle)
    if len(pts) < 2:
        raise SystemExit("[error] not enough visible shuttle points to compute speed")
    mapper = CourtMapper.load(args.court)

    per_frame, frames, court_y, speeds = compute_speeds(pts, mapper, fps)
    shots = segment_shots(frames, court_y, speeds)

    with open("speeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x_px", "y_px", "speed_mps", "speed_kmh"])
        for fr in sorted(per_frame):
            x, y, v, _ = per_frame[fr]
            w.writerow([fr, round(x, 1), round(y, 1), round(v, 2), round(v * 3.6, 2)])
    with open("shots.json", "w") as f:
        json.dump(shots, f, indent=2)

    render(args.source, args.output, per_frame, shots, args.unit)

    peak = max((s["peak_mps"] for s in shots), default=0.0)
    print(f"[done] {len(pts)} shuttle pts, {len(shots)} shots, peak "
          f"{peak:.1f} m/s ({peak * 3.6:.1f} km/h)")
    print(f"       wrote {args.output}, speeds.csv, shots.json")


if __name__ == "__main__":
    main()
