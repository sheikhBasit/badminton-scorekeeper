"""
Stage 7: full demo render -- everything in one pass.

Combines all stages into a single output video (the Girsta-style render):
  - players: YOLOv11 + ByteTrack boxes / IDs / traces        (Stage 1)
  - shuttle: marker + trail + live speed + SMASH flag         (Stages 3-4)
  - scoreboard: rules-correct semi-automatic score overlay    (Stage 5)

Reuses the tested functions from speed.py and scoring.py -- no duplicated logic.

Prereqs (produced by earlier stages):
  court.npz      Stage 2  (python src/calibrate_court.py ...)
  shuttle.json   Stage 3  (python src/shuttle_tracker.py ...)

    python src/demo.py --source match.mp4 --shuttle shuttle.json \
        --court court.npz --output demo.mp4

Add --winners-file rallies.json to score with your corrected winners
(otherwise winners are auto-guessed from the shuttle's landing half).
"""
import argparse
import json
from collections import deque

import cv2
import supervision as sv
import torch
from ultralytics import YOLO

from calibrate_court import CourtMapper
from pipeline import filter_to_court
from speed import compute_speeds, segment_shots, shot_peak_lookup, load_shuttle, SMASH_MPS
from scoring import (load_all_shuttle, detect_rallies, guess_winner,
                     build_timeline, draw_scoreboard)

PERSON_CLASS_ID = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--shuttle", required=True, help="shuttle.json (Stage 3)")
    ap.add_argument("--court", required=True, help="court.npz (Stage 2)")
    ap.add_argument("--winners-file", help="rallies.json with corrected winners (Stage 5)")
    ap.add_argument("--output", default="demo.mp4")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--unit", choices=["mps", "kmh"], default="mps")
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--court-margin", type=float, default=2.0,
                    help="metres of slack around the court when filtering players")
    ap.add_argument("--no-court-filter", action="store_true",
                    help="disable filtering players to the court region")
    args = ap.parse_args()

    info = sv.VideoInfo.from_video_path(args.source)
    fps = args.fps or info.fps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] {info.width}x{info.height} @ {fps}fps, device={device}")

    mapper = CourtMapper.load(args.court)

    # --- Stage 4: per-frame speed + shots -------------------------------------
    speed_pts = load_shuttle(args.shuttle)
    if len(speed_pts) >= 2:
        per_frame, frames, court_y, speeds = compute_speeds(speed_pts, mapper, fps)
        shots = segment_shots(frames, court_y, speeds)
    else:
        per_frame, shots = {}, []
        print("[warn] too few shuttle points for speed")
    peak_lut = shot_peak_lookup(shots)

    # --- Stage 5: rallies + score timeline ------------------------------------
    if args.winners_file:
        with open(args.winners_file) as f:
            rallies = json.load(f)
    else:
        all_pts = load_all_shuttle(args.shuttle)
        rallies = detect_rallies(all_pts)
        for r in rallies:
            r["winner"] = guess_winner(r, mapper)
        with open("rallies.json", "w") as f:
            json.dump(rallies, f, indent=2)
        print(f"[info] auto-detected {len(rallies)} rallies -> rallies.json "
              "(edit winners + rerun with --winners-file to correct the score)")
    timeline, log = build_timeline(rallies, info.total_frames)

    # --- detectors + annotators (Stage 1) -------------------------------------
    model = YOLO(args.model)
    tracker = sv.ByteTrack()
    box = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=30)

    trail = deque(maxlen=20)
    run_max = [0.0]
    to_unit = (lambda v: v) if args.unit == "mps" else (lambda v: v * 3.6)
    unit_label = "m/s" if args.unit == "mps" else "km/h"

    def cb(frame, idx):
        # players
        det = sv.Detections.from_ultralytics(
            model(frame, conf=args.conf, device=device, verbose=False)[0])
        det = det[det.class_id == PERSON_CLASS_ID]
        if not args.no_court_filter:
            det = filter_to_court(det, mapper, args.court_margin)
        det = tracker.update_with_detections(det)
        out = box.annotate(frame.copy(), det)
        out = label_ann.annotate(out, det, [f"#{t}" for t in det.tracker_id])
        out = trace_ann.annotate(out, det)

        # shuttle + speed
        if idx in per_frame:
            x, y, v, _ = per_frame[idx]
            p = (int(x), int(y))
            trail.append(p)
            run_max[0] = max(run_max[0], v)
            for j in range(1, len(trail)):
                cv2.line(out, trail[j - 1], trail[j], (0, 220, 255), 2)
            cv2.circle(out, p, 7, (0, 0, 255), -1)
            cv2.circle(out, p, 9, (255, 255, 255), 2)
            cv2.putText(out, f"{to_unit(v):.1f} {unit_label}", (p[0] + 12, p[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if v >= SMASH_MPS and abs(v - peak_lut.get(idx, 0.0)) < 1e-6:
                cv2.putText(out, "SMASH!", (p[0] - 30, p[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            trail.clear()

        # speed HUD (bottom-left so it doesn't clash with the scoreboard up top)
        cur = per_frame.get(idx, (0, 0, 0.0, 0))[2]
        h = out.shape[0]
        cv2.rectangle(out, (15, h - 80), (300, h - 15), (0, 0, 0), -1)
        cv2.putText(out, f"shuttle: {to_unit(cur):5.1f} {unit_label}", (25, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, f"max:     {to_unit(run_max[0]):5.1f} {unit_label}", (25, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # scoreboard (top-center)
        snap = timeline.get(idx, timeline.get(info.total_frames - 1))
        out = draw_scoreboard(out, snap)
        return out

    sv.process_video(source_path=args.source, target_path=args.output, callback=cb)

    final = log[-1] if log else {"a": 0, "b": 0, "games_a": 0, "games_b": 0}
    peak = max((s["peak_mps"] for s in shots), default=0.0)
    print(f"[done] {args.output}")
    print(f"       rallies={len(rallies)}  score={final.get('a')}-{final.get('b')}"
          f"  games={final.get('games_a', 0)}-{final.get('games_b', 0)}"
          f"  peak={peak:.1f} m/s ({peak * 3.6:.1f} km/h)")


if __name__ == "__main__":
    main()
