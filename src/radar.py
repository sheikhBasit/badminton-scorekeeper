"""
Top-down "radar" / minimap view.

Maps tracked player feet (and the shuttle) through the court homography onto a
clean bird's-eye court, with movement trails and the score. This is the tactical
panel view: one dot per player (near = blue, far = green), shuttle = yellow.

    python src/radar.py --source match.mp4 --court court.npz \
        [--shuttle shuttle.json] [--winners-file rallies.json] --output radar.mp4

Add --side-by-side to render the broadcast frame and the radar together.
"""
import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from calibrate_court import CourtMapper, COURT_W_M, COURT_L_M
from pipeline import filter_to_court, PERSON_CLASS_ID

PPM = 34          # radar pixels per court metre
MARGIN = 26       # border around the court (px)
TRAIL = 16        # player movement trail length (frames)
NEAR_COLOR = (255, 150, 40)   # BGR blue-ish  (bottom / near half)
FAR_COLOR = (60, 220, 60)     # BGR green     (top / far half)
SHUTTLE_COLOR = (60, 240, 255)  # BGR yellow


def court_dims():
    w = int(COURT_W_M * PPM) + 2 * MARGIN
    h = int(COURT_L_M * PPM) + 2 * MARGIN
    return w, h


def to_canvas(m):
    """metre (x,y) -> radar pixel (col,row)."""
    return (int(MARGIN + m[0] * PPM), int(MARGIN + m[1] * PPM))


def base_court():
    w, h = court_dims()
    img = np.full((h, w, 3), (35, 28, 24), np.uint8)               # dark surround
    tl, br = to_canvas((0, 0)), to_canvas((COURT_W_M, COURT_L_M))
    cv2.rectangle(img, tl, br, (70, 105, 70), -1)                  # court surface
    white = (235, 235, 235)

    def line(x1, y1, x2, y2, t=1):
        cv2.line(img, to_canvas((x1, y1)), to_canvas((x2, y2)), white, t)

    cv2.rectangle(img, tl, br, white, 2)                           # outer boundary
    line(0, COURT_L_M / 2, COURT_W_M, COURT_L_M / 2, 2)            # net
    line(0.46, 0, 0.46, COURT_L_M)                                 # singles sidelines
    line(COURT_W_M - 0.46, 0, COURT_W_M - 0.46, COURT_L_M)
    for sy in (COURT_L_M / 2 - 1.98, COURT_L_M / 2 + 1.98):        # short service lines
        line(0, sy, COURT_W_M, sy)
    line(COURT_W_M / 2, 0, COURT_W_M / 2, COURT_L_M / 2 - 1.98)    # centre lines
    line(COURT_W_M / 2, COURT_L_M / 2 + 1.98, COURT_W_M / 2, COURT_L_M)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--court", required=True)
    ap.add_argument("--shuttle", help="shuttle.json to also plot the shuttle")
    ap.add_argument("--winners-file", help="rallies.json for the score overlay")
    ap.add_argument("--output", default="radar.mp4")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--court-margin", type=float, default=2.0)
    ap.add_argument("--side-by-side", action="store_true")
    args = ap.parse_args()

    mapper = CourtMapper.load(args.court)
    info = sv.VideoInfo.from_video_path(args.source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] {info.width}x{info.height} @ {info.fps}fps, device={device}")

    shuttle = {}
    if args.shuttle:
        import json
        raw = json.load(open(args.shuttle))
        shuttle = {int(k): v for k, v in raw.items() if v.get("visible")}

    timeline = None
    if args.winners_file:
        import json
        from scoring import build_timeline
        rallies = json.load(open(args.winners_file))
        timeline, _ = build_timeline(rallies, info.total_frames)

    model = YOLO(args.model)
    tracker = sv.ByteTrack()
    trails = defaultdict(lambda: deque(maxlen=TRAIL))

    rw, rh = court_dims()
    out_w = info.width + rw if args.side_by_side else rw
    out_h = max(info.height, rh) if args.side_by_side else rh
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"avc1"),
                             info.fps, (out_w, out_h))
    if not writer.isOpened():  # fallback if avc1 unavailable
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (out_w, out_h))

    gen = sv.get_video_frames_generator(args.source)
    for idx, frame in enumerate(gen):
        radar = base_court()

        det = sv.Detections.from_ultralytics(
            model(frame, conf=args.conf, device=device, verbose=False)[0])
        det = det[det.class_id == PERSON_CLASS_ID]
        det = filter_to_court(det, mapper, args.court_margin)
        det = tracker.update_with_detections(det)

        if len(det):
            feet = np.column_stack([(det.xyxy[:, 0] + det.xyxy[:, 2]) / 2.0,
                                    det.xyxy[:, 3]])
            mets = mapper.to_metres(feet)
            for tid, m in zip(det.tracker_id, mets):
                p = to_canvas(m)
                trails[int(tid)].append(p)
                color = FAR_COLOR if m[1] < COURT_L_M / 2 else NEAR_COLOR
                tr = trails[int(tid)]
                for j in range(1, len(tr)):
                    cv2.line(radar, tr[j - 1], tr[j], color, 2)
                cv2.circle(radar, p, 9, color, -1)
                cv2.circle(radar, p, 11, (255, 255, 255), 1)

        if idx in shuttle:
            sp = mapper.to_metres([[shuttle[idx]["x"], shuttle[idx]["y"]]])[0]
            cv2.circle(radar, to_canvas(sp), 5, SHUTTLE_COLOR, -1)

        if timeline is not None:
            s = timeline.get(idx, timeline.get(info.total_frames - 1))
            txt = f"A {s['a']} - {s['b']} B"
            cv2.putText(radar, txt, (MARGIN, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)

        if args.side_by_side:
            canvas = np.zeros((out_h, out_w, 3), np.uint8)
            canvas[:info.height, :info.width] = frame
            canvas[:rh, info.width:info.width + rw] = radar
            writer.write(canvas)
        else:
            writer.write(radar)

    writer.release()
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
