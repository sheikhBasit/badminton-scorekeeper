"""
Stage 1: player detection + tracking + overlay.

Runs YOLOv11 on each frame, keeps only people, tracks them with ByteTrack,
and writes an annotated video (boxes + IDs + motion traces).

Usage:
    python src/pipeline.py --source input.mp4 --output out.mp4
    python src/pipeline.py --source input.mp4 --output out.mp4 --model yolo11x.pt

Shuttle tracking, court homography, speed and scoring are added in later stages.
"""
import argparse

import supervision as sv
import torch
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO 'person'


def build_annotators():
    return (
        sv.BoxAnnotator(thickness=2),
        sv.LabelAnnotator(text_scale=0.5, text_thickness=1),
        sv.TraceAnnotator(thickness=2, trace_length=30),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input video path")
    ap.add_argument("--output", required=True, help="annotated output video path")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights (n/s/m/l/x)")
    ap.add_argument("--conf", type=float, default=0.3, help="detection confidence")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")
    if device == "cpu":
        print("[warn] no GPU detected — fine for a short test, slow for full video.")

    model = YOLO(args.model)
    tracker = sv.ByteTrack()
    box, label, trace = build_annotators()

    info = sv.VideoInfo.from_video_path(args.source)
    print(f"[info] {info.width}x{info.height} @ {info.fps}fps, {info.total_frames} frames")

    def callback(frame, index):
        result = model(frame, conf=args.conf, device=device, verbose=False)[0]
        det = sv.Detections.from_ultralytics(result)
        det = det[det.class_id == PERSON_CLASS_ID]
        det = tracker.update_with_detections(det)

        labels = [f"#{tid}" for tid in det.tracker_id]
        out = box.annotate(frame.copy(), det)
        out = label.annotate(out, det, labels)
        out = trace.annotate(out, det)
        return out

    sv.process_video(
        source_path=args.source,
        target_path=args.output,
        callback=callback,
    )
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
