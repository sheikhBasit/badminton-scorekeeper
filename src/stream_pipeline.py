"""
Phase 1: Real-time streaming pipeline.

Stateful, frame-by-frame version of the batch pipeline.  Feed it numpy frames
one at a time (from a camera, a WebSocket, or a video file) and get back a
structured dict per frame — ready to push over a WebSocket to the display app.

Usage (standalone test against a video file):

    python src/stream_pipeline.py --source samples/match_long.mp4 \
        --court samples/court_long.npz \
        --init-score 13,9 --first-server A \
        --name-a "AN S.Y." --name-b "WANG Z.Y."

Output (one JSON line per frame, printed to stdout):
    {"frame": 0, "players": [...], "shuttle": null, "score": {...}, "speed_kmh": null, "rally_active": false}
    ...
"""
import argparse
import json
import sys
from collections import deque

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from calibrate_court import CourtMapper, COURT_L_M
from pipeline import filter_to_court, PERSON_CLASS_ID
from scoring import BadmintonMatch

# ── tuning ────────────────────────────────────────────────────────────────────
RALLY_GAP_FRAMES = 45       # frames of shuttle-invisible before we call rally over
MIN_RALLY_FRAMES = 20       # ignore micro-spikes shorter than this
SPEED_WINDOW = 9            # shuttle frames in the rolling speed window
SPEED_MIN_KMH = 5           # below this → don't display (hallucination noise)
SPEED_MAX_KMH = 500         # above this → reject outlier
SPEED_DECAY_FRAMES = 60     # hide speed N frames after shuttle disappears


class StreamPipeline:
    """
    Stateful per-frame pipeline.  Call .process(frame, frame_idx) for each frame.
    All state (score, rally, speed history) is kept inside this object.
    """

    def __init__(self, mapper: CourtMapper, model_path: str = "yolo11n.pt",
                 conf: float = 0.25, court_margin: float = 0.4,
                 tracknet_repo: str = None, tracknet_ckpt: str = None,
                 inpaint_ckpt: str = None, fps: float = 30.0,
                 initial_score: dict = None, first_server: str = "A",
                 names: dict = None):

        self.mapper = mapper
        self.fps = fps
        self.names = names or {"A": "A", "B": "B"}

        # ── player detection ──────────────────────────────────────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(model_path)
        self.conf = conf
        self.court_margin = court_margin
        self.tracker = sv.ByteTrack()

        # ── shuttle (TrackNetV3) ──────────────────────────────────────────────
        # TrackNetV3 needs 3 consecutive frames; keep a rolling buffer.
        # If no repo path provided, shuttle tracking is disabled (players only).
        self._tracknet = None
        self._frame_buf: deque = deque(maxlen=3)   # (frame_idx, np.ndarray)
        if tracknet_repo:
            self._tracknet = _TrackNetInference(
                tracknet_repo, tracknet_ckpt, inpaint_ckpt, device)

        # ── speed ─────────────────────────────────────────────────────────────
        self._sh_history: deque = deque(maxlen=SPEED_WINDOW)  # (idx, x, y)
        self._last_speed: float = None
        self._last_sh_idx: int = -1

        # ── scoring ───────────────────────────────────────────────────────────
        self._match = BadmintonMatch(first_server=first_server,
                                     initial_score=initial_score)
        self._rally_frames: int = 0        # consecutive visible-shuttle frames
        self._invisible_frames: int = 0    # consecutive invisible frames
        self._rally_active: bool = False

    # ─────────────────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, frame_idx: int) -> dict:
        """
        Process one frame.  Returns a JSON-serialisable dict:
        {
          "frame": int,
          "players": [{"x_m": f, "y_m": f, "half": "near"|"far", "tid": int}, ...],
          "shuttle": {"x": f, "y": f, "x_m": f, "y_m": f} | null,
          "score":   {"a": int, "b": int, "server": "A"|"B", "game_no": int,
                      "name_a": str, "name_b": str},
          "speed_kmh": float | null,
          "rally_active": bool,
        }
        """
        result = {
            "frame": frame_idx,
            "players": [],
            "shuttle": None,
            "score": self._score_snap(),
            "speed_kmh": None,
            "rally_active": self._rally_active,
        }

        # ── players ───────────────────────────────────────────────────────────
        det = sv.Detections.from_ultralytics(
            self.model(frame, conf=self.conf, device=self.device, verbose=False)[0])
        det = det[det.class_id == PERSON_CLASS_ID]
        det = filter_to_court(det, self.mapper, self.court_margin)
        if len(det):
            feet_m = self.mapper.to_metres(
                np.column_stack([(det.xyxy[:, 0] + det.xyxy[:, 2]) / 2,
                                  det.xyxy[:, 3]]))
            areas = (det.xyxy[:, 2] - det.xyxy[:, 0]) * (det.xyxy[:, 3] - det.xyxy[:, 1])
            keep = []
            for half in (feet_m[:, 1] < COURT_L_M / 2, feet_m[:, 1] >= COURT_L_M / 2):
                idxs = np.where(half)[0]
                if len(idxs):
                    keep.append(idxs[np.argmax(areas[idxs])])
            if keep:
                det = det[np.array(keep)]
        det = self.tracker.update_with_detections(det)

        if len(det):
            feet = np.column_stack([(det.xyxy[:, 0] + det.xyxy[:, 2]) / 2,
                                     det.xyxy[:, 3]])
            mets = self.mapper.to_metres(feet)
            for tid, m in zip(det.tracker_id, mets):
                result["players"].append({
                    "tid": int(tid),
                    "x_m": round(float(m[0]), 3),
                    "y_m": round(float(m[1]), 3),
                    "half": "far" if m[1] < COURT_L_M / 2 else "near",
                })

        # ── shuttle ───────────────────────────────────────────────────────────
        self._frame_buf.append((frame_idx, frame))
        sh_visible = False

        if self._tracknet and len(self._frame_buf) == 3:
            sx, sy, vis = self._tracknet.predict(list(self._frame_buf))
            if vis:
                sh_visible = True
                sh_m = self.mapper.to_metres([[sx, sy]])[0]
                result["shuttle"] = {
                    "x": round(sx, 1), "y": round(sy, 1),
                    "x_m": round(float(sh_m[0]), 3),
                    "y_m": round(float(sh_m[1]), 3),
                }
                # speed
                self._sh_history.append((frame_idx, sx, sy))
                self._last_sh_idx = frame_idx
                if len(self._sh_history) >= 3:
                    fi, fx, fy = self._sh_history[0]
                    li, lx, ly = self._sh_history[-1]
                    dt = (li - fi) / self.fps
                    if dt > 0:
                        kmh = self.mapper.speed_kmh((fx, fy), (lx, ly), dt)
                        if SPEED_MIN_KMH <= kmh < SPEED_MAX_KMH:
                            self._last_speed = kmh

        # speed display (persists for SPEED_DECAY_FRAMES after last visible)
        if (self._last_speed is not None and
                frame_idx - self._last_sh_idx < SPEED_DECAY_FRAMES):
            result["speed_kmh"] = round(self._last_speed, 1)

        # ── rally state machine ───────────────────────────────────────────────
        if sh_visible:
            self._rally_frames += 1
            self._invisible_frames = 0
            if self._rally_frames >= MIN_RALLY_FRAMES and not self._rally_active:
                self._rally_active = True
        else:
            self._invisible_frames += 1
            if self._rally_active and self._invisible_frames >= RALLY_GAP_FRAMES:
                # rally ended — winner heuristic: last visible shuttle position
                self._end_rally(result)

        result["rally_active"] = self._rally_active
        result["score"] = self._score_snap()
        return result

    def award_point(self, winner: str):
        """Manually award a point (for UI override / line-judge call)."""
        self._match.award(winner)
        self._reset_rally()

    def _end_rally(self, result: dict):
        """Called when shuttle disappears for RALLY_GAP_FRAMES — guess winner."""
        # Use last known shuttle position to guess landing half
        if self._sh_history:
            _, lx, ly = self._sh_history[-1]
            lm = self.mapper.to_metres([[lx, ly]])[0]
            winner = "B" if lm[1] < COURT_L_M / 2 else "A"
        else:
            winner = "A"
        self._match.award(winner)
        self._reset_rally()

    def _reset_rally(self):
        self._rally_active = False
        self._rally_frames = 0
        self._invisible_frames = 0
        self._sh_history.clear()
        self._last_speed = None

    def _score_snap(self) -> dict:
        s = self._match.snapshot()
        return {
            "a": s["a"], "b": s["b"],
            "server": s["server"],
            "game_no": s["game_no"],
            "name_a": self.names["A"],
            "name_b": self.names["B"],
        }


# ── TrackNetV3 inference wrapper ──────────────────────────────────────────────

class _TrackNetInference:
    """
    Minimal wrapper around TrackNetV3's model for single-frame-at-a-time inference.
    Loads TrackNet + InpaintNet once, then runs predict() on demand.
    Requires TrackNetV3 vendored at tracknet_repo (via setup_tracknet.sh).
    """

    def __init__(self, repo: str, tracknet_ckpt: str, inpaint_ckpt: str, device: str):
        import sys as _sys
        _sys.path.insert(0, repo)
        from Model import TrackNet, InpaintNet  # noqa: F401 (TrackNetV3 module)
        import torch as _torch

        self.device = device
        self.H, self.W = 288, 512   # TrackNetV3 input resolution
        self.sigma = 2.5

        self.tracknet = TrackNet(in_dim=9, out_dim=3)
        self.tracknet.load_state_dict(
            _torch.load(tracknet_ckpt, map_location=device)["model"])
        self.tracknet.to(device).eval()

        self.inpaintnet = InpaintNet()
        self.inpaintnet.load_state_dict(
            _torch.load(inpaint_ckpt, map_location=device)["model"])
        self.inpaintnet.to(device).eval()

    def predict(self, frame_buf: list) -> tuple:
        """
        frame_buf: list of 3 (frame_idx, np.ndarray BGR) tuples.
        Returns (x, y, visible): shuttle pixel coords in the ORIGINAL frame
        resolution, or (0, 0, False) if not visible.
        """
        import torch as _torch

        frames = [cv2.resize(f, (self.W, self.H)) for _, f in frame_buf]
        # stack as (1, 9, H, W) — 3 frames × 3 channels
        imgs = np.concatenate(
            [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
             for f in frames], axis=0).astype(np.float32)
        inp = _torch.from_numpy(imgs).unsqueeze(0).to(self.device)

        with _torch.no_grad():
            heatmap = self.tracknet(inp)[0, -1].cpu().numpy()  # last frame heatmap

        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        conf = float(heatmap[y, x])
        if conf < 0.5:
            return 0.0, 0.0, False

        # scale back to original frame resolution
        _, orig = frame_buf[-1]
        oh, ow = orig.shape[:2]
        sx = x / self.W * ow
        sy = y / self.H * oh
        return float(sx), float(sy), True


# ── CLI test harness ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--court", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--court-margin", type=float, default=0.4)
    ap.add_argument("--tracknet-repo", default=None)
    ap.add_argument("--tracknet-ckpt", default=None)
    ap.add_argument("--inpaint-ckpt", default=None)
    ap.add_argument("--init-score", default=None, help="e.g. '13,9'")
    ap.add_argument("--first-server", default="A")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    mapper = CourtMapper.load(args.court)
    info = sv.VideoInfo.from_video_path(args.source)

    init = None
    if args.init_score:
        a, b = (int(x) for x in args.init_score.split(","))
        init = {"A": a, "B": b}

    pipeline = StreamPipeline(
        mapper=mapper,
        model_path=args.model,
        conf=args.conf,
        court_margin=args.court_margin,
        tracknet_repo=args.tracknet_repo,
        tracknet_ckpt=args.tracknet_ckpt,
        inpaint_ckpt=args.inpaint_ckpt,
        fps=info.fps,
        initial_score=init,
        first_server=args.first_server,
        names={"A": args.name_a, "B": args.name_b},
    )

    gen = sv.get_video_frames_generator(args.source)
    for idx, frame in enumerate(gen):
        if args.max_frames and idx >= args.max_frames:
            break
        result = pipeline.process(frame, idx)
        print(json.dumps(result), flush=True)

    print("[done]", file=sys.stderr)


if __name__ == "__main__":
    main()
