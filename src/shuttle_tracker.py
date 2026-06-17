"""
Stage 3: shuttlecock tracking via TrackNetV3.

The shuttle is tiny and motion-blurred, so a generic detector misses it.
TrackNetV3 (qaz812345/TrackNetV3) is a heatmap tracker purpose-built for it.

Rather than reimplement their model, we drive their `predict.py` as a subprocess
and parse its CSV output (columns: Frame, Visibility, X, Y) into per-frame
positions our pipeline can consume. This keeps us aligned with upstream weights
and inference exactly.

Setup (once, see README / setup_tracknet.sh):
    git clone https://github.com/qaz812345/TrackNetV3 third_party/TrackNetV3
    # download TrackNet_best.pt and InpaintNet_best.pt into third_party/TrackNetV3/

Usage as a library:
    from shuttle_tracker import TrackNetV3Tracker
    tracker = TrackNetV3Tracker(repo="third_party/TrackNetV3")
    track = tracker.track("match.mp4")     # {frame_idx: ShuttlePoint}

CLI (writes shuttle.json next to the video):
    python src/shuttle_tracker.py --source match.mp4 --repo third_party/TrackNetV3
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict


@dataclass
class ShuttlePoint:
    frame: int
    x: float
    y: float
    visible: bool


class ShuttleTracker:
    """Interface so the detector is swappable (TrackNet now, others later)."""

    def track(self, video_path):
        """Return {frame_index: ShuttlePoint}. Missing frames => not detected."""
        raise NotImplementedError


class TrackNetV3Tracker(ShuttleTracker):
    def __init__(self, repo="third_party/TrackNetV3",
                 tracknet_ckpt="TrackNet_best.pt",
                 inpaint_ckpt="InpaintNet_best.pt",
                 predict_script="predict.py",
                 use_inpaint=True):
        self.repo = os.path.abspath(repo)
        self.tracknet_ckpt = self._resolve(tracknet_ckpt)
        self.inpaint_ckpt = self._resolve(inpaint_ckpt) if use_inpaint else None
        self.predict_script = predict_script
        if not os.path.isdir(self.repo):
            raise FileNotFoundError(
                f"TrackNetV3 repo not found at {self.repo}. "
                "Clone it first (see README / setup_tracknet.sh).")

    def _resolve(self, ckpt):
        p = ckpt if os.path.isabs(ckpt) else os.path.join(self.repo, ckpt)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"checkpoint not found: {p} -- download TrackNetV3 weights "
                "(see README).")
        return p

    def track(self, video_path, save_dir=None):
        video_path = os.path.abspath(video_path)
        save_dir = os.path.abspath(save_dir or os.path.dirname(video_path) or ".")
        os.makedirs(save_dir, exist_ok=True)

        cmd = [
            sys.executable, self.predict_script,
            "--video_file", video_path,
            "--tracknet_file", self.tracknet_ckpt,
            "--save_dir", save_dir,
        ]
        if self.inpaint_ckpt:
            cmd += ["--inpaintnet_file", self.inpaint_ckpt]

        print(f"[info] running TrackNetV3:\n  {' '.join(cmd)}")
        subprocess.run(cmd, cwd=self.repo, check=True)

        csv_path = self._find_csv(save_dir, video_path)
        return self._parse_csv(csv_path)

    @staticmethod
    def _find_csv(save_dir, video_path):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        # TrackNetV3 typically writes "<stem>_ball.csv"; fall back to any csv.
        candidates = [
            os.path.join(save_dir, f"{stem}_ball.csv"),
            os.path.join(save_dir, f"{stem}.csv"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        csvs = [f for f in os.listdir(save_dir) if f.endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"no TrackNetV3 CSV produced in {save_dir}")
        return os.path.join(save_dir, sorted(csvs)[0])

    @staticmethod
    def _parse_csv(csv_path):
        track = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                key = {k.lower(): k for k in row}
                fi = int(float(row[key["frame"]]))
                vis = int(float(row[key.get("visibility", key.get("vis"))])) == 1
                x = float(row[key["x"]])
                y = float(row[key["y"]])
                track[fi] = ShuttlePoint(fi, x, y, vis and (x > 0 or y > 0))
        print(f"[info] parsed {len(track)} frames from {csv_path}")
        return track


def fill_gaps(track, max_gap=5):
    """Linearly interpolate short invisible gaps so speed estimation is stable."""
    if not track:
        return track
    frames = sorted(track)
    out = dict(track)
    vis = [f for f in frames if track[f].visible]
    for a, b in zip(vis, vis[1:]):
        gap = b - a
        if 1 < gap <= max_gap:
            pa, pb = track[a], track[b]
            for k in range(1, gap):
                t = k / gap
                out[a + k] = ShuttlePoint(
                    a + k, pa.x + (pb.x - pa.x) * t,
                    pa.y + (pb.y - pa.y) * t, True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input video path")
    ap.add_argument("--repo", default="third_party/TrackNetV3")
    ap.add_argument("--tracknet-ckpt", default="TrackNet_best.pt")
    ap.add_argument("--inpaint-ckpt", default="InpaintNet_best.pt")
    ap.add_argument("--no-inpaint", action="store_true")
    ap.add_argument("--out", default="shuttle.json")
    ap.add_argument("--fill-gaps", type=int, default=5,
                    help="max invisible gap (frames) to interpolate; 0 disables")
    args = ap.parse_args()

    tracker = TrackNetV3Tracker(
        repo=args.repo,
        tracknet_ckpt=args.tracknet_ckpt,
        inpaint_ckpt=args.inpaint_ckpt,
        use_inpaint=not args.no_inpaint,
    )
    track = tracker.track(args.source)
    if args.fill_gaps:
        track = fill_gaps(track, max_gap=args.fill_gaps)

    with open(args.out, "w") as f:
        json.dump({str(k): asdict(v) for k, v in sorted(track.items())}, f)
    visible = sum(1 for p in track.values() if p.visible)
    print(f"[done] {len(track)} frames ({visible} visible) -> {args.out}")


if __name__ == "__main__":
    main()
