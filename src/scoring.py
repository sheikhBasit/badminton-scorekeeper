"""
Stage 5: scoring.

Three parts:
  1. BadmintonMatch  -- a rules-correct rally-point state machine
                        (to 21, win by 2, cap at 30, best of 3 games).
  2. detect_rallies  -- segment the shuttle track into rallies automatically
                        (a rally = an active span of shuttle motion; the gap
                        between rallies is when the shuttle is gone/still).
  3. a semi-automatic winner workflow + scoreboard overlay render.

Deciding the point winner automatically is the hard, unreliable part (net shots,
out calls, lets). So this is SEMI-automatic:

  Pass 1 (detect): we segment rallies and write rallies.json, pre-filling each
  rally's winner with a guess from which court half the shuttle last landed in
  (landing in A's half => B wins the rally). We also render with those guesses.

      python src/scoring.py --source match.mp4 --shuttle shuttle.json \
          --court court.npz --output scored.mp4

  Pass 2 (correct): open rallies.json, fix any wrong "winner" values
  ("A" or "B"), then re-render exactly:

      python src/scoring.py --source match.mp4 --shuttle shuttle.json \
          --court court.npz --winners-file rallies.json --output scored.mp4

Outputs: scored.mp4 (scoreboard overlay), rallies.json (editable), score_log.json.
"""
import argparse
import json

import cv2
import numpy as np
import supervision as sv

from calibrate_court import CourtMapper, COURT_L_M

# rally detection tuning
MERGE_GAP = 22          # invisible frames within this are still the same rally
                        # (TrackNetV3 routinely drops the shuttle for ~0.5s
                        # mid-rally; must exceed that or one rally splits in two)
MIN_RALLY_FRAMES = 20   # spans shorter than this are noise, not rallies
MIN_RALLY_TRAVEL = 120  # min pixel travel for a span to count as a rally
LANDING_MARGIN_M = 0.6  # how far outside the court a point still counts as "in play"


class BadmintonMatch:
    """Rally-point scoring: to 21, win by 2, hard cap 30, best of 3 games."""

    def __init__(self, target=21, cap=30, win_by=2, games_to_win=2, first_server="A"):
        self.target, self.cap, self.win_by = target, cap, win_by
        self.games_to_win = games_to_win
        self.score = {"A": 0, "B": 0}
        self.games = {"A": 0, "B": 0}
        self.server = first_server
        self.game_no = 1
        self.match_over = False
        self.winner = None

    def _game_won_by(self, me, opp):
        s, o = self.score[me], self.score[opp]
        return (s >= self.target and s - o >= self.win_by) or s >= self.cap

    def award(self, w):
        """Award a rally to 'A' or 'B'; returns a snapshot of the new state."""
        if self.match_over:
            return self.snapshot(event="match_over")
        self.score[w] += 1
        self.server = w
        event = "point"
        other = "B" if w == "A" else "A"
        if self._game_won_by(w, other):
            self.games[w] += 1
            event = "game"
            if self.games[w] >= self.games_to_win:
                self.match_over = True
                self.winner = w
                event = "match"
            else:
                self.score = {"A": 0, "B": 0}
                self.game_no += 1
                self.server = w  # game winner serves next game
        return self.snapshot(event=event)

    def snapshot(self, event="point"):
        return {
            "a": self.score["A"], "b": self.score["B"],
            "games_a": self.games["A"], "games_b": self.games["B"],
            "server": self.server, "game_no": self.game_no,
            "event": event, "match_over": self.match_over, "winner": self.winner,
        }


def load_all_shuttle(path):
    """shuttle.json -> sorted list of (frame, x, y, visible) for all frames."""
    with open(path) as f:
        raw = json.load(f)
    pts = [(int(k), float(v["x"]), float(v["y"]), bool(v.get("visible", True)))
           for k, v in raw.items()]
    pts.sort()
    return pts


def detect_rallies(pts):
    """Group visible shuttle motion into rallies. Returns list of dicts."""
    vis = [(f, x, y) for f, x, y, v in pts if v]
    if not vis:
        return []
    spans, cur = [], [vis[0]]
    for prev, nxt in zip(vis, vis[1:]):
        if nxt[0] - prev[0] <= MERGE_GAP:
            cur.append(nxt)
        else:
            spans.append(cur)
            cur = [nxt]
    spans.append(cur)

    rallies = []
    for span in spans:
        fs = [p[0] for p in span]
        xy = np.array([[p[1], p[2]] for p in span], dtype=float)
        if fs[-1] - fs[0] < MIN_RALLY_FRAMES:
            continue
        travel = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        if travel < MIN_RALLY_TRAVEL:
            continue
        rallies.append({
            "start_frame": fs[0],
            "end_frame": fs[-1],
            "landing_px": [float(xy[-1, 0]), float(xy[-1, 1])],
            "track_px": xy.tolist(),
        })
    return rallies


def estimate_landing(track_px, mapper):
    """Best guess at where the shuttle came down, in metres.

    The raw last visible point is unreliable (TrackNetV3 often loses the shuttle
    when it's lofted high and out of frame, leaving a 'landing' at the frame top).
    So restrict to points that are actually inside the court, and take the last
    one of those -- that's far closer to the real landing than the last point
    overall.
    """
    pts = np.asarray(track_px, dtype=float)
    inside = mapper.contains(pts, margin_m=LANDING_MARGIN_M)
    cand = mapper.to_metres(pts[inside]) if inside.any() else mapper.to_metres(pts)
    return cand[-1]


def guess_winner(rally, mapper):
    """Heuristic: shuttle landing in a side's half => the OTHER side won."""
    track = rally.get("track_px") or [rally["landing_px"]]
    land = estimate_landing(track, mapper)
    near_half = land[1] < COURT_L_M / 2      # 'A' defends the near (top) half
    return "B" if near_half else "A"


def draw_scoreboard(frame, snap):
    h, w = frame.shape[:2]
    cx = w // 2
    box_w, box_h = 360, 70
    x0, y0 = cx - box_w // 2, 15
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)

    def side(label, score, games, serving, x):
        col = (0, 255, 255) if serving else (255, 255, 255)
        dot = " *" if serving else ""
        cv2.putText(frame, f"{label}{dot}", (x, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(frame, f"{score}", (x, y0 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.putText(frame, f"(g{games})", (x + 40, y0 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    side("A", snap["a"], snap["games_a"], snap["server"] == "A", x0 + 20)
    cv2.putText(frame, "-", (cx - 6, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)
    side("B", snap["b"], snap["games_b"], snap["server"] == "B", x0 + box_w - 110)

    banner = None
    if snap["match_over"]:
        banner = f"MATCH: {snap['winner']} wins"
    elif snap["event"] == "game":
        banner = f"GAME {snap['game_no'] - 1}"
    if banner:
        cv2.putText(frame, banner, (x0, y0 + box_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


def build_timeline(rallies, total_frames):
    """frame index -> scoreboard snapshot in effect at that frame."""
    match = BadmintonMatch()
    log = []
    # snapshot before any point
    snaps = [(0, match.snapshot(event="start"))]
    for r in rallies:
        snap = match.award(r["winner"])
        log.append({**r, **snap})
        snaps.append((r["end_frame"], snap))

    timeline, si = {}, 0
    for f in range(total_frames):
        while si + 1 < len(snaps) and snaps[si + 1][0] <= f:
            si += 1
        timeline[f] = snaps[si][1]
    return timeline, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--shuttle", required=True, help="shuttle.json (Stage 3)")
    ap.add_argument("--court", required=True, help="court.npz (Stage 2)")
    ap.add_argument("--winners-file", help="rallies.json with corrected 'winner' fields")
    ap.add_argument("--output", default="scored.mp4")
    args = ap.parse_args()

    info = sv.VideoInfo.from_video_path(args.source)
    mapper = CourtMapper.load(args.court)

    if args.winners_file:
        with open(args.winners_file) as f:
            rallies = json.load(f)
        for r in rallies:
            if r.get("winner") not in ("A", "B"):
                raise SystemExit(f"[error] rally {r} has no valid 'winner' (A/B)")
        print(f"[info] using {len(rallies)} corrected rallies from {args.winners_file}")
    else:
        pts = load_all_shuttle(args.shuttle)
        rallies = detect_rallies(pts)
        for r in rallies:
            r["winner"] = guess_winner(r, mapper)
        # keep rallies.json compact + hand-editable: track_px was only needed
        # for the landing estimate above.
        editable = [{k: v for k, v in r.items() if k != "track_px"} for r in rallies]
        with open("rallies.json", "w") as f:
            json.dump(editable, f, indent=2)
        print(f"[info] detected {len(rallies)} rallies -> rallies.json "
              "(edit 'winner' fields, then rerun with --winners-file rallies.json)")

    timeline, log = build_timeline(rallies, info.total_frames)
    with open("score_log.json", "w") as f:
        json.dump(log, f, indent=2)

    def cb(frame, idx):
        snap = timeline.get(idx, timeline.get(info.total_frames - 1))
        return draw_scoreboard(frame, snap)

    sv.process_video(source_path=args.source, target_path=args.output, callback=cb)

    final = log[-1] if log else {"a": 0, "b": 0}
    print(f"[done] {len(rallies)} rallies. final game {final.get('a')}-{final.get('b')}"
          f", games {final.get('games_a', 0)}-{final.get('games_b', 0)}")
    print(f"       wrote {args.output}, rallies.json, score_log.json")


if __name__ == "__main__":
    main()
