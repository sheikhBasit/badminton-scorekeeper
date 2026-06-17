"""
Stage 2: court calibration -> homography (pixels -> real-world metres).

A badminton court has fixed dimensions, so 4 known points let us build a
homography that maps any image pixel to a metre coordinate on the court.
That mapping is what turns shuttle pixel-motion into real km/h in Stage 4.

Court reference (outer boundary, doubles): 6.10 m wide x 13.40 m long.
Pick the 4 OUTER corners of the court in this order:
    1 top-left  2 top-right  3 bottom-right  4 bottom-left

Two ways to supply the corners:

  Headless / Kaggle (no GUI window):
    python src/calibrate_court.py --source match.mp4 --frame 0 \
        --corners "315,210 980,205 1180,690 110,700" --out court.npz

  Local desktop (click them on the frame):
    python src/calibrate_court.py --source match.mp4 --frame 0 \
        --interactive --out court.npz

Outputs:
    court.npz            homography H + metadata (load with CourtMapper.load)
    court_corners.jpg    the frame with your 4 picked corners drawn
    court_topdown.jpg    warped top-down view -- court should look rectangular
"""
import argparse
import json

import cv2
import numpy as np

# Doubles court outer boundary, in metres (width x length).
COURT_W_M = 6.10
COURT_L_M = 13.40

# Destination points in metres, same order as the picked corners:
# top-left, top-right, bottom-right, bottom-left.
COURT_PTS_M = np.float32([
    [0.0,       0.0],
    [COURT_W_M, 0.0],
    [COURT_W_M, COURT_L_M],
    [0.0,       COURT_L_M],
])

# Pixels per metre used only for the top-down preview image.
PREVIEW_PPM = 50


class CourtMapper:
    """Maps image pixels to court metres and estimates real-world speed."""

    def __init__(self, H, image_corners, frame_size):
        self.H = np.asarray(H, dtype=np.float64)
        self.image_corners = np.asarray(image_corners, dtype=np.float32)
        self.frame_size = tuple(frame_size)  # (w, h)

    def to_metres(self, pts):
        """pts: (N,2) array of pixel coords -> (N,2) array of metre coords."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def contains(self, pts, margin_m=0.0):
        """Boolean mask: which pixel pts map inside the court (+margin metres).

        Used to reject off-court people (crowd, coaches, line judges, umpire).
        """
        m = self.to_metres(pts)
        x, y = m[:, 0], m[:, 1]
        return ((x >= -margin_m) & (x <= COURT_W_M + margin_m) &
                (y >= -margin_m) & (y <= COURT_L_M + margin_m))

    def speed_kmh(self, px_a, px_b, dt_seconds):
        """Real-world speed of an object moving px_a -> px_b over dt seconds."""
        if dt_seconds <= 0:
            return 0.0
        a, b = self.to_metres([px_a, px_b])
        metres = float(np.linalg.norm(b - a))
        return metres / dt_seconds * 3.6

    def save(self, path):
        np.savez(
            path,
            H=self.H,
            image_corners=self.image_corners,
            frame_size=np.asarray(self.frame_size),
            court_pts_m=COURT_PTS_M,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d["H"], d["image_corners"], tuple(d["frame_size"]))


def grab_frame(source, frame_index):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[error] cannot open {source}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"[error] cannot read frame {frame_index}")
    return frame


def parse_corners(text):
    pts = []
    for tok in text.replace(",", " ").split():
        pts.append(float(tok))
    if len(pts) != 8:
        raise SystemExit("[error] --corners needs 4 'x,y' pairs (8 numbers)")
    return np.float32(pts).reshape(4, 2)


def pick_corners_interactive(frame):
    """Local-only: click the 4 corners on an OpenCV window."""
    pts, labels = [], ["top-left", "top-right", "bottom-right", "bottom-left"]
    disp = frame.copy()

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(disp, str(len(pts)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    win = "click 4 court corners: TL, TR, BR, BL  (q when done)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        hint = labels[len(pts)] if len(pts) < 4 else "press q"
        view = disp.copy()
        cv2.putText(view, f"next: {hint}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow(win, view)
        if cv2.waitKey(20) & 0xFF == ord("q") and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    return np.float32(pts)


def draw_corners(frame, corners):
    out = frame.copy()
    pts = corners.astype(int)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
    for i, (x, y) in enumerate(pts, 1):
        cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(out, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out


def topdown_preview(frame, H):
    w = int(COURT_W_M * PREVIEW_PPM)
    h = int(COURT_L_M * PREVIEW_PPM)
    # H maps px->metres; scale metres->preview pixels.
    scale = np.float64([[PREVIEW_PPM, 0, 0],
                        [0, PREVIEW_PPM, 0],
                        [0, 0, 1]])
    return cv2.warpPerspective(frame, scale @ H, (w, h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input video path")
    ap.add_argument("--frame", type=int, default=0, help="frame index to calibrate on")
    ap.add_argument("--corners", help='4 "x,y" pairs: TL TR BR BL (headless mode)')
    ap.add_argument("--interactive", action="store_true", help="click corners in a GUI window")
    ap.add_argument("--out", default="court.npz", help="output homography file")
    args = ap.parse_args()

    frame = grab_frame(args.source, args.frame)
    h, w = frame.shape[:2]
    print(f"[info] frame {args.frame}: {w}x{h}")

    if args.interactive:
        corners = pick_corners_interactive(frame)
    elif args.corners:
        corners = parse_corners(args.corners)
    else:
        raise SystemExit("[error] provide --corners or --interactive")

    H, _ = cv2.findHomography(corners, COURT_PTS_M)
    if H is None:
        raise SystemExit("[error] homography failed -- corners likely collinear")

    mapper = CourtMapper(H, corners, (w, h))
    mapper.save(args.out)

    cv2.imwrite("court_corners.jpg", draw_corners(frame, corners))
    cv2.imwrite("court_topdown.jpg", topdown_preview(frame, H))

    # sanity: corners should map to the 4 court reference points in metres.
    mapped = mapper.to_metres(corners)
    print("[info] corner -> metre check (expect ~0/6.1/13.4):")
    print(json.dumps(np.round(mapped, 2).tolist(), indent=2))
    print(f"[done] saved {args.out}, court_corners.jpg, court_topdown.jpg")
    print("       verify court_topdown.jpg looks like a clean rectangle.")


if __name__ == "__main__":
    main()
