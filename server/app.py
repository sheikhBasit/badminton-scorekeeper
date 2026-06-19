"""
FastAPI server — receives camera frames, runs StreamPipeline, broadcasts
results to display clients over WebSocket, serves the PWA web app.

Endpoints
---------
POST /calibrate       — initialise CourtMapper + StreamPipeline from 4 corners
POST /frame           — accept JPEG frame, run inference, return + broadcast JSON
WS   /ws              — WebSocket for display clients (receive-only, push JSON)
GET  /status          — health-check + current score
GET  /history         — list saved game records
POST /history         — save a completed game record
GET  /                — serves webapp/index.html (the PWA)

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Run inside Kaggle / Colab kernel:
    see _kg_stream/kernel.py
"""
import asyncio
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from calibrate_court import COURT_PTS_M, CourtMapper  # noqa: E402
from stream_pipeline import StreamPipeline             # noqa: E402

app = FastAPI(title="Badminton Scorekeeper")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

WEBAPP = Path(__file__).parent.parent / "webapp" / "index.html"
HISTORY_FILE = Path(__file__).parent.parent / "data" / "history.json"
HISTORY_FILE.parent.mkdir(exist_ok=True)

# ── global state ──────────────────────────────────────────────────────────────
_pipeline: Optional[StreamPipeline] = None
_frame_idx: int = 0
_last_result: Optional[dict] = None
_display_clients: list[WebSocket] = []
# Single-worker executor keeps ML inference strictly serial
_executor = ThreadPoolExecutor(max_workers=1)


# ── WebSocket broadcast ───────────────────────────────────────────────────────

async def _broadcast(result: dict):
    msg = json.dumps(result)
    dead = []
    for ws in _display_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _display_clients.remove(ws)


# ── /calibrate ────────────────────────────────────────────────────────────────

class CalibrateRequest(BaseModel):
    # 4 pixel corners [[x,y], ...] order: TL, TR, BR, BL
    corners: list[list[float]]
    frame_w: int = 854
    frame_h: int = 480
    model: str = "yolo11n.pt"
    conf: float = 0.25
    court_margin: float = 0.4
    init_score: Optional[str] = None   # "13,9"
    first_server: str = "A"
    name_a: str = "A"
    name_b: str = "B"
    fps: float = 30.0
    # TrackNetV3 (optional — omit for player-only mode)
    tracknet_repo: Optional[str] = None
    tracknet_ckpt: Optional[str] = None
    inpaint_ckpt: Optional[str] = None


@app.post("/calibrate")
async def calibrate(req: CalibrateRequest):
    global _pipeline, _frame_idx

    corners = np.float32(req.corners)
    H, _ = cv2.findHomography(corners, COURT_PTS_M)
    if H is None:
        raise HTTPException(400, "Homography failed — corners likely collinear")

    mapper = CourtMapper(H, corners, (req.frame_w, req.frame_h))

    init = None
    if req.init_score:
        a, b = (int(x) for x in req.init_score.split(","))
        init = {"A": a, "B": b}

    _pipeline = StreamPipeline(
        mapper=mapper,
        model_path=req.model,
        conf=req.conf,
        court_margin=req.court_margin,
        tracknet_repo=req.tracknet_repo,
        tracknet_ckpt=req.tracknet_ckpt,
        inpaint_ckpt=req.inpaint_ckpt,
        fps=req.fps,
        initial_score=init,
        first_server=req.first_server,
        names={"A": req.name_a, "B": req.name_b},
    )
    _frame_idx = 0
    print(f"[calibrate] pipeline ready — corners={req.corners}", flush=True)
    return {"status": "ok"}


# ── /frame  (camera phone POSTs here) ────────────────────────────────────────

@app.post("/frame")
async def process_frame(file: UploadFile = File(...)):
    global _frame_idx, _last_result

    if _pipeline is None:
        raise HTTPException(400, "Not calibrated — POST /calibrate first")

    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image — send JPEG bytes")

    idx = _frame_idx
    _frame_idx += 1

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _pipeline.process, frame, idx)
    _last_result = result

    await _broadcast(result)
    return result


# ── /ws  (display phones connect here) ───────────────────────────────────────

@app.websocket("/ws")
async def display_ws(websocket: WebSocket):
    await websocket.accept()
    _display_clients.append(websocket)
    print(f"[ws] display client connected ({len(_display_clients)} total)", flush=True)
    # immediately push last known state so the display isn't blank on connect
    if _last_result:
        await websocket.send_text(json.dumps(_last_result))
    try:
        while True:
            # keep connection alive; display clients don't send data
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _display_clients:
            _display_clients.remove(websocket)
        print(f"[ws] display client disconnected ({len(_display_clients)} remaining)", flush=True)


# ── /status ───────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "calibrated": _pipeline is not None,
        "frame": _frame_idx,
        "display_clients": len(_display_clients),
        "last_result": _last_result,
    }


# ── / (Phase 3 will serve the web app here) ──────────────────────────────────

@app.get("/history")
async def get_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


@app.post("/history")
async def save_game(game: dict):
    history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
    history.insert(0, game)          # newest first
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    print(f"[history] saved game #{len(history)}", flush=True)
    return {"saved": True, "total": len(history)}


@app.get("/")
async def root():
    if WEBAPP.exists():
        return FileResponse(WEBAPP, media_type="text/html")
    return JSONResponse({"status": "web app not built yet — run Phase 3"})
