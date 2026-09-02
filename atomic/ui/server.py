"""Server: FastAPI app that exposes the ATOMIC-PC UI.

Routes:
  GET  /                          -> static index.html (the tile wall + control frame)
  GET  /run/{name}                -> same HTML; selects a program by name
  GET  /api/programs              -> list registered programs
  GET  /api/control/{name}        -> control frame schema (sliders + tap) for a program
  GET  /api/snapshot/{name}       -> current bus + series snapshot
  GET  /api/views/{name}          -> viz layout (tile -> module + as)
  POST /api/feed/{name}           -> {ticks: [..], params: {id:{k:v}}}  (live feed)
  POST /api/tap/{name}            -> 1-tick tap event (recorded for the current tick)
  POST /api/batch/{name}          -> {ticks: N}  -> run a batch and return the final snapshot
  GET  /api/stream/{name}         -> SSE tick stream (server pushes snapshot every tick)
  WS   /ws/{name}                 -> WebSocket live tick stream (preferred)

The UI is a pure-web client.  The server is the bridge between the
HTML5 tile wall (canvas redraws per tick) and the Python engine
(atomic.engine.Engine).  Mode 2 (LiveOracle) is NOT used here -- the
viewer drives Engine.tick() directly inside an asyncio task so the
WS/SSE stream is single-loop.  Parity with mode 1 is preserved by
feeding the same feeds the LiveOracle would consume.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..gates import param_range
from ..program import Program
from .viewer import Viewer


_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC = os.path.join(_HERE, "static")
_INDEX = os.path.join(_STATIC, "index.html")
_DT = 1.0 / 30.0


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from .programs import build as _build, all_programs
    for name in all_programs():
        prog = _build(name)
        if prog:
            Viewer.put(name, Viewer(prog, name=name))
    yield
    Viewer._registry.clear()

def create_app() -> FastAPI:
    app = FastAPI(title="atomic-pc-ui", version="0.1", lifespan=_lifespan)
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/run/{name}", response_class=HTMLResponse)
    async def index(name: str = "default"):
        if not os.path.exists(_INDEX):
            raise HTTPException(500, "ui static not built")
        return FileResponse(_INDEX)

    @app.get("/api/programs")
    async def programs():
        from .programs import all_programs
        return {
            "programs": all_programs(),
            "active": list(Viewer._registry.keys()),
        }

    @app.get("/api/control/{name}")
    async def control(name: str):
        v = Viewer.get(name)
        if v is None:
            v = _auto_register(name)
        if v is None:
            raise HTTPException(404, f"program {name!r} not found")
        return _control_schema(v)

    @app.get("/api/snapshot/{name}")
    async def snapshot(name: str):
        v = Viewer.get(name)
        if v is None:
            v = _auto_register(name)
        if v is None:
            raise HTTPException(404, f"program {name!r} not found")
        return v.snapshot()

    @app.get("/api/views/{name}")
    async def views(name: str):
        v = Viewer.get(name)
        if v is None:
            v = _auto_register(name)
        if v is None:
            raise HTTPException(404, f"program {name!r} not found")
        return {"views": v.view_layout, "display": {"cols": 4, "rows": 4}}

    @app.post("/api/feed/{name}")
    async def feed(name: str, payload: dict):
        v = Viewer.get(name)
        if v is None:
            raise HTTPException(404, f"program {name!r} not found")
        for t in payload.get("ticks") or []:
            v.apply_feed(int(t), payload)
        return {"ok": True, "applied": len(payload.get("ticks") or [])}

    @app.post("/api/tap/{name}")
    async def tap(name: str):
        v = Viewer.get(name)
        if v is None:
            raise HTTPException(404, f"program {name!r} not found")
        v.tap()
        return {"ok": True, "tick": v.tick}

    @app.post("/api/batch/{name}")
    async def batch(name: str, payload: dict):
        v = Viewer.get(name)
        if v is None:
            raise HTTPException(404, f"program {name!r} not found")
        ticks = int(payload.get("ticks", 60))
        for t, f in (payload.get("feeds") or {}).items():
            v.apply_feed(int(t), f)
        return v.batch(ticks)

    @app.get("/api/stream/{name}")
    async def stream(name: str, fps: float = 30.0, ticks: int = 600):
        v = Viewer.get(name)
        if v is None:
            raise HTTPException(404, f"program {name!r} not found")
        period = 1.0 / max(1.0, fps)

        async def gen():
            for _ in range(int(ticks)):
                snap = v.tick_once()
                yield f"data: {json.dumps(snap)}\n\n"
                await asyncio.sleep(period)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.websocket("/ws/{name}")
    async def ws(name: str, ws: WebSocket):
        await ws.accept()
        v = Viewer.get(name)
        if v is None:
            v = _auto_register(name)
        if v is None:
            await ws.send_text(json.dumps({"error": f"program {name!r} not found"}))
            await ws.close()
            return
        q = await v.ws_connect()
        try:
            await ws.send_text(json.dumps(v.snapshot()))
            stop = asyncio.Event()

            async def tick_loop():
                first = True
                while not stop.is_set():
                    t0 = time.perf_counter()
                    v.tick_once()
                    eng_us = (time.perf_counter() - t0) * 1e6
                    prev_bus = getattr(v, "_last_bus", None)
                    t1 = time.perf_counter()
                    diff = v.snapshot_diff(prev_bus=prev_bus)
                    if first:
                        diff["_lat_eng"] = eng_us
                        first = False
                    try:
                        await ws.send_text(json.dumps(diff))
                    except Exception:
                        stop.set()
                        return
                    ws_us = (time.perf_counter() - t1) * 1e6
                    v.set_last_latency(eng_us, ws_us)
                    await asyncio.sleep(v.dt)

            async def recv_loop():
                while not stop.is_set():
                    try:
                        msg = await ws.receive_json()
                    except Exception:
                        stop.set()
                        return
                    await _handle_ws_msg(v, ws, msg)

            t1 = asyncio.create_task(tick_loop())
            t2 = asyncio.create_task(recv_loop())
            try:
                await asyncio.wait({t1, t2},
                                   return_when=asyncio.FIRST_COMPLETED)
            finally:
                stop.set()
                for t in (t1, t2):
                    t.cancel()
        finally:
            v.ws_disconnect(q)

    @app.get("/api/wsstats")
    async def wsstats(name: str = "default"):
        v = Viewer.get(name)
        if v is None:
            v = _auto_register(name)
        if v is None:
            raise HTTPException(404, f"program {name!r} not found")
        return v.ws_stats()

    # ── iter 4: record live WS to .qbf shard, list runs, replay ─────────────
    @app.post("/api/record/{name}")
    async def record(name: str, payload: dict):
        from .qbf_records import save_record, get_qbf_dir
        frames = payload.get("frames") or []
        if not isinstance(frames, list):
            raise HTTPException(400, "frames must be a list")
        run_id = save_record(name, frames)
        return {"ok": True, "run_id": run_id, "frames": len(frames),
                "dir": str(get_qbf_dir())}

    @app.get("/api/replay_runs/{name}")
    async def replay_runs(name: str):
        from .qbf_records import list_records
        return {"runs": list_records(name)}

    @app.get("/api/replay/{name}")
    async def replay(name: str, run_id: str = ""):
        from .qbf_records import load_record
        frames = load_record(name, run_id)
        if frames is None:
            raise HTTPException(404, f"no record {run_id!r} for {name!r}")
        return frames

    return app


async def _drive_tick(v: Viewer) -> dict:
    return v.tick_once()


async def _handle_ws_msg(v: Viewer, ws: WebSocket, msg: dict):
    kind = msg.get("type", "")
    if kind == "ping":
        # iter 4: RTT echo. Client sends {_t: <ms timestamp>}; we reply
        # immediately with {_pong: true} so the client can compute RTT.
        await ws.send_text(json.dumps({"_pong": True}))
    elif kind == "tap":
        v.tap()
        await ws.send_text(json.dumps({"ack": "tap", "tick": v.tick}))
    elif kind == "param":
        mid = msg.get("module", "")
        params = msg.get("params") or {}
        for k, val in params.items():
            v.set_param(mid, k, val)
        await ws.send_text(json.dumps({"ack": "param", "module": mid}))
    elif kind == "feed":
        for t in (msg.get("ticks") or []):
            v.apply_feed(int(t), msg)
        await ws.send_text(json.dumps({"ack": "feed", "applied": len(msg.get("ticks") or [])}))
    elif kind == "batch":
        ticks = int(msg.get("ticks", 60))
        snap = v.batch(ticks)
        await ws.send_text(json.dumps({"ack": "batch", "snapshot": snap}))
    else:
        await ws.send_text(json.dumps({"error": f"unknown msg type {kind!r}"}))


def _control_schema(v: Viewer) -> dict:
    params: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for m in v.modules:
        mid = m.get("id", "")
        prim = m.get("primitive", "")
        for k, val in (m.get("params") or {}).items():
            if (mid, k) in seen:
                continue
            seen.add((mid, k))
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            rng = param_range(prim, k)
            if rng is not None:
                mn, mx, step, unit = rng
            else:
                mn, mx, step, unit = _fallback_range(k)
            params.append({
                "module": mid, "key": k, "value": fv,
                "min": mn, "max": mx, "step": step, "unit": unit,
            })
    return {
        "name": v.name,
        "dt": v.dt,
        "tick": v.tick,
        "window": 512,
        "params": params,
        "taps": [{"module": m["id"], "name": "tap"} for m in v.modules
                 if m.get("primitive") in ("tap", "sensor")],
    }


def _fallback_range(key: str) -> tuple[float, float, float, str]:
    key_l = key.lower()
    if key_l in ("bpm",):
        return (1.0, 300.0, 1.0, "bpm")
    if key_l in ("gain", "factor", "amp", "offset", "value"):
        return (-4.0, 4.0, 0.01, "")
    if key_l in ("hz", "rate_hz"):
        return (0.01, 20.0, 0.01, "Hz")
    if key_l in ("smooth", "alpha"):
        return (0.0, 1.0, 0.001, "")
    if key_l in ("threshold", "hi", "lo"):
        return (-4.0, 4.0, 0.01, "")
    if key_l in ("per_tick", "reset"):
        return (-100.0, 100.0, 0.1, "")
    return (-1e6, 1e6, 0.01, "")


def _auto_register(name: str) -> Viewer | None:
    if name in Viewer._registry:
        return Viewer._registry[name]
    try:
        from .programs import build as _build
        prog = _build(name)
        if prog is None:
            return None
        Viewer.put(name, Viewer(prog, name=name))
        return Viewer._registry[name]
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"failed to build program {name!r}: {exc}")


app = create_app()
