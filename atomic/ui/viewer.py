"""Viewer: wires a Program + Engine into a live-capable session.

A Viewer holds a compiled Program + running Engine and exposes two
paths:

  batch(): one-shot run of N ticks, returns snapshot dict
  live():  creates a LiveOracle (mode 2) and drives the engine one
           tick at a time; callers drive tick() externally

The live path connects to the UI WebSocket: each tick's snapshot is
pushed to all connected clients.  A feed dict {tick: {taps:[..],
params:{id:{k:v}}}} is applied BEFORE each module ticks (engine.py
_tick handles the merge).

Bus protocol (JSON-serialisable):
  snapshot = {"t": int, "running": bool, "bus": {k: float|null},
              "series": {k: [float]}, "views": [{id, module, output, as,
              viz, tile_row, tile_col}]}
  feed    = {"ticks": [int], "params": {id: {k: float}}}

Iter 3 (UI):
  - WS backpressure: per-client bounded queue (maxsize=2). A full
    queue on put is a *drop* (not a block) so a slow consumer cannot
    back up the engine's tick loop. The viewer tracks
    `drops_total` per client; `ws_stats()` reports the count.
  - Snapshot diff: the engine's bus is small (one float per key) but
    the series ring is up to 512 samples -- a 60 fps client sees
    ~30 kB/s of series payload. `snapshot_diff(prev)` emits only
    CHANGED bus keys + the last N series samples (default 64). The
    full snapshot is always available via `snapshot()` (used by the
    first frame on connect).
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from ..engine import Engine
from ..oracle import LiveOracle
from ..program import Program


class Viewer:
    _registry: dict[str, Viewer] = {}

    def __init__(self, program: Program, name: str = "default",
                 dt: float = 1.0 / 30.0):
        self.program = program
        self.name = name
        self.dt = dt
        self.patch = program.compile("microfx")
        modules = self.patch["modules"]
        wires = self.patch["wires"]
        views = self.patch.get("views") or []
        self.modules = modules
        self.wires = wires
        self.views = views
        self._engine: Engine | None = None
        self._oracle: LiveOracle | None = None
        self._running = False
        self._playing = True  # iter 45: server-side play/pause; WS still live but no tick
        self._tick_count = 0
        self._clients: list[asyncio.Queue] = []
        self._client_drops: dict[int, int] = {}
        self._client_ticks: dict[int, int] = {}
        self._lock = threading.RLock()
        self._feeds: dict[int, dict] = {}
        self._pending_feeds: dict[int, dict] = {}
        self._pending_taps: dict[int, list[int]] = {}
        self._view_layout: list[dict] = []
        self._last_bus: dict[str, float | None] = {}
        self._diff_series_window = 64
        self._ws_queue_max = 2

    def _init_engine(self) -> Engine:
        with self._lock:
            if self._engine is None:
                self._engine = Engine(
                    self.modules, self.wires,
                    views=self.views, dt=self.dt,
                    feeds=self._feeds)
            return self._engine

    @property
    def engine(self) -> Engine:
        return self._init_engine()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def tick(self) -> int:
        return self._tick_count

    def set_playing(self, playing: bool) -> None:
        """iter 45: toggle the engine's playing state. WS remains connected;
        tick_once() skips the engine.tick() call while paused but still
        snapshots + pushes to clients (so the UI stays responsive)."""
        with self._lock:
            self._playing = bool(playing)

    def step_once(self) -> dict:
        """iter 45: advance exactly one tick even while paused (for STEP button)."""
        eng = self.engine
        self.apply_pending(self._tick_count)
        eng.tick()
        self._tick_count = eng._t
        return self.snapshot()

    def reset_engine(self) -> dict:
        """iter 45: reset engine state (t=0, cleared bus/series/feeds)."""
        with self._lock:
            self._engine = None
            self._tick_count = 0
            self._feeds.clear()
            self._pending_feeds.clear()
            self._pending_taps.clear()
        return self.snapshot()

    @property
    def view_layout(self) -> list[dict]:
        if not self._view_layout:
            # Strategy: if the compiled patch has rich views with span info,
            # use them; otherwise fall back to auto-layout.
            views = self.views
            has_spans = any(
                (v.get("tile_rows", 1) > 1 or v.get("tile_cols", 1) > 1)
                for v in views
            )
            if views and not has_spans:
                # Views exist but are sparse (1x1 each). Use auto-layout to
                # assign sensible spans (video/wxyz3d/xy → 4x4; series → compact).
                auto = _auto_views(self.modules, cols=4, rows=4)
                if auto:
                    self._view_layout = auto
            if not self._view_layout:
                self._view_layout = _layout_views(views, cols=4)
        return self._view_layout

    def apply_feed(self, tick: int, feed: dict):
        taps = feed.get("ticks")
        params = feed.get("params")
        with self._lock:
            f = self._feeds.setdefault(tick, {})
            if taps is not None:
                f["taps"] = list(taps)
            if params:
                f["params"] = _deep_merge(f.get("params") or {}, params)

    def apply_pending(self, tick: int):
        with self._lock:
            if tick in self._pending_feeds or tick in self._pending_taps:
                f = self._feeds.setdefault(tick, {})
                if tick in self._pending_taps:
                    f["taps"] = self._pending_taps.pop(tick)
                pf = self._pending_feeds.pop(tick, {})
                if pf.get("params"):
                    f["params"] = _deep_merge(f.get("params") or {}, pf["params"])

    def tap(self, tick: int | None = None):
        t = tick if tick is not None else self._tick_count
        with self._lock:
            self._pending_taps.setdefault(t, []).append(t)

    def set_param(self, module_id: str, key: str, value: float, tick: int | None = None):
        t = tick if tick is not None else self._tick_count
        with self._lock:
            self._pending_feeds.setdefault(t, {})
            self._pending_feeds[t].setdefault("params", {})[module_id] = \
                {**self._pending_feeds[t].get("params", {}).get(module_id, {}),
                 key: value}

    def batch(self, ticks: int) -> dict:
        self._running = True
        eng = self.engine
        try:
            for _ in range(int(ticks)):
                self.apply_pending(self._tick_count)
                eng.tick()
            self._tick_count = eng._t
        finally:
            self._running = False
        return self.snapshot()

    def tick_once(self) -> dict:
        eng = self.engine
        self.apply_pending(self._tick_count)
        eng.tick()
        self._tick_count = eng._t
        return self.snapshot()

    def snapshot(self) -> dict:
        from ..engine import VIEW_WINDOW
        eng = self._engine
        bus = eng.bus.snapshot() if eng else {}
        series = {k: list(v) for k, v in eng.series.items()} if eng else {}
        return {
            "t": self._tick_count,
            "running": self._running,
            "bus": bus,
            "series": series,
            "views": self.view_layout,
            "window": VIEW_WINDOW,
        }

    def snapshot_diff(self, prev_bus: dict[str, float | None] | None = None,
                      prev_series: dict[str, list[float]] | None = None,
                      n_series: int | None = None) -> dict:
        from ..engine import VIEW_WINDOW
        eng = self._engine
        if eng is None:
            return {"t": self._tick_count, "running": self._running,
                    "bus": {}, "series": {}, "diff": True}
        bus = eng.bus.snapshot()
        if prev_bus is not None:
            bus = {k: v for k, v in bus.items()
                   if prev_bus.get(k) != v}
        series = eng.series
        w = n_series if n_series is not None else self._diff_series_window
        if prev_series is not None:
            series = {k: list(v)[-w:]
                      for k, v in series.items()
                      if list(v)[-w:] != prev_series.get(k, [])[-w:]}
        elif w < VIEW_WINDOW:
            series = {k: list(v)[-w:] for k, v in series.items()}
        self._last_bus = eng.bus.snapshot()
        return {
            "t": self._tick_count,
            "running": self._running,
            "bus": bus,
            "series": series,
            "diff": True,
        }

    def ws_stats(self) -> dict:
        with self._lock:
            return {
                "clients": len(self._clients),
                "drops": {cid: self._client_drops.get(cid, 0)
                          for cid in range(len(self._clients))},
            }

    def _broadcast(self, msg: dict):
        for i, q in enumerate(list(self._clients)):
            try:
                q.put_nowait(json.dumps(msg))
            except Exception:
                self._clients.remove(q)
                self._client_drops[i] = self._client_drops.get(i, 0) + 1

    async def ws_connect(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._ws_queue_max)
        with self._lock:
            idx = len(self._clients)
            self._clients.append(q)
            self._client_drops[idx] = 0
            self._client_ticks[idx] = 0
        return q

    def ws_disconnect(self, q: asyncio.Queue):
        with self._lock:
            if q in self._clients:
                idx = self._clients.index(q)
                self._clients.remove(q)
                self._client_drops.pop(idx, None)
                self._client_ticks.pop(idx, None)

    def ws_broadcast(self, msg: dict):
        for q in list(self._clients):
            try:
                q.put_nowait(json.dumps(msg))
            except Exception:
                self._clients.remove(q)

    def set_last_latency(self, eng_us: float, ws_us: float):
        self._last_eng_us = eng_us
        self._last_ws_us = ws_us

    def feed_frame(self, module_id: str, frame_bytes: bytes | bytearray | memoryview):
        eng = self.engine
        key = module_id + ".frame"
        eng.bus.set(key, bytes(frame_bytes))
        return True

    # ── iter 33: feed_video — server-push video frames into the engine ──────

    def feed_video_tick(self, frame_bytes: bytes | bytearray,
                        module_id: str = "vv") -> None:
        """Feed one RGBA frame into the engine and advance one tick.

        This is the server-push path: a video session (H3 or local stub)
        generates a frame, calls this method, and the engine tick renders it
        onto the tile wall via the viz_video sink. The frame_bytes are written
        to bus[<module_id>.frame] so the viz_video atom can decode + render.
        """
        eng = self.engine
        key = module_id + ".frame"
        eng.bus.set(key, bytes(frame_bytes))
        self.apply_pending(self._tick_count)
        eng.tick()
        self._tick_count = eng._t

    def feed_video_batch(self, frames: list[bytes],
                         module_id: str = "vv") -> dict:
        """Feed a list of RGBA frames into the engine, one per tick.

        Returns the final snapshot after all frames are consumed.
        """
        self._running = True
        eng = self.engine
        try:
            for raw in frames:
                key = module_id + ".frame"
                eng.bus.set(key, bytes(raw))
                self.apply_pending(self._tick_count)
                eng.tick()
                self._tick_count = eng._t
        finally:
            self._running = False
        return self.snapshot()

    @property
    def last_latency(self) -> tuple[float, float]:
        return (getattr(self, '_last_eng_us', 0.0),
                getattr(self, '_last_ws_us', 0.0))

    @classmethod
    def get(cls, name: str) -> "Viewer | None":
        return cls._registry.get(name)

    @classmethod
    def put(cls, name: str, viewer: "Viewer"):
        cls._registry[name] = viewer

    @classmethod
    def delete(cls, name: str):
        cls._registry.pop(name, None)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _layout_views(views: list[dict], cols: int = 4) -> list[dict]:
    out = []
    for i, v in enumerate(views):
        row = i // cols
        col = i % cols
        module = v.get("module", "")
        output = v.get("output", "cv")
        key = (module + "." + output).lower()
        vtype = v.get("as", "series")
        out.append({
            "id": v.get("id") or v.get("module", "v%d" % i),
            "module": module,
            "output": output,
            "key": key,
            "as": vtype,
            "viz": vtype,
            "tile_row": row,
            "tile_col": col,
            "tile_rows": v.get("tile_rows", 1),
            "tile_cols": v.get("tile_cols", 1),
        })
    return out


_VIZ_TYPES = {
    "viz_series": "series",
    "viz_xy": "xy",
    "viz_wxyz3d": "wxyz3d",
    "viz_video": "video",
    "viz_video_h3": "video",
}


def _auto_views(modules: list[dict], cols: int = 4, rows: int = 4) -> list[dict]:
    out = []
    video_mods = []   # viz_video / viz_fasth3_video — span 4x4
    wxyz3d_mods = []  # viz_wxyz3d — span 4x4
    xy_mods = []       # viz_xy — span 4x4
    series_mods = []   # viz_series — compact one-tile each

    for m in modules:
        prim = m.get("primitive", "")
        vtype = _VIZ_TYPES.get(prim)
        if vtype is None:
            continue
        mid = m.get("id", "")
        if vtype == "series":
            series_mods.append({"id": mid, "module": mid, "output": "cv",
                                "key": (mid + ".cv").lower(), "as": vtype, "viz": vtype,
                                "tile_rows": 1, "tile_cols": 1})
        elif vtype == "xy":
            xy_mods.append({"id": mid, "module": mid, "output": "y",
                            "key": (mid + ".y").lower(), "as": vtype, "viz": vtype,
                            "tile_rows": rows, "tile_cols": cols})
        elif vtype == "wxyz3d":
            wxyz3d_mods.append({"id": mid, "module": mid, "output": "z",
                               "key": (mid + ".z").lower(), "as": vtype, "viz": vtype,
                               "tile_rows": rows, "tile_cols": cols})
        elif vtype == "video":
            video_mods.append({"id": mid, "module": mid, "output": "ready",
                              "key": (mid + ".frame").lower(), "as": vtype, "viz": vtype,
                              "tile_rows": rows, "tile_cols": cols})

    # Layout order: video → wxyz3d → xy → series
    # Spanning views default to (0,0) — they cover the whole matrix.
    for v in video_mods + wxyz3d_mods + xy_mods:
        v.setdefault("tile_row", 0)
        v.setdefault("tile_col", 0)
    out.extend(video_mods)
    out.extend(wxyz3d_mods)
    out.extend(xy_mods)

    # Series: pack compactly in remaining free slots (row-major)
    cursor = 0
    for v in series_mods:
        while _view_slot_taken(out, cursor // cols, cursor % cols):
            cursor += 1
        v["tile_row"] = cursor // cols
        v["tile_col"] = cursor % cols
        out.append(v)
        cursor += 1

    return out


def _view_slot_taken(views: list[dict], row: int, col: int) -> bool:
    """True if any existing view occupies the (row, col) slot (including span)."""
    for v in views:
        r0 = v.get("tile_row", 0)
        c0 = v.get("tile_col", 0)
        rs = v.get("tile_rows", 1)
        cs = v.get("tile_cols", 1)
        if r0 <= row < r0 + rs and c0 <= col < c0 + cs:
            return True
    return False
