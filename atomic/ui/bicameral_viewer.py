"""BicameralViewer: live bridge depth + dual-engine UI snapshot.

Mirrors the Viewer class but wraps a BicameralPipeline (two engines
piped through HostBridge). The snapshot includes:
  - sub bus + sub series (the subconscious engine, GPU1)
  - con bus + con series (the conscious engine, GPU0)
  - bridge_depth (queue size at snapshot time)
  - bridge_latency (config: ticks)
  - bridge_pushed / bridge_popped (cumulative counters)

WebSocket frames carry the same shape as Viewer.snapshot() but with
`sub`/`con`/`bridge` keys at the top level instead of `bus`/`series`.
The static tile wall renders two side-by-side panes (sub on left,
con on right) using the existing renderer; the bridge depth bar is
plotted in the control header.
"""
from __future__ import annotations

import asyncio
import json
import threading

from ..bridge import BicameralPipeline, HostBridge
from ..program import Program
from .viewer import Viewer


class BicameralViewer:
    _registry: dict[str, BicameralViewer] = {}

    def __init__(self, sub_program: Program, con_program: Program,
                 bridge_map=None, bridge_latency=1, use_h4=False,
                 name: str = "bicameral", dt: float = 1.0 / 30.0):
        self.sub_program = sub_program
        self.con_program = con_program
        self.bridge_map = list(bridge_map or [])
        self.bridge_latency = int(bridge_latency)
        self.use_h4 = bool(use_h4)
        self.name = name
        self.dt = dt
        self._pipe: BicameralPipeline | None = None
        self._running = False
        self._tick_count = 0
        self._clients: list[asyncio.Queue] = []
        self._client_drops: dict[int, int] = {}
        self._lock = threading.RLock()
        self._view_layout: list[dict] = []
        self._depth_history: list[int] = []
        self._ws_queue_max = 2

    @property
    def pipeline(self) -> BicameralPipeline:
        if self._pipe is None:
            self._pipe = BicameralPipeline(
                self.sub_program, self.con_program,
                bridge_map=self.bridge_map,
                bridge_latency=self.bridge_latency,
                use_h4=self.use_h4,
                dt=self.dt,
            )
        return self._pipe

    @property
    def running(self) -> bool:
        return self._running

    @property
    def tick(self) -> int:
        return self._tick_count

    @property
    def bridge(self) -> HostBridge:
        return self.pipeline.bridge

    @property
    def view_layout(self) -> list[dict]:
        if not self._view_layout:
            sub_views = self.pipeline.sub.series
            con_views = self.pipeline.con.series
            cols = 4
            i = 0
            for series_dict in (sub_views, con_views):
                for key in series_dict:
                    r, c = i // cols, i % cols
                    self._view_layout.append({
                        "id": key,
                        "module": key.split(".")[0],
                        "output": key.split(".")[-1],
                        "key": key.lower(),
                        "as": "series",
                        "viz": "series",
                        "tile_row": r,
                        "tile_col": c,
                        "pane": 1 if series_dict is sub_views else 2,
                    })
                    i += 1
        return self._view_layout

    def tick_once(self) -> dict:
        pipe = self.pipeline
        pipe.tick()
        self._tick_count = pipe._t
        self._depth_history.append(pipe.bridge.depth())
        if len(self._depth_history) > 512:
            self._depth_history.pop(0)
        return self.snapshot()

    def batch(self, ticks: int) -> dict:
        self._running = True
        try:
            for _ in range(int(ticks)):
                self.tick_once()
        finally:
            self._running = False
        return self.snapshot()

    # ── iter 34: feed_video_tick (server-push frame into conscious engine) ─

    def feed_video_tick(self, frame_bytes, module_id="vv"):
        """Push one RGBA frame into the conscious engine and tick once.

        The frame is written to bus[con.<module_id>.frame] so the
        viz_video atom can decode + render it. The pipeline then ticks
        the conscious engine so the frame appears on the rendered bus
        at the next snapshot.
        """
        pipe = self.pipeline
        con = pipe.con
        key = module_id + ".frame"
        con.bus.set(key, bytes(frame_bytes))
        pipe.tick()
        self._tick_count = pipe._t
        self._depth_history.append(pipe.bridge.depth())
        if len(self._depth_history) > 512:
            self._depth_history.pop(0)
        return self.snapshot()

    def feed_video_batch(self, frames, module_id="vv"):
        """Push a list of RGBA frames into the conscious engine, one per tick."""
        self._running = True
        try:
            for raw in frames:
                self.feed_video_tick(raw, module_id=module_id)
        finally:
            self._running = False
        return self.snapshot()

    # ── iter 35: feed_ivl_tick (InfiniteVideoLoop step) ──────────────────

    def feed_ivl_tick(self, loop):
        """Step the InfiniteVideoLoop and capture the rendered viz_video output.

        Args:
            loop: an InfiniteVideoLoop wrapping this viewer.

        Returns:
            A snapshot dict with the updated sub/con/bridge state plus
            the latest frame metadata (`_ivl_frame`) on the top level.

        The H3Frame is captured from `loop.step()` BEFORE it returns None;
        if the loop is exhausted (max_ticks reached), `_ivl_frame` is None
        and the snapshot is returned unchanged.
        """
        frame = loop.step()
        snap = self.snapshot()
        if frame is not None:
            try:
                w, x, y, z = frame.rgba[-4:] if len(frame.rgba) >= 4 else (b"\x00\x00\x00\xff",)
                if isinstance(w, int):
                    a_log = math.log(max(1, frame.rgba[-1]))
                    w_v, z_v, y_v, x_v = self._h4(frame.rgba)
                else:
                    w_v = z_v = y_v = x_v = 0.0
            except Exception:
                w_v = z_v = y_v = x_v = 0.0
            snap["_ivl_frame"] = {
                "t": frame.t,
                "seed": frame.seed,
                "prompt": frame.prompt,
                "h3_latency_ms": frame.h3_latency_ms,
                "size_bytes": frame.size_bytes,
                "rgba_sha256": frame.sha256,
                "w": w_v,
                "x": x_v,
                "y": y_v,
                "z": z_v,
            }
        else:
            snap["_ivl_frame"] = None
        return snap

    def feed_ivl_batch(self, loop, ticks: int):
        """Step the InfiniteVideoLoop `ticks` times."""
        out = []
        for _ in range(int(ticks)):
            snap = self.feed_ivl_tick(loop)
            out.append(snap)
        return self.snapshot()

    @staticmethod
    def _h4(rgba: bytes) -> tuple[float, float, float, float]:
        """Compute H4 (W/Z/Y/X) channels from an RGBA frame.

        Uses the last pixel (a, r, g, b) as a sample.
        W = log(alpha)
        X = linear red
        Y = linear green
        Z = linear blue
        """
        import math as _m
        if len(rgba) < 4:
            return (0.0, 0.0, 0.0, 0.0)
        a_raw = rgba[-1]
        r_raw = rgba[-4]
        g_raw = rgba[-3]
        b_raw = rgba[-2]
        a_log = _m.log(max(1, a_raw))
        # Apply Hadamard gate: (a_log, b, g, r) -> (W, Z, Y, X)
        from ..qbf import h4_gate
        w, z, y, x = h4_gate((a_log, float(b_raw),
                              float(g_raw), float(r_raw)))
        return (w, z, y, x)

    def snapshot(self) -> dict:
        pipe = self.pipeline
        sub_bus = pipe.sub.bus.snapshot() if pipe.sub else {}
        con_bus = pipe.con.bus.snapshot() if pipe.con else {}
        sub_series = {k: list(v) for k, v in (pipe.sub.series if pipe.sub else {}).items()}
        con_series = {k: list(v) for k, v in (pipe.con.series if pipe.con else {}).items()}
        br = pipe.bridge
        return {
            "t": self._tick_count,
            "running": self._running,
            "sub": {"bus": sub_bus, "series": sub_series},
            "con": {"bus": con_bus, "series": con_series},
            "bridge": {
                "depth": br.depth(),
                "latency": br.latency,
                "pushed": br._pushed,
                "popped": br._popped,
                "queued": len(br._q),
                "history": list(self._depth_history),
                "use_h4": br.use_h4,
            },
            "views": self.view_layout,
            "window": 512,
        }

    def ws_stats(self) -> dict:
        with self._lock:
            return {
                "clients": len(self._clients),
                "drops": {cid: self._client_drops.get(cid, 0)
                          for cid in range(len(self._clients))},
            }

    async def ws_connect(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._ws_queue_max)
        with self._lock:
            idx = len(self._clients)
            self._clients.append(q)
            self._client_drops[idx] = 0
        return q

    def ws_disconnect(self, q: asyncio.Queue):
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def ws_broadcast(self, msg: dict):
        payload = json.dumps(msg)
        for q in list(self._clients):
            try:
                q.put_nowait(payload)
            except Exception:
                self._clients.remove(q)

    @classmethod
    def get(cls, name: str) -> "BicameralViewer | None":
        return cls._registry.get(name)

    @classmethod
    def put(cls, name: str, viewer: "BicameralViewer"):
        cls._registry[name] = viewer

    @classmethod
    def delete(cls, name: str):
        cls._registry.pop(name, None)