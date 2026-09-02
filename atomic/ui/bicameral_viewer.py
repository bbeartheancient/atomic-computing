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