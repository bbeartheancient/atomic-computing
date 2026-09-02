"""Bridge: bicameral two-GPU split (goal 9).

Hardware reality (BICAMERAL_FRAMEWORK.md §1): the two B70s share no
P2P — every cross-device transfer goes through host RAM. The bridge
is a host-RAM FIFO with a fixed tick latency (measured: 1 tick per
hop, same as wire latch) plus an optional H(4) compact flag that
mirrors the QSA/DCT compact contract (row_cos gate).

The harness does not need real GPUs: each half is an isolated Engine
with its own bus/state/trace, and the bridge is a tick-indexed queue
(the host RAM) that copies scalars between them. Determinism is
preserved: same programs + same latency => bit-identical pair of
final buses.

API:
  HostBridge(latency=1, capacity=64, use_h4=False)
    push(tick, payload: dict)   # payload = {port: value}
    pop(tick) -> dict | None    # returns payload whose arrival == tick
    depth(tick) -> int

  BicameralPipeline(sub_program, con_program,
                    bridge_map=[("pre.w","dec.in")],
                    bridge_latency=1, use_h4=False, dt=1/30)
    tick() -> None
    run(ticks) -> {sub:{final,series}, con:{final,series}, bridge_depth}
    snapshot() helpers

Bridge ports are qualified bus keys: "block.port". The subconscious
engine writes them; the pipeline reads bus.get(key) after the sub
tick and enqueues the dict; when it arrives, the conscious engine's
target node inputs are seeded BEFORE its tick (so the conscious sees
the value with exactly bridge_latency + the sub's 1-tick wire).
"""

from __future__ import annotations

import collections

from .engine import Engine
from .qbf import h4_gate, h4_inverse

__all__ = ["HostBridge", "BicameralPipeline", "BicameralResult",
           "h4_row_cosine", "row_cos_gate", "h4_streaming_metrics",
           "latency_histogram", "bridge_benchmark"]


class HostBridge:
    """Host-RAM FIFO with fixed tick latency."""

    def __init__(self, latency=1, capacity=64, use_h4=False):
        self.latency = max(1, int(latency))
        self.capacity = int(capacity)
        self.use_h4 = bool(use_h4)
        self._q = collections.deque()  # (arrival_tick, payload)
        self._pushed = 0
        self._popped = 0

    def push(self, tick, payload: dict):
        arrival = int(tick) + self.latency
        # optional H4 compact: if exactly 4 scalar values and use_h4, encode
        # them via the H(4) gate and store the gated tuple (lossless via inverse).
        # Smaller payloads are stored verbatim.
        stored = dict(payload)
        if self.use_h4 and len(stored) == 4:
            try:
                vals = [float(v) for v in stored.values()]
                w, z, y, x = h4_gate(tuple(vals))
                keys = list(stored.keys())
                stored = {keys[0]: w, keys[1]: z, keys[2]: y, keys[3]: x, "_h4": True, "_h4_keys": keys}
            except Exception:
                pass
        # iter 27: frame-blob round-trip. A "frame" entry is RGBA bytes
        # (H3 output). The bridge stores it verbatim, then a viz_video
        # sink (or its renderer) reads it off the bus. We tag the entry
        # so downstream consumers know it's a frame, not a scalar dict.
        if "frame" in stored and isinstance(stored["frame"], (bytes, bytearray)):
            stored["frame"] = bytes(stored["frame"])
            stored["_frame"] = True
            # also surface the W (log-alpha) and X/Y/Z (linear RGB)
            # channel decoder so consumers can sample a single channel
            # without re-decoding the whole frame.
            try:
                import math as _m
                # 4 bytes from the end -> (a, r, g, b) sample
                if len(stored["frame"]) >= 4:
                    j = len(stored["frame"]) - 4
                    a_raw = stored["frame"][j + 3]
                    r_raw = stored["frame"][j]
                    g_raw = stored["frame"][j + 1]
                    b_raw = stored["frame"][j + 2]
                    a_log = _m.log(max(1, a_raw))
                    w, z, y, x = h4_gate((a_log, float(b_raw), float(g_raw), float(r_raw)))
                    stored["_w"] = w  # log-alpha (amplitude)
                    stored["_x"] = x  # linear red
                    stored["_y"] = y  # linear green
                    stored["_z"] = z  # linear blue
            except Exception:
                pass
        if len(self._q) >= self.capacity:
            self._q.popleft()
        self._q.append((arrival, stored))
        self._pushed += 1

    def pop(self, tick):
        out = {}
        while self._q and self._q[0][0] <= int(tick):
            _, payload = self._q.popleft()
            # de-gate if H4 was applied
            if payload.get("_h4"):
                try:
                    keys = payload.get("_h4_keys", list(payload.keys()))
                    # keys excludes _h4/_h4_keys
                    keys = [k for k in keys if k not in ("_h4", "_h4_keys")]
                    if len(keys) == 4:
                        w = float(payload[keys[0]])
                        z = float(payload[keys[1]])
                        y = float(payload[keys[2]])
                        x = float(payload[keys[3]])
                        a, b, c, d = h4_inverse((w, z, y, x))
                        out.update({keys[0]: a, keys[1]: b, keys[2]: c, keys[3]: d})
                    else:
                        out.update({k: v for k, v in payload.items() if not k.startswith("_")})
                except Exception:
                    out.update({k: v for k, v in payload.items() if not k.startswith("_")})
            else:
                out.update(payload)
            self._popped += 1
        return out if out else None

    def peek_arrivals(self):
        return [a for a, _ in self._q]

    def depth(self, tick=None):
        if tick is None:
            return len(self._q)
        return sum(1 for a, _ in self._q if a > int(tick))

    def latency_histogram(self):
        """Tick-latency histogram: arrival_tick -> count (for viz)."""
        hist = {}
        for arrival, _ in self._q:
            hist[arrival] = hist.get(arrival, 0) + 1
        # also account for history already popped? use pushed/popped counts
        # For live bridge this shows queued arrivals; for post-run it's empty,
        # so callers should sample during run.
        return dict(sorted(hist.items()))

    def benchmark(self, ticks=100, payload_keys=4):
        """Micro-benchmark: push/pop throughput over `ticks` ticks.

        Returns dict with pushed, popped, queued, elapsed_s, ticks_per_s,
        avg_latency_ticks (which is self.latency by construction), and
        latency_histogram.
        """
        import time as _time
        t0 = _time.perf_counter()
        for t in range(int(ticks)):
            payload = {f"k{i}": float(t + i) for i in range(int(payload_keys))}
            self.push(t, payload)
            self.pop(t)
        elapsed = _time.perf_counter() - t0
        return {
            "ticks": int(ticks),
            "pushed": self._pushed,
            "popped": self._popped,
            "queued": len(self._q),
            "elapsed_s": elapsed,
            "ticks_per_s": (ticks / elapsed) if elapsed > 0 else float("inf"),
            "latency": self.latency,
            "histogram": self.latency_histogram(),
        }

    def to_tiles(self, display=None):
        """Wire bridge histogram to a tiles viz: returns a per-tile heat record.

        If display is given, maps histogram bins to tiles (cyclic). Otherwise
        returns the raw histogram dict.
        """
        hist = self.latency_histogram()
        if display is None or not hist:
            return hist
        tiles = display.tiles
        out = {}
        keys = sorted(hist.keys())
        for i, tk in enumerate(keys):
            tile = tiles[i % len(tiles)]
            out[(tile.row, tile.col)] = hist[tk]
        return out

    def __repr__(self):
        return "HostBridge(latency=%d pushed=%d queued=%d)" % (self.latency, self._pushed, len(self._q))


class BicameralResult:
    def __init__(self, sub, con, bridge, ticks):
        self.sub = sub
        self.con = con
        self.bridge = bridge
        self.ticks = ticks

    def summary(self):
        return {"ticks": self.ticks, "sub_final": self.sub.get("final"),
                "con_final": self.con.get("final"), "bridge_queued": self.bridge.depth()}


class BicameralPipeline:
    """Two engines pipelined through a HostBridge (GPU1 -> host -> GPU0)."""

    def __init__(self, sub_program, con_program,
                 bridge_map=None, bridge_latency=1, use_h4=False,
                 dt=1.0/30.0, sub_views=None, con_views=None):
        self.sub_program = sub_program
        self.con_program = con_program
        self.bridge_map = list(bridge_map or [])  # [(sub_key, con_key/input)]
        self.bridge = HostBridge(latency=bridge_latency, use_h4=use_h4)
        self.dt = float(dt)
        self._t = 0
        # compile once
        self._sub_patch = sub_program.compile("microfx") if hasattr(sub_program, "compile") else sub_program
        self._con_patch = con_program.compile("microfx") if hasattr(con_program, "compile") else con_program
        self.sub = Engine(self._sub_patch["modules"], self._sub_patch.get("wires", []),
                          views=self._sub_patch.get("views") or sub_views or [], dt=self.dt)
        self.con = Engine(self._con_patch["modules"], self._con_patch.get("wires", []),
                          views=self._con_patch.get("views") or con_views or [], dt=self.dt)
        # resolve conscious dest block ids/ports for fast injection
        self._con_targets = []
        for src_key, dst_key in self.bridge_map:
            if "." in dst_key:
                dst_id, dst_port = dst_key.split(".", 1)
                self._con_targets.append((src_key, dst_id, dst_port.lower()))

    def tick(self):
        t = self._t
        # 1) subconscious ticks (GPU1: preprocess / subconscious)
        self.sub.tick()
        # 2) sample bridge sources from sub bus AFTER sub tick + wire latch
        payload = {}
        for src_key, _dst_id, _dst_port in self._con_targets:
            # src_key is a fully qualified bus key like "pre.w"
            v = self.sub.bus.get(src_key)
            if v is not None:
                # use the src port name as dict key; injection uses dst mapping
                payload[src_key] = float(v)
        if payload:
            self.bridge.push(t, payload)
        # 3) deliver anything whose arrival == t (or earlier) into conscious inputs
        incoming = self.bridge.pop(t)
        if incoming:
            for src_key, dst_id, dst_port in self._con_targets:
                if src_key in incoming:
                    node = self.con.by_id.get(dst_id)
                    if node is not None:
                        node.inputs[dst_port] = float(incoming[src_key])
        # 4) conscious ticks (GPU0: decision / conscious)
        self.con.tick()
        self._t += 1

    def run(self, ticks):
        for _ in range(int(ticks)):
            self.tick()
        return BicameralResult(self.sub.snapshot(), self.con.snapshot(), self.bridge, self._t)

    def snapshot(self):
        return {"sub": self.sub.snapshot(), "con": self.con.snapshot(),
                "tick": self._t, "bridge_depth": self.bridge.depth()}

    def metrics(self):
        """H4 + bridge streaming metrics over the current con/sub buses.

        Returns dict with sub_final, con_final, bridge_depth, histogram,
        and h4_streaming sample if any 4-tuple bridge payload is present.
        """
        hist = self.bridge.latency_histogram()
        bm = self.bridge.benchmark(ticks=10) if self.bridge.depth() == 0 else {"histogram": hist}
        # sample H4 over last bridge payload if use_h4
        h4_sample = None
        try:
            # peek last queued payload
            if self.bridge._q:
                _, payload = self.bridge._q[-1]
                if payload.get("_h4"):
                    # show gated values
                    h4_sample = {k: payload[k] for k in payload if not k.startswith("_")}
        except Exception:
            h4_sample = None
        return {"bridge_depth": self.bridge.depth(), "histogram": hist,
                "benchmark": bm, "h4_sample": h4_sample}


# ---------------------------------------------------------------- H4 streaming metrics + row_cos gate

def h4_row_cosine(a, b):
    """Cosine similarity between two equal-length vectors (for H4 rows).

    Row vectors are +1/-1; cosine = dot(a,b) / (||a||*||b||).
    For unit-scale, returns float in [-1, 1]; orthogonal rows -> 0.
    """
    import math as _m
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = _m.sqrt(sum(float(x) * float(x) for x in a))
    nb = _m.sqrt(sum(float(x) * float(x) for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def row_cos_gate(groups, threshold=0.1):
    """The H4 compact contract's row_cos gate: pass if avg row cosine < threshold.

    Groups: iterable of 4-tuples (a,b,c,d) that have been gated (or raw).
    Measure pairwise cosine between the four H4 rows across groups;
    returns (passed: bool, avg_cos: float). Orthogonal H4 rows -> cos ~0 -> passes.
    """
    if not groups:
        return True, 0.0
    # collect rows: each group -> (W,Z,Y,X) via h4_gate
    rows = {"w": [], "z": [], "y": [], "x": []}
    for g in groups:
        w, z, y, x = h4_gate(tuple(g))
        rows["w"].append(w)
        rows["z"].append(z)
        rows["y"].append(y)
        rows["x"].append(x)
    # average absolute cosine between distinct rows
    pairs = [("w", "z"), ("w", "y"), ("w", "x"), ("z", "y"), ("z", "x"), ("y", "x")]
    cos_vals = [abs(h4_row_cosine(rows[a], rows[b])) for a, b in pairs]
    avg = sum(cos_vals) / len(cos_vals) if cos_vals else 0.0
    return (avg < float(threshold)), avg


def h4_streaming_metrics(groups):
    """Streaming view of H4 energy + row_cos over `groups` (4-tuples).

    Returns dict with:
      w_energy_frac (W dominance ~0.61 on random), row_cos_avg,
      row_cos_pass (threshold 0.1), n_groups, w_mean
    """
    if not groups:
        return {"n_groups": 0, "w_energy_frac": 0.0, "row_cos_avg": 0.0,
                "row_cos_pass": True, "w_mean": 0.0}
    gated = [h4_gate(tuple(g)) for g in groups]
    # energy per row
    import math as _m
    energies = [0.0, 0.0, 0.0, 0.0]  # w,z,y,x
    for w, z, y, x in gated:
        energies[0] += w * w
        energies[1] += z * z
        energies[2] += y * y
        energies[3] += x * x
    total = sum(energies) or 1.0
    w_frac = energies[0] / total
    w_mean = sum(g[0] for g in gated) / len(gated)
    passed, avg_cos = row_cos_gate(groups, threshold=0.1)
    return {"n_groups": len(groups), "w_energy_frac": w_frac,
            "row_cos_avg": avg_cos, "row_cos_pass": passed, "w_mean": w_mean}


def latency_histogram(delays):
    """Build a histogram dict from a list of per-hop latencies (ticks).

    Delays: iterable of ints (e.g. measured arrival - push_tick).
    Returns {latency: count} sorted by latency.
    """
    hist = {}
    for d in delays:
        k = int(d)
        hist[k] = hist.get(k, 0) + 1
    return dict(sorted(hist.items()))


def bridge_benchmark(bridge=None, ticks=200, payload_keys=4):
    """Standalone bridge benchmark (host-RAM FIFO, no P2P).

    If bridge is None, creates a fresh HostBridge(). Otherwise reuses the
    given bridge's latency/capacity and reports its histogram + H4 metrics.
    """
    if bridge is None:
        bridge = HostBridge(latency=1, capacity=64)
    res = bridge.benchmark(ticks=ticks, payload_keys=payload_keys)
    # also report H4 gate efficiency if bridge is H4-enabled
    h4_info = None
    if getattr(bridge, "use_h4", False) and payload_keys == 4:
        sample = [(1.0, 2.0, 3.0, 4.0), (2.0, 3.0, 4.0, 5.0)]
        h4_info = h4_streaming_metrics(sample)
    return {**res, "h4": h4_info}
