"""Trace: the flow observer -- "the trace is the bridge" (step 6).

Sits on top of the engine (the Python twin of evaluatePatch). Every
tick it records (a) the tick's external stimulus (the resolved
ui.tap + any live param overrides) and (b) one FrameEntry per node:

  - in_ports:    the inputs latched at the START of this node's tick
                 (1-tick latency: the previous tick's bus writes,
                 exactly what input() reads)
  - out_ports:   the node's bus keys at the END of its tick (the
                 declared output ports; for the tap node the
                 runner-owned "ui.tap" key, since the tap module is
                 inert -- contract 11)
  - latency_us:  wall time for this node's tick (metadata only;
                 non-deterministic, never part of a parity compare)
  - seq / t:     the monotonic entry counter / the tick index

The FrameEntry field names mirror the sibling dma_trace.FrameEntry so
a harness trace exports to the same JSON shape the fabric flow
observer emits. One documented divergence: `t` is the tick INDEX
(deterministic, replayable -- the harness is tick-indexed), not the
sibling's wall clock.

The trace is a PURE OBSERVER: recording reads node.inputs / the bus
and writes only into its own rings. It never touches bus values, node
vars, hostState or mem -- a traced run is bit-identical to an
untraced one (pinned by tests/test_trace.py). That is the bicameral
"GPU1 captures / GPU0 decides" split in harness form: the capture
side sees the path the data took, without perturbing the signal.

The rings are circular (maxlen = max_frames) so they never grow
unboundedly. replay() reconstructs the stimulus from the recorded
ticks (per-tick taps -> a fresh ui_taps set; per-tick param
overrides -> feeds) and drives a FRESH engine with it: by
determinism (contract 8) the replayed outputs are identical.

Iter 33 (Aspect 4): the trace also stores a third ring of
VideoFrameEntry records (H3 RGBA frames + provenance). This is the
portable capture path: the .qbf shard round-trips the frames so
a future H3 session can replay the same H4-decoded colors.
"""

import json
from collections import deque
from dataclasses import asdict, dataclass, fields

__all__ = [
    "FrameEntry", "VideoFrameEntry", "FlowTrace",
    "replay", "replay_events",
    "latency_histogram_from_trace",
]


MAX_FRAMES = 10_000


@dataclass
class VideoFrameEntry:
    """A single decoded H3 RGBA frame with provenance metadata."""
    seq: int
    t: int
    rgba: bytes
    prompt: str = ""
    seed: int = 0
    h3_latency_ms: float = 0.0
    width: int = 0
    height: int = 0
    w_gate: float = 0.0
    x_gate: float = 0.0
    y_gate: float = 0.0
    z_gate: float = 0.0


@dataclass
class FrameEntry:
    seq: int
    t: float
    module_id: str
    module_type: str
    in_ports: dict
    out_ports: dict
    latency_us: float
    tag: str = ""


class FlowTrace:
    """Two circular rings: one per-tick stimulus entry, one per-node
    FrameEntry. Both wrap at max_frames; seq counts every entry ever
    recorded. Iter 33: a third optional ring holds VideoFrameEntry
    records (H3 RGBA frames + provenance)."""

    def __init__(self, max_frames=MAX_FRAMES, active=True):
        self.max_frames = int(max_frames)
        self._ticks = deque(maxlen=self.max_frames)
        self._frames = deque(maxlen=self.max_frames)
        self._video = deque(maxlen=self.max_frames)
        self._vseq = 0
        self._seq = 0
        self._active = bool(active)

    # -- control -------------------------------------------------------------

    @property
    def active(self):
        return self._active

    def start(self):
        self._active = True

    def stop(self):
        self._active = False

    def clear(self):
        self._ticks.clear()
        self._frames.clear()
        self._video.clear()
        self._seq = 0
        self._vseq = 0

    # -- recording (called from inside the engine's pinned loop) -------------

    @property
    def ticks(self):
        return list(self._ticks)

    @property
    def frames(self):
        return list(self._frames)

    def begin_tick(self, t, tap, params=None):
        """The resolved external stimulus of tick t."""
        if not self._active:
            return
        self._ticks.append({"t": t, "tap": tap,
                            "params": dict(params) if params else None})

    def record_node(self, module_id, module_type, in_ports, out_ports,
                    latency_us, t=0, tag=""):
        if not self._active:
            return None
        self._seq += 1
        entry = FrameEntry(
            seq=self._seq,
            t=float(t),
            module_id=str(module_id),
            module_type=str(module_type),
            in_ports=dict(in_ports or {}),
            out_ports=dict(out_ports or {}),
            latency_us=float(latency_us),
            tag=str(tag),
        )
        self._frames.append(entry)
        return entry

    def record_video_frame(self, rgba, t, prompt="", seed=0,
                            h3_latency_ms=0.0, width=0, height=0,
                            w_gate=0.0, x_gate=0.0, y_gate=0.0,
                            z_gate=0.0):
        """Record one decoded H3 RGBA frame with provenance metadata.

        Stored in a separate ring (deque, maxlen=max_frames). seq is
        monotonic per trace. The rgba bytes are stored verbatim so the
        H4-decoded colors can be reproduced bit-exact during replay.
        """
        if not self._active:
            return None
        self._vseq += 1
        entry = VideoFrameEntry(
            seq=self._vseq,
            t=int(t),
            rgba=bytes(rgba) if rgba is not None else b"",
            prompt=str(prompt),
            seed=int(seed),
            h3_latency_ms=float(h3_latency_ms),
            width=int(width),
            height=int(height),
            w_gate=float(w_gate),
            x_gate=float(x_gate),
            y_gate=float(y_gate),
            z_gate=float(z_gate),
        )
        self._video.append(entry)
        return entry

    @property
    def video(self):
        return list(self._video)

    @property
    def video_seq(self):
        return self._vseq

    # -- export ----------------------------------------------------------------

    @classmethod
    def from_snapshot(cls, snap):
        """Rebuild a FlowTrace from a snapshot() dict (e.g. one loaded
        back out of a .qbf shard): same shape, fresh rings."""
        tr = cls(max_frames=snap.get("max_frames") or MAX_FRAMES,
                 active=bool(snap.get("active", True)))
        tr._ticks = deque(snap.get("ticks") or [], maxlen=tr.max_frames)
        tr._frames = deque((FrameEntry(**f) for f in (snap.get("frames") or [])),
                           maxlen=tr.max_frames)
        tr._seq = int(snap.get("seq") or len(tr._frames))
        # video frames (iter 33)
        video = snap.get("video") or []
        tr._video = deque(
            (VideoFrameEntry(**_filter_vf_keys(f)) for f in video),
            maxlen=tr.max_frames)
        tr._vseq = int(snap.get("vseq") or len(tr._video))
        return tr

    def snapshot(self):
        return {
            "active": self._active,
            "seq": self._seq,
            "vseq": self._vseq,
            "n_frames": len(self._frames),
            "n_ticks": len(self._ticks),
            "n_video": len(self._video),
            "max_frames": self.max_frames,
            "ticks": list(self._ticks),
            "frames": [asdict(f) for f in self._frames],
            "video": [asdict(f) for f in self._video],
        }

    def export(self, path=None):
        """The snapshot as a JSON string (and written to path if given)
        -- the same shape the sibling dma_trace.export_trace emits."""
        txt = json.dumps(self.snapshot(), indent=2, sort_keys=True)
        if path:
            with open(path, "w") as fh:
                fh.write(txt)
        return txt


def replay(trace, modules, wires, views=(), dt=1.0 / 30.0, atoms=None):
    """Re-run a recorded run from scratch.

    Reconstructs the stimulus the trace captured (per-tick taps -> a
    fresh ui_taps set; per-tick live param overrides -> feeds) and
    drives a FRESH engine with it. By determinism (contract 8: same
    program + same feed sequence == identical outputs) the replayed
    run is bit-identical to the recorded one. Intended for
    contiguous runs (batch run(ticks)); a live run with tick gaps
    re-runs the gaps at the default stimulus, so compare `final` at
    the recorded tick positions only.
    """
    from .engine import Engine
    ui_taps = set()
    feeds = {}
    total = 0
    for tick in trace.ticks:
        total = max(total, tick["t"] + 1)
        if tick["tap"]:
            ui_taps.add(tick["t"])
        if tick["params"]:
            feeds[tick["t"]] = {"params": tick["params"]}
    engine = Engine(modules, wires, views=views, dt=dt,
                    ui_taps=ui_taps, atoms=atoms, feeds=feeds)
    return engine.run(total)


def replay_events(frames):
    """Re-emit the port events of recorded frames (twin of the
    sibling dma_trace.replay): one event per in_port / out_port,
    {seq, module_id, port, value, t, direction} -- the event stream a
    second engine can be driven with for an A/B compare."""
    events = []
    for frame in frames:
        # frames may be FrameEntry dataclass or dict
        if hasattr(frame, "in_ports"):
            in_ports = frame.in_ports or {}
            out_ports = frame.out_ports or {}
            seq = frame.seq
            mid = frame.module_id
            t = frame.t
        else:
            in_ports = frame.get("in_ports") or {}
            out_ports = frame.get("out_ports") or {}
            seq = frame.get("seq")
            mid = frame.get("module_id")
            t = frame.get("t")
        for port, val in in_ports.items():
            events.append({"seq": seq,
                           "module_id": mid,
                           "port": port, "value": val,
                           "t": t, "direction": "in"})
        for port, val in out_ports.items():
            events.append({"seq": seq,
                           "module_id": mid,
                           "port": port, "value": val,
                           "t": t, "direction": "out"})
    return events


def latency_histogram_from_trace(trace):
    """Build a latency histogram from a FlowTrace's per-node latency_us.

    Returns dict {latency_bucket_ms: count} binned by int(latency_us/1000).
    Also returns raw list for bridge_benchmark compatibility.
    """
    buckets = {}
    raw = []
    for fr in trace.frames:
        if hasattr(fr, "latency_us"):
            us = float(fr.latency_us)
        elif isinstance(fr, dict):
            us = float(fr.get("latency_us", 0))
        else:
            continue
        ms = int(us // 1000)
        buckets[ms] = buckets.get(ms, 0) + 1
        raw.append(us)
    return dict(sorted(buckets.items())), raw


def _filter_vf_keys(d):
    """Filter a dict to only the fields VideoFrameEntry accepts."""
    if isinstance(d, dict):
        return {k: v for k, v in d.items()
                if k in {"seq", "t", "rgba", "prompt", "seed",
                         "h3_latency_ms", "width", "height",
                         "w_gate", "x_gate", "y_gate", "z_gate"}}
    return d
