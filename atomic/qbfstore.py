"""QbfTraceStore: goal 6 -- "the trace is the bridge", made portable.

Wires the dma_trace-style flow trace (atomic.trace) into a .qbf
(Quantum Blob Format, see qbf.py) shard -- the "middle" container: a
working named-blob mechanism, 64-bit sizes (no 50 MB tier), no search
index to poison, and the H(4) gate available as an optional per-blob
flag. The trace store keeps the gate OFF: trace blobs are plain JSON,
because replay demands the exact original stimulus bytes.

Each run is atomized into blobs in one shard:

  index                  {"runs": [0, 1, ...]}     (head pointer)
  r%04d/manifest         run metadata: counts, dt, recorded tick
                         indices, and the program patch (so replay
                         is self-contained)
  r%04d/ticks            the recorded per-tick stimulus, one JSON blob
  r%04d/f%06d            one blob per recorded node frame

load_run() reassembles the exact snapshot() dict; flow_trace()
rebuilds a live FlowTrace from it; replay_run() drives a fresh
engine from the stored stimulus (contract 8: bit-identical, incl.
the conformance fact acc.acc == 2 under the COUNTER patch). append_run()
rewrites the shard (simple + correct at archive scale; a streaming
append mode is a future extension).

One live store per path per process (a registry); close_all()
releases every open shard. Default dir: ~/.runtime/atomic_qbf
(override with the ATOMIC_QBF_DIR env var).
"""

import os
from pathlib import Path

from .qbf import QbfFile, QbfError
from .trace import FlowTrace, replay

__all__ = ["QbfTraceStore", "open_trace_store", "close_all", "DEFAULT_DIR"]

DEFAULT_DIR = os.environ.get(
    "ATOMIC_QBF_DIR", str(Path.home() / ".runtime" / "atomic_qbf"))

_REGISTRY = {}


class QbfTraceStore:
    """A .qbf shard holding one or more archived flow traces."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = None

    @property
    def file(self):
        """The open QbfFile behind this store (created on first use)."""
        if self._file is None:
            self._file = (QbfFile.open(self.path) if self.path.exists()
                          else QbfFile.create(self.path))
        return self._file

    def close(self):
        self._file = None

    # -- archive ----------------------------------------------------------------

    def append_run(self, trace, dt=None, note="", program=None):
        """Snapshot the trace and archive it as run <rid>.

        program is the patch dict (modules/wires/views) so
        replay_run can re-drive the stored stimulus without it.
        Returns the stored manifest dict.
        """
        snap = trace.snapshot()
        if not snap["frames"]:
            raise QbfError("refusing to archive an empty trace")
        f = self.file
        runs = f.get_json("index")["runs"] if "index" in f else []
        rid = max(runs) + 1 if runs else 0
        manifest = {
            "run_id": rid, "note": note, "dt": dt,
            "seq": snap["seq"], "vseq": snap.get("vseq", 0),
            "n_ticks": snap["n_ticks"],
            "n_frames": snap["n_frames"],
            "n_video": snap.get("n_video", 0),
            "max_frames": snap["max_frames"],
            "tick_ts": [tick["t"] for tick in snap["ticks"]],
            "program": program,
        }
        f.put_json("index", {"runs": runs + [rid]})
        f.put_json("r%04d/manifest" % rid, manifest)
        f.put_json("r%04d/ticks" % rid, snap["ticks"])
        for i, fr in enumerate(snap["frames"]):
            f.put_json("r%04d/f%06d" % (rid, i), fr)
        # iter 33 Aspect 4: store video frames as separate blobs
        n_vid = snap.get("n_video", 0)
        for i, vf in enumerate(snap.get("video") or []):
            # Store rgba as base64 so JSON can round-trip
            import base64
            vf_stored = dict(vf)
            if "rgba" in vf_stored and isinstance(vf_stored["rgba"], bytes):
                vf_stored["rgba_b64"] = base64.b64encode(
                    vf_stored.pop("rgba")).decode("ascii")
            f.put_json("r%04d/v%06d" % (rid, i), vf_stored)
        f.write()
        return manifest

    def runs(self):
        """The run ids archived in this shard, in append order."""
        f = self.file
        return list(f.get_json("index")["runs"]) if "index" in f else []

    # -- read back ---------------------------------------------------------------

    def load_run(self, rid):
        """Reassemble run rid: {'manifest', 'ticks', 'frames', 'video'} --
        exactly the snapshot() dict shape (frames as plain dicts, not
        FrameEntry)."""
        f = self.file
        m = f.get_json("r%04d/manifest" % rid)
        ticks = f.get_json("r%04d/ticks" % rid)
        if len(ticks) != m["n_ticks"]:
            raise QbfError("run %d tick count mismatch" % rid)
        frames = [f.get_json("r%04d/f%06d" % (rid, i))
                   for i in range(m["n_frames"])]
        # iter 33 Aspect 4: load video frames
        import base64
        video = []
        for i in range(m.get("n_video", 0)):
            vf = f.get_json("r%04d/v%06d" % (rid, i))
            if vf and "rgba_b64" in vf:
                vf = dict(vf)
                vf["rgba"] = base64.b64decode(vf.pop("rgba_b64"))
            video.append(vf)
        return {"manifest": m, "ticks": ticks, "frames": frames, "video": video}

    def flow_trace(self, rid):
        """A live FlowTrace rebuilt from run rid (replay-ready)."""
        d = self.load_run(rid)
        m = d["manifest"]
        snap = {
            "active": True,
            "seq": m["seq"],
            "vseq": m.get("vseq", 0),
            "n_ticks": m["n_ticks"],
            "n_frames": m["n_frames"],
            "n_video": m.get("n_video", 0),
            "max_frames": m["max_frames"],
            "ticks": d["ticks"],
            "frames": d["frames"],
            "video": d["video"],
        }
        return FlowTrace.from_snapshot(snap)

    def export_run(self, rid, path=None):
        """The dma_trace-style JSON of a stored run -- byte-identical
        to the original trace.export() (the whole point of goal 6)."""
        return self.flow_trace(rid).export(path)

    def replay_run(self, rid, modules=None, wires=None, views=None):
        """Re-run a stored run on a fresh engine (bit-identical).

        The program patch comes from the stored manifest unless
        modules/wires are passed explicitly.
        """
        m = self.load_run(rid)["manifest"]
        prog = m.get("program") or {}
        if modules is None:
            if not prog:
                raise QbfError("run %d stored no program; "
                                "pass modules/wires explicitly" % rid)
            modules = prog["modules"]
            if wires is None:
                wires = prog.get("wires", [])
            if views is None:
                views = prog.get("views") or []
        if wires is None:
            wires = []
        if views is None:
            views = []
        return replay(self.flow_trace(rid), modules, wires,
                      views=views, dt=m.get("dt") or 1.0 / 30.0)

    def __repr__(self):
        return "QbfTraceStore(%r)" % str(self.path)


def open_trace_store(name, shard_dir=None):
    """Open (or create) the named .qbf shard under shard_dir (default
    DEFAULT_DIR). One live handle per path per process."""
    base = Path(shard_dir) if shard_dir else Path(DEFAULT_DIR)
    path = base / ("%s.qbf" % name)
    key = str(path)
    if key in _REGISTRY:
        return _REGISTRY[key]
    base.mkdir(parents=True, exist_ok=True)
    store = QbfTraceStore(path)
    _REGISTRY[key] = store
    return store


def close_all():
    """Release every open shard in the process registry."""
    for store in _REGISTRY.values():
        store.close()
    _REGISTRY.clear()
