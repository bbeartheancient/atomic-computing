"""QBF persistence round-trip (goal 6, the operator's "middle format").

Exercises the .qbf trace store end-to-end on one trace, then proves the
load side reproduces the run (bit-identical replay, the contract the
test suite pins).

Pinned: a 60-tick run of a tiny program (const -> gain -> viz_series),
record a FlowTrace, archive it as run 0 of a fresh .qbf shard under a
temp dir, then read it back and replay it. The replay's final bus must
equal the original's; the manifest must agree with the original
snapshot.

  ~/runtime/.venv/bin/python -m examples.qbf_persistence_round_trip
"""

from __future__ import annotations

import os
import sys
import tempfile

# allow `python examples/qbf_persistence_round_trip.py` too
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from atomic.engine import Engine
from atomic.program import Block, Program, Wire
from atomic.qbfstore import QbfTraceStore, close_all
from atomic.trace import FlowTrace


def _build_program() -> Program:
    return Program(
        "qbf_round_trip",
        description="const -> gain -> viz_series (qbf round-trip demo)",
        blocks=[
            Block("c0", "const", {"value": 1.5}),
            Block("g1", "gain", {"factor": 2.0}),
            Block("v0", "viz_series"),
        ],
        wires=[Wire("c0.cv", "g1.in"),
               Wire("g1.cv", "v0.in")],
    )


def demo(ticks: int = 60, dt: float = 1.0 / 30.0):
    """Record a trace, archive it to .qbf, load + replay, return the
    final bus of the replay (== original by contract 8).

    Also asserts: manifest tick count, replay-final == recorded-final,
    shard has exactly one run, and export_run() round-trips the
    FlowTrace.export() output (the dma_trace twin shape).
    """
    prog = _build_program()
    patch = prog.compile("microfx")
    modules = patch["modules"]
    wires = patch["wires"]
    views = patch.get("views") or []

    # 1) Record the original run.
    trace = FlowTrace()
    eng = Engine(modules, wires, views=views, dt=dt, trace=trace)
    original = eng.run(ticks)
    original_final = dict(original["bus"])
    original_snapshot = trace.snapshot()
    original_export = trace.export()

    # 2) Archive to a fresh .qbf shard in a temp dir.
    td = tempfile.mkdtemp(prefix="atomic_qbf_demo_")
    shard_path = os.path.join(td, "round_trip.qbf")
    try:
        store = QbfTraceStore(shard_path)
        # explicit program payload so replay_run is self-contained
        program_payload = {"modules": modules, "wires": wires,
                           "views": views}
        manifest = store.append_run(trace, dt=dt, note="round-trip demo",
                                    program=program_payload)

        # 3) Assertions on the archive.
        assert manifest["n_ticks"] == ticks, (manifest["n_ticks"], ticks)
        assert manifest["n_frames"] == len(trace.frames)
        runs = store.runs()
        assert runs == [0], runs

        # 4) load_run() reproduces the snapshot exactly.
        loaded = store.load_run(0)
        assert loaded["manifest"] == manifest
        assert loaded["ticks"] == original_snapshot["ticks"]
        assert loaded["frames"] == [dict(f) for f in original_snapshot["frames"]]

        # 5) export_run() == original trace export (the dma_trace twin).
        reloaded_export = store.export_run(0)
        assert reloaded_export == original_export, \
            "export_run diverged from original trace.export"

        # 6) replay_run() drives a fresh engine: final bus must match.
        replayed = store.replay_run(0, modules=modules, wires=wires,
                                    views=views)
        replayed_final = dict(replayed["bus"])
        assert replayed_final == original_final, (
            f"replay diverged:\n  original={original_final}\n"
            f"  replayed={replayed_final}")

        # 7) flow_trace() yields a working FlowTrace we can re-snapshot.
        rebuilt = store.flow_trace(0)
        rebuilt_snap = rebuilt.snapshot()
        assert rebuilt_snap["n_frames"] == original_snapshot["n_frames"]
        assert rebuilt_snap["n_ticks"] == original_snapshot["n_ticks"]
        assert rebuilt.export() == original_export

        print("[qbf round-trip] ok -- %d ticks, %d frames archived as run 0"
              % (manifest["n_ticks"], manifest["n_frames"]))
        print("  shard:    %s" % shard_path)
        print("  final:    %s" % {k: round(v, 6) for k, v in original_final.items()
                                 if isinstance(v, (int, float))})
        print("  manifest: ticks=%d frames=%d seq=%d" % (
            manifest["n_ticks"], manifest["n_frames"], manifest["seq"]))
    finally:
        close_all()
        # keep the shard around for inspection
        return {"shard": shard_path, "manifest": manifest,
                "original_final": original_final, "replayed_final": replayed_final}


if __name__ == "__main__":
    demo()
