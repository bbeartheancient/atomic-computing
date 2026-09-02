"""Trace (build-order step 6): the flow observer -- "the trace is the
bridge". Per tick, per node, a FrameEntry in the sibling dma_trace
shape: in_ports latched at tick start, out_ports on the bus at tick
end, plus wall-clock latency_us. Pinned here:

  1. the FrameEntry field set == the sibling dma_trace.FrameEntry
  2. in_ports follow the 1-TICK latency (tap@5 -> acc reads 1.0 at t6)
  3. the tap node's out_ports read the runner-owned "ui.tap" key
     (contract 11; the tap module itself is inert)
  4. a TRACED run is bit-identical to an untraced run on all 6
     conformance patches (the observer never perturbs the signal)
  5. the ring is circular (wraps at max_frames; seq keeps counting)
  6. snapshot/export round-trip through JSON (sibling shape)
  7. start/stop gate the recording
  8. REPLAY: the recorded stimulus (per-tick taps + live params)
     drives a FRESH engine -> bit-identical final + series
     (contract 8), incl. the conformance counter fact acc.acc==2
"""

import dataclasses
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomic import Engine, FrameEntry, FlowTrace, replay, replay_events
from test_parity import PATCHES

DT = 1 / 30

COUNTER = {
    "modules": [
        {"id": "ui", "primitive": "tap", "params": {}},
        {"id": "acc", "primitive": "accum", "params": {"per_tick": 1}}],
    "wires": [{"from": "ui.tap", "to": "acc.in"}],
    "views": [],
}


def test_entry_shape_matches_sibling():
    import fabric.dma_trace as mfdt
    assert {f.name for f in dataclasses.fields(FrameEntry)} == \
        {f.name for f in dataclasses.fields(mfdt.FrameEntry)}


def test_records_per_node_per_tick():
    trace = FlowTrace()
    eng = Engine(COUNTER["modules"], COUNTER["wires"], dt=DT,
                 ui_taps=[5, 30], trace=trace)
    eng.run(8)
    frames = trace.frames
    assert len(frames) == 16             # 2 nodes x 8 ticks
    assert [f.seq for f in frames] == list(range(1, 17))
    by_id = {}
    for f in frames:
        by_id.setdefault(f.module_id, []).append(f)
    assert [f.t for f in by_id["ui"]] == [float(t) for t in range(8)]
    assert by_id["acc"][0].module_type == "accum"


def test_in_ports_one_tick_latency():
    trace = FlowTrace()
    eng = Engine(COUNTER["modules"], COUNTER["wires"], dt=DT,
                 ui_taps=[5, 30], trace=trace)
    eng.run(8)
    acc = [f for f in trace.frames if f.module_id == "acc"]
    assert acc[0].in_ports == {}                  # nothing latched pre-t0
    assert acc[5].in_ports == {"in": 0.0}         # tap@5 not latched yet
    assert acc[6].in_ports == {"in": 1.0}         # latch@end-t5 -> read@t6
    assert acc[6].out_ports == {"acc": 1.0}       # rising edge -> count 1


def test_tap_node_reads_runner_key():
    trace = FlowTrace()
    eng = Engine(COUNTER["modules"], COUNTER["wires"], dt=DT,
                 ui_taps=[5, 30], trace=trace)
    eng.run(8)
    ui = [f for f in trace.frames if f.module_id == "ui"]
    assert ui[4].out_ports == {"tap": 0}
    assert ui[5].out_ports == {"tap": 1}


@pytest.mark.parametrize("label,ticks,ui_taps,tol,patch", PATCHES,
                         ids=[p[0] for p in PATCHES])
def test_trace_is_pure_observer(label, ticks, ui_taps, tol, patch):
    plain = Engine(patch["modules"], patch["wires"],
                   views=patch.get("views") or [], dt=DT,
                   ui_taps=ui_taps).run(ticks)
    trace = FlowTrace()
    traced = Engine(patch["modules"], patch["wires"],
                    views=patch.get("views") or [], dt=DT,
                    ui_taps=ui_taps, trace=trace).run(ticks)
    assert traced["final"] == plain["final"], label
    assert traced["series"] == plain["series"], label
    assert trace.snapshot()["n_frames"] == ticks * len(patch["modules"])


def test_circular_ring_wraps():
    trace = FlowTrace(max_frames=20)
    eng = Engine(COUNTER["modules"], COUNTER["wires"], dt=DT,
                 ui_taps=[5, 30, 55], trace=trace)
    eng.run(10)                        # 2 nodes x 10 ticks == full ring
    assert trace.snapshot()["n_frames"] == 20
    eng.tick()                         # tick 11 evicts the oldest pair
    snap = trace.snapshot()
    assert snap["n_frames"] == 20
    assert snap["seq"] == 22
    assert snap["frames"][0]["seq"] == 3       # seq 1,2 dropped


def test_snapshot_and_export_roundtrip(tmp_path):
    trace = FlowTrace()
    Engine(COUNTER["modules"], COUNTER["wires"], dt=DT,
           ui_taps=[5], trace=trace).run(6)
    snap = trace.snapshot()
    assert set(snap) >= {"active", "seq", "n_frames", "max_frames",
                         "frames", "ticks"}
    assert set(snap["frames"][0]) >= {"seq", "t", "module_id",
                                      "module_type", "in_ports",
                                      "out_ports", "latency_us", "tag"}
    assert json.loads(trace.export()) == snap
    p = tmp_path / "trace.json"
    trace.export(str(p))
    with open(p) as fh:
        assert json.load(fh) == snap


def test_start_stop_gate():
    trace = FlowTrace(active=False)
    assert trace.record_node("a", "const", {}, {"cv": 1.0}, 12.0, 0) is None
    trace.begin_tick(0, 1)
    assert trace.frames == [] and trace.ticks == []
    trace.start()
    e = trace.record_node("a", "const", {}, {"cv": 1.0}, 12.0, 0)
    assert e is not None and e.seq == 1
    trace.stop()
    assert trace.record_node("b", "gain", {"in": 1.0}, {}, 5.0, 1) is None
    assert trace.snapshot()["n_frames"] == 1


def test_replay_counter_matches_original():
    trace = FlowTrace()
    eng = Engine(COUNTER["modules"], COUNTER["wires"], dt=DT,
                 ui_taps=[5, 30], trace=trace)
    orig = eng.run(60)
    assert orig["final"]["acc.acc"] == 2     # the conformance fact
    again = replay(trace, COUNTER["modules"], COUNTER["wires"],
                   views=COUNTER["views"], dt=DT)
    assert again["final"] == orig["final"]
    assert again["final"]["acc.acc"] == 2


def test_replay_live_feeds_matches_original():
    patch = {
        "modules": [
            {"id": "c1", "primitive": "const", "params": {"value": 5}},
            {"id": "g1", "primitive": "gain", "params": {"factor": 2}}],
        "wires": [{"from": "c1.cv", "to": "g1.in"}],
        "views": [],
    }
    feeds = {2: {"taps": [2], "params": {"g1": {"factor": 4.0}}},
             9: {"params": {"g1": {"factor": 0.5}}}}
    trace = FlowTrace()
    eng = Engine(patch["modules"], patch["wires"], dt=DT,
                 ui_taps=None, feeds=feeds, trace=trace)
    orig = eng.run(15)
    assert orig["final"]["g1.cv"] == 2.5     # 5 * 0.5 after the last step
    again = replay(trace, patch["modules"], patch["wires"],
                   views=patch["views"], dt=DT)
    assert again["final"] == orig["final"]
    assert again["final"]["g1.cv"] == 2.5


def test_replay_events_stream():
    trace = FlowTrace()
    Engine(COUNTER["modules"], COUNTER["wires"], dt=DT,
           ui_taps=[2], trace=trace).run(4)
    events = replay_events(trace.snapshot()["frames"])
    ins = [e for e in events if e["direction"] == "in"]
    assert [e["t"] for e in ins] == [1.0, 2.0, 3.0]
    assert [e["value"] for e in ins] == [0.0, 0.0, 1.0]
    assert all(e["module_id"] == "acc" and e["port"] == "in"
               for e in ins)
    outs = [e for e in events if e["direction"] == "out"]
    ui_outs = [e for e in outs if e["module_id"] == "ui"]
    acc_outs = [e for e in outs if e["module_id"] == "acc"]
    assert [e["value"] for e in ui_outs] == [0.0, 0.0, 1.0, 0.0]
    assert [e["value"] for e in acc_outs] == [0.0, 0.0, 0.0, 1.0]
    assert all(e["port"] == "tap" for e in ui_outs)
    assert all(e["port"] == "acc" for e in acc_outs)
