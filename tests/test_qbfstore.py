"""Goal 6 on QBF: the dma_trace-style trace JSON wired into the
portable .qbf shard -- atomized as blobs (index, per-run manifest,
per-tick stimulus, one blob per node frame), lossless end to end.

Pinned contracts:

  1. append_run -> load_run: the snapshot() dict round-trips exactly
  2. the dma_trace-style export JSON is identical through the .qbf
     shard (the sibling dma_trace/export parity, now portable)
  3. replay_run with the stored program is bit-identical (contract 8),
     incl. the conformance fact acc.acc == 2
  4. multiple runs coexist in one shard; the index head orders them
  5. an oversized frame (>2.2 KB, the .mv2 poison line) stores clean
  6. the H(4) gate stays OPTIONAL: trace blobs are plain JSON, not
     H4-gated (the gate is a separate, format-level capability)
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomic import (Engine, FlowTrace, QbfError, QbfFile, JSON, H4)
from atomic.qbfstore import close_all, open_trace_store
from test_trace import COUNTER

DT = 1 / 30


@pytest.fixture
def shard_dir(tmp_path):
    yield tmp_path
    close_all()


def traced_counter(ticks=60, ui_taps=(5, 30)):
    trace = FlowTrace()
    eng = Engine(COUNTER["modules"], COUNTER["wires"], dt=DT,
                 ui_taps=list(ui_taps), trace=trace)
    return trace, eng.run(ticks)


def test_append_load_roundtrip(shard_dir):
    trace, _ = traced_counter()
    store = open_trace_store("counter", shard_dir)
    store.append_run(trace, dt=DT, program=COUNTER)
    snap = trace.snapshot()
    data = store.load_run(0)
    assert data["manifest"]["n_frames"] == snap["n_frames"] == 120
    assert data["manifest"]["seq"] == snap["seq"] == 120
    assert data["ticks"] == snap["ticks"]
    assert data["frames"] == snap["frames"]


def test_export_parity_through_qbf(shard_dir):
    trace, _ = traced_counter()
    store = open_trace_store("parity", shard_dir)
    store.append_run(trace, dt=DT, program=COUNTER)
    assert store.export_run(0) == trace.export()


def test_replay_from_stored_program(shard_dir):
    trace, orig = traced_counter()
    assert orig["final"]["acc.acc"] == 2
    store = open_trace_store("replay", shard_dir)
    store.append_run(trace, dt=DT, program=COUNTER)
    again = store.replay_run(0)
    assert again["final"] == orig["final"]
    assert again["final"]["acc.acc"] == 2
    assert store.flow_trace(0).snapshot()["seq"] == trace.snapshot()["seq"]


def test_two_runs_one_shard(shard_dir):
    t1, _ = traced_counter(20, (5,))
    t2, _ = traced_counter(30, (10, 25))
    store = open_trace_store("multi", shard_dir)
    store.append_run(t1, dt=DT, program=COUNTER)
    store.append_run(t2, dt=DT, program=COUNTER)
    assert store.runs() == [0, 1]
    assert store.load_run(0)["manifest"]["n_frames"] == 40
    assert store.load_run(1)["manifest"]["n_frames"] == 60
    assert store.load_run(1)["manifest"]["seq"] == 60


def test_replay_needs_program_or_args(shard_dir):
    trace, _ = traced_counter(10, (5,))
    store = open_trace_store("noprog", shard_dir)
    store.append_run(trace, dt=DT)
    try:
        store.replay_run(0)
        assert False, "a run stored without a program must raise"
    except QbfError:
        pass
    again = store.replay_run(0, modules=COUNTER["modules"],
                              wires=COUNTER["wires"])
    assert again["final"]["acc.acc"] == 1


def test_oversized_frame_no_poison(shard_dir):
    trace = FlowTrace()
    big_in = {"in%03d" % i: i * 0.25 for i in range(300)}
    trace.begin_tick(0, 0)
    trace.record_node("sink", "sink", big_in, {}, 1.0, t=0)
    assert len(json.dumps(trace.snapshot())) > 2200   # the .mv2 poison line
    store = open_trace_store("big", shard_dir)
    store.append_run(trace, program=None)
    assert store.load_run(0)["frames"][0]["in_ports"] == big_in


def test_h4_optional_off_by_default(shard_dir):
    trace, _ = traced_counter(10, (5,))
    store = open_trace_store("plain", shard_dir)
    store.append_run(trace, dt=DT, program=COUNTER)
    f = QbfFile.open(store.path)
    for name in f.names():
        assert f.info(name)["type"] != H4, name      # the gate stays OFF
    assert f.info("r0000/manifest")["type"] == JSON
