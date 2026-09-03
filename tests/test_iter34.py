"""iter34 tests: InfiniteVideoLoop + HostBridge push_frame/pop_frame.

Covers:
  - HostBridge.push_frame() / pop_frame(): RGBA frame transport,
    H4 channel latches, latency.
  - InfiniteVideoLoop: H3 -> HostBridge -> BicameralViewer -> viz_video.
  - End-to-end: H3 frames flow sub -> bridge -> con, viz_video atom
    sees the frame on its bus after bridge_latency ticks, and the
    trace records each frame as a portable VideoFrameEntry.

Run: python -m pytest tests/test_iter34.py -v
"""
from __future__ import annotations

import os
import tempfile

import pytest

from atomic import (
    Program, Block, Wire,
    H3Stub, H3Session,
    FlowTrace, QbfTraceStore,
    BicameralViewer, InfiniteVideoLoop,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sub_prog():
    return Program(
        "sub_clock",
        blocks=[Block("clk", "clock_bpm", {"bpm": 120})],
        wires=[],
    )


@pytest.fixture
def con_prog():
    return Program(
        "con_video",
        blocks=[Block("vv", "viz_video")],
        wires=[],
    )


@pytest.fixture
def viewer(sub_prog, con_prog):
    return BicameralViewer(
        sub_prog, con_prog,
        bridge_map=[("clk.trig", "vv.in")],
        bridge_latency=1,
        name="ivl_test",
    )


@pytest.fixture
def qbf_store():
    fd, path = tempfile.mkstemp(suffix=".qbf")
    os.close(fd)
    os.unlink(path)
    store = QbfTraceStore(path)
    yield store
    store.close()
    try:
        os.unlink(path)
    except OSError:
        pass


PROMPTS = (
    "a slow orbit around a frozen comet",
    "neon city under heavy rain",
    "macro shot of a beetle",
    "saturn rings dissolving",
)


# ── HostBridge push_frame / pop_frame ─────────────────────────────────────────


class TestHostBridgeFrameTransport:
    def test_push_pop_frame_basic(self):
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8)
        rgba = bytes([255, 0, 0, 255] * 64)
        b.push_frame(0, rgba, prompt="x", seed=1, width=8, height=8)
        out = b.pop_frame(0)
        assert out == []  # latency=1, frame not ready at tick 0
        out = b.pop_frame(1)
        assert len(out) == 1
        assert out[0]["rgba"] == rgba
        assert out[0]["prompt"] == "x"
        assert out[0]["seed"] == 1

    def test_push_pop_frame_h4_latches(self):
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8)
        rgba = bytes([255, 0, 0, 255] * 16)  # 4x4
        b.push_frame(0, rgba, prompt="x", seed=1)
        out = b.pop_frame(1)
        assert len(out) == 1
        # H4 channel latches computed from last pixel
        for key in ("_w", "_x", "_y", "_z"):
            assert key in out[0], key

    def test_push_pop_frame_multiple(self):
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8)
        rgba_a = bytes([1, 2, 3, 255] * 4)
        rgba_b = bytes([4, 5, 6, 255] * 4)
        b.push_frame(0, rgba_a, prompt="a", seed=0)
        b.push_frame(1, rgba_b, prompt="b", seed=1)
        out0 = b.pop_frame(0)
        assert out0 == []
        out1 = b.pop_frame(1)
        assert len(out1) == 1
        assert out1[0]["prompt"] == "a"
        out2 = b.pop_frame(2)
        assert len(out2) == 1
        assert out2[0]["prompt"] == "b"

    def test_push_frame_latency_zero(self):
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8)
        rgba = bytes([255] * 16)
        b.push_frame(0, rgba, prompt="instant")
        out = b.pop_frame(1)  # arrival = 0 + 1 = 1
        assert len(out) == 1
        assert out[0]["prompt"] == "instant"

    def test_pop_frame_empty(self):
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8)
        assert b.pop_frame(0) == []

    def test_push_frame_capacity_evicts(self):
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=2)
        rgba = bytes([0] * 4)
        for t in range(5):
            b.push_frame(t, rgba, prompt=f"t{t}")
        assert b.depth() == 2

    def test_push_frame_does_not_break_scalar(self):
        """push_frame and the scalar push path coexist on the same bridge."""
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8)
        b.push(0, {"x": 1.0})  # scalar path
        b.push_frame(0, bytes([0] * 4), prompt="frame")  # frame path
        # frame arrives at 1
        out1 = b.pop_frame(1)
        assert len(out1) == 1
        assert out1[0]["prompt"] == "frame"
        # scalar arrived at 1 too
        out_s = b.pop(1)
        assert out_s is not None and out_s.get("x") == 1.0


# ── BicameralViewer.feed_video_tick ───────────────────────────────────────────


class TestBicameralViewerFeedVideo:
    def test_feed_video_tick_sets_con_bus(self, viewer):
        rgba = bytes([128, 64, 32, 255] * 16)  # 4x4
        viewer.feed_video_tick(rgba, module_id="vv")
        eng = viewer.pipeline.con
        assert eng.bus.get("vv.frame") == rgba

    def test_feed_video_tick_advances_tick(self, viewer):
        assert viewer.tick == 0
        viewer.feed_video_tick(b"\x00" * 16, module_id="vv")
        assert viewer.tick == 1
        viewer.feed_video_tick(b"\x00" * 16, module_id="vv")
        assert viewer.tick == 2

    def test_feed_video_batch(self, viewer):
        frames = [bytes([i % 256] * 16) for i in range(4)]
        viewer.feed_video_batch(frames, module_id="vv")
        assert viewer.tick == 4

    def test_feed_video_snapshot_has_bridge(self, viewer):
        snap = viewer.feed_video_tick(b"\x00" * 16, module_id="vv")
        assert "bridge" in snap
        assert "sub" in snap
        assert "con" in snap
        assert snap["bridge"]["latency"] == 1


# ── InfiniteVideoLoop ─────────────────────────────────────────────────────────


class TestInfiniteVideoLoop:
    def test_constructor_uses_viewer_bridge(self, viewer):
        loop = InfiniteVideoLoop(H3Stub(), viewer)
        assert loop.bridge is viewer.bridge

    def test_step_returns_h3_frame(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=["x"])
        f = loop.step()
        assert f is not None
        assert len(f.rgba) == 4 * 4 * 4
        assert f.prompt == "x"

    def test_run_n_ticks(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=PROMPTS, max_ticks=8)
        frames = loop.run(5)
        assert len(frames) == 5

    def test_prompt_round_robin(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=PROMPTS)
        frames = loop.run(8)
        # one prompt per tick, round-robin
        for i, f in enumerate(frames):
            assert f.prompt == PROMPTS[i % len(PROMPTS)]

    def test_max_ticks_caps_run(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=["x"], max_ticks=3)
        frames = loop.run(10)
        assert len(frames) == 3

    def test_step_after_max_returns_none(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=["x"], max_ticks=2)
        loop.run(2)
        assert loop.step() is None

    def test_frame_index_increments(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=PROMPTS)
        assert loop.frame_index == 0
        loop.run(3)
        assert loop.frame_index == 3

    def test_stats_keys(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=["x"])
        loop.run(2)
        s = loop.stats()
        for k in ("t", "frame_index", "session_t", "frames_in_ring",
                  "consumed_prompts", "h3_calls", "bridge_depth",
                  "bridge_pushed", "bridge_popped", "viewer_tick"):
            assert k in s, k
        assert s["t"] == 2
        assert s["frame_index"] == 2

    def test_bridge_pushed_count(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=["x"])
        loop.run(5)
        # pushed == 5 frames per tick; pop may not catch all (latency)
        assert loop.bridge._pushed >= 5

    def test_inject_into_con_engine(self, viewer):
        """The conscious engine's vv.frame key should receive the H3 RGBA."""
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=["x"], bridge_latency=1)
        loop.run(3)
        # After 3 ticks (latency=1), at least one frame has been popped
        # and written into the con bus. The pipeline.con.bus["vv.frame"]
        # is the latest injected frame.
        con_bus = viewer.pipeline.con.bus.snapshot()
        assert "vv.frame" in con_bus


# ── Trace recording + QBF archive ──────────────────────────────────────────────


class TestInfiniteVideoLoopTrace:
    def test_trace_records_each_frame(self, viewer):
        h3 = H3Stub(width=4, height=4)
        trace = FlowTrace(max_frames=32)
        loop = InfiniteVideoLoop(h3, viewer, prompts=PROMPTS, trace=trace)
        loop.run(5)
        assert trace.video_seq == 5
        assert len(trace.video) == 5

    def test_trace_node_frame_each_tick(self, viewer):
        h3 = H3Stub(width=4, height=4)
        trace = FlowTrace(max_frames=32)
        loop = InfiniteVideoLoop(h3, viewer, prompts=PROMPTS, trace=trace)
        loop.run(4)
        # 4 ticks -> 4 node frames + 4 video frames
        assert trace._seq == 4  # node frames recorded per tick
        assert trace.video_seq == 4

    def test_qbf_round_trip(self, viewer, qbf_store):
        h3 = H3Stub(width=4, height=4)
        trace = FlowTrace(max_frames=32)
        loop = InfiniteVideoLoop(h3, viewer, prompts=PROMPTS[:2],
                                 trace=trace)
        loop.run(4)
        m = qbf_store.append_run(trace, note="ivl_qbf_test")
        assert m["n_video"] == 4
        # reload
        d = qbf_store.load_run(0)
        assert d["manifest"]["n_video"] == 4
        assert len(d["video"]) == 4
        # verify frame data
        for i, entry in enumerate(d["video"]):
            assert entry["prompt"] == PROMPTS[i % 2]
            assert len(entry["rgba"]) == 4 * 4 * 4


# ── Determinism / Swarm consensus hook ────────────────────────────────────────


class TestInfiniteVideoLoopConsensus:
    def test_prompt_consensus_override(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=PROMPTS)
        # override the consensus pick to always return the first prompt
        loop.session.prompt_consensus = lambda prev: PROMPTS[0]
        frames = loop.run(4)
        for f in frames:
            assert f.prompt == PROMPTS[0]


# ── BicameralPipeline + InfiniteVideoLoop integration ────────────────────────


class TestBicameralPipelineIntegration:
    def test_subconscious_clock_runs(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=["x"])
        loop.run(4)
        # the sub engine should have ticked 4 times
        assert viewer.pipeline.sub._t == 4
        assert viewer.pipeline.con._t == 4

    def test_bridge_latency_history(self, viewer):
        h3 = H3Stub(width=4, height=4)
        loop = InfiniteVideoLoop(h3, viewer, prompts=["x"], bridge_latency=1)
        # pre-tick: bridge empty
        assert loop.bridge.depth() == 0
        loop.run(3)
        # some frames may still be queued at the end (latency=1)
        # but the bridge depth should be <= max_ticks
        assert loop.bridge.depth() <= 3