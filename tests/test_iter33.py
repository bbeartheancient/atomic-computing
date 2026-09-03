"""iter33 tests: feed_video, swarm_video, QBF frame trace.

Covers:
  - FeedVideoViewer: feed_video_tick + feed_video_batch
  - REST endpoints: /api/feed_video/{name}/start|stop|push_frame|batch|status
  - WS endpoint: /ws/feed_video/{name}
  - feed_video_live program
  - Swarm H4 consensus -> H3 routing (Aspect 3)
  - QBF frame trace: VideoFrameEntry, FlowTrace.record_video_frame,
    QbfTraceStore append_video_run, load_run with video

Run: python -m pytest tests/test_iter33.py -v
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time

import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from atomic import (
    Program, Block, Wire,
    FlowTrace, VideoFrameEntry,
    QbfTraceStore,
    H3Stub, H3Session,
    Swarm, Agent, Display,
    qbf as _qbf,
)
from atomic.ui.viewer import Viewer
from atomic.ui.programs import build as _build


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def feed_prog():
    return _build("feed_video_live")


@pytest.fixture
def feed_viewer(feed_prog):
    return Viewer(feed_prog, name="feed_video_live")


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


# ── FeedVideoViewer ─────────────────────────────────────────────────────────────

class TestFeedVideoViewer:
    def test_feed_video_tick_single_frame(self, feed_viewer):
        w, h = 8, 8
        rgba = bytes([255, 0, 0, 255] * (w * h))
        feed_viewer.feed_video_tick(rgba, module_id="vv")
        assert feed_viewer.tick == 1
        snap = feed_viewer.snapshot()
        assert "vv.rgba" in snap["bus"] or snap["bus"].get("vv.rgba") is not None

    def test_feed_video_batch_multiple_frames(self, feed_viewer):
        w, h = 8, 8
        frames = [bytes([i % 256, 0, 0, 255] * (w * h)) for i in range(4)]
        snap = feed_viewer.feed_video_batch(frames, module_id="vv")
        assert feed_viewer.tick == 4
        assert snap["t"] == 4

    def test_feed_video_tick_sets_frame_key(self, feed_viewer):
        w, h = 4, 4
        rgba = bytes([0, 255, 0, 255] * (w * h))
        feed_viewer.feed_video_tick(rgba, module_id="vv")
        eng = feed_viewer.engine
        assert eng.bus.get("vv.frame") is not None

    def test_feed_video_empty_frame(self, feed_viewer):
        feed_viewer.feed_video_tick(b"", module_id="vv")
        assert feed_viewer.tick == 1


# ── REST endpoints ─────────────────────────────────────────────────────────────

class TestFeedVideoREST:
    @pytest.fixture
    def client(self):
        from atomic.ui.server import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    def test_feed_video_start(self, client):
        r = client.post("/api/feed_video/feed_video_live/start",
                        json={"width": 8, "height": 8})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["ok"] is True
        assert d["name"] == "feed_video_live"
        assert d["module_id"] == "vv"
        assert d["width"] == 8
        assert d["height"] == 8
        assert "prompts" in d

    def test_feed_video_stop(self, client):
        client.post("/api/feed_video/feed_video_live/start",
                    json={"width": 8, "height": 8})
        r = client.post("/api/feed_video/feed_video_live/stop")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["ok"] is True
        assert d["running"] is False

    def test_feed_video_push_frame_raw(self, client):
        client.post("/api/feed_video/feed_video_live/start",
                    json={"width": 8, "height": 8})
        rgba = bytes([255, 0, 0, 255] * 64)
        r = client.post("/api/feed_video/feed_video_live/push_frame",
                        content=rgba,
                        headers={"Content-Type": "application/octet-stream"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["ok"] is True
        assert d["bytes"] == len(rgba)

    def test_feed_video_push_frame_json_batch(self, client):
        client.post("/api/feed_video/feed_video_live/start",
                    json={"width": 8, "height": 8})
        rgba = bytes([0, 255, 0, 255] * 64)
        frames = [base64.b64encode(rgba).decode("ascii")]
        r = client.post("/api/feed_video/feed_video_live/push_frame",
                        json={"frames": frames})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["ok"] is True
        assert d["frames"] == 1

    def test_feed_video_batch(self, client):
        client.post("/api/feed_video/feed_video_live/start",
                    json={"width": 8, "height": 8})
        r = client.post("/api/feed_video/feed_video_live/batch",
                        json={"n_frames": 4})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["ok"] is True
        assert d["n_frames"] == 4
        assert len(d["bytes_per_frame"]) == 4

    def test_feed_video_status_running(self, client):
        client.post("/api/feed_video/feed_video_live/start",
                    json={"width": 8, "height": 8})
        r = client.get("/api/feed_video/feed_video_live/status")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["running"] is True
        assert d["frames_generated"] >= 0

    def test_feed_video_status_idle(self, client):
        r = client.get("/api/feed_video/feed_video_live/status")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["running"] is False


# ── feed_video_live program ────────────────────────────────────────────────────

class TestFeedVideoProgram:
    def test_feed_video_live_builds(self):
        prog = _build("feed_video_live")
        assert prog is not None

    def test_feed_video_live_one_viz_video_block(self):
        prog = _build("feed_video_live")
        blocks = prog.blocks
        assert len(blocks) == 1
        assert blocks[0].primitive == "viz_video"

    def test_feed_video_live_compiles(self):
        prog = _build("feed_video_live")
        patch = prog.compile("microfx")
        assert "modules" in patch
        assert len(patch["modules"]) == 1
        assert patch["modules"][0]["primitive"] == "viz_video"


# ── Swarm H4 consensus -> H3 routing (Aspect 3) ───────────────────────────────

class TestSwarmH3Consensus:
    def test_swarm_consensus_w_channel(self):
        d = Display(400, 400, 4, 4)
        swarm = Swarm(display=d)
        for i, val in enumerate([1.0, 2.0, 3.0, 4.0]):
            prog = Program(f"a{i}", blocks=[
                Block(f"c{i}", "const", {"value": val}),
                Block("g1", "gain", {"factor": 1.0}),
            ], wires=[Wire(f"c{i}.cv", "g1.in")])
            g = d.link(f"g{i}", i // 4, i % 4, 1, 1)
            swarm.add_agent(Agent(f"a{i}", prog, tile_group=g))
        # 2 ticks: gain sees input after 1-tick latency (const at t0, gain at t1)
        res = swarm.run(2)
        w = res.consensus(port="g1.cv")
        # All 4 agents contribute their const: 1+2+3+4 = 10; W = sum
        assert abs(w - 10.0) < 1e-9

    def test_h3session_tick_generates_frame(self):
        h3 = H3Stub(width=8, height=8)
        session = H3Session(h3, prompts=["test prompt"], frames_per_prompt=2)
        frame = session.tick()
        assert frame is not None
        assert len(frame.rgba) == 8 * 8 * 4

    def test_h3session_multiple_ticks(self):
        h3 = H3Stub(width=4, height=4)
        session = H3Session(h3, prompts=["p0", "p1"], frames_per_prompt=1)
        frames = []
        for _ in range(3):
            f = session.tick()
            if f:
                frames.append(f)
        assert len(frames) == 3
        prompts = [f.prompt for f in frames]
        assert len(set(prompts)) >= 1

    def test_swarm_consensus_deterministic_parallel(self):
        d = Display(400, 400, 4, 4)
        swarm = Swarm(display=d)
        for i, val in enumerate([1.0, 2.0, 3.0, 4.0]):
            prog = Program(f"a{i}", blocks=[
                Block(f"c{i}", "const", {"value": val}),
                Block("g1", "gain", {"factor": 1.0}),
            ], wires=[Wire(f"c{i}.cv", "g1.in")])
            g = d.link(f"g{i}", i // 4, i % 4, 1, 1)
            swarm.add_agent(Agent(f"a{i}", prog, tile_group=g))
        res_s = swarm.run(2, parallel=False)
        res_p = swarm.run(2, parallel=True)
        ws = res_s.consensus(port="g1.cv")
        wp = res_p.consensus(port="g1.cv")
        assert abs(ws - wp) < 1e-9


# ── QBF frame trace (Aspect 4) ────────────────────────────────────────────────

class TestQbfframeTrace:
    def test_video_frame_entry_dataclass(self):
        rgba = bytes([255, 0, 0, 255] * 64)
        entry = VideoFrameEntry(
            seq=1, t=5, rgba=rgba,
            prompt="test", seed=42,
            h3_latency_ms=10.0,
            width=8, height=8,
            w_gate=2.0, x_gate=1.0, y_gate=0.5, z_gate=-0.5,
        )
        assert entry.seq == 1
        assert entry.t == 5
        assert len(entry.rgba) == 8 * 8 * 4
        assert entry.w_gate == 2.0

    def test_flow_trace_record_video_frame(self):
        trace = FlowTrace(max_frames=100)
        rgba = bytes([0, 255, 0, 255] * 64)
        entry = trace.record_video_frame(
            rgba=rgba, t=3,
            prompt="comet", seed=7,
            h3_latency_ms=5.0,
            width=8, height=8,
            w_gate=1.5, x_gate=0.8, y_gate=0.2, z_gate=-0.1,
        )
        assert entry is not None
        assert entry.seq == 1
        assert entry.t == 3
        assert len(entry.rgba) == 8 * 8 * 4
        assert entry.prompt == "comet"
        assert entry.w_gate == 1.5
        assert trace._vseq == 1
        assert trace.video_seq == 1

    def test_flow_trace_video_ring_wrap(self):
        trace = FlowTrace(max_frames=5)
        rgba = bytes([0, 0, 255, 255] * 16)
        for i in range(10):
            trace.record_video_frame(rgba=rgba, t=i, seed=i)
        assert len(trace.video) == 5
        # Oldest entries evicted
        assert trace.video[0].t == 5

    def test_flow_trace_snapshot_video(self):
        trace = FlowTrace(max_frames=100)
        rgba = bytes([255, 255, 0, 255] * 16)
        trace.record_video_frame(rgba=rgba, t=0, prompt="test", seed=1,
                                 width=4, height=4, w_gate=1.0)
        snap = trace.snapshot()
        assert snap["n_video"] == 1
        assert snap["vseq"] == 1
        assert "video" in snap
        assert len(snap["video"]) == 1

    def test_flow_trace_from_snapshot_video(self):
        trace = FlowTrace(max_frames=100)
        rgba = bytes([128, 128, 128, 255] * 16)
        trace.record_video_frame(rgba=rgba, t=7, prompt="from_snap",
                                 seed=99, width=4, height=4)
        snap = trace.snapshot()
        rebuilt = FlowTrace.from_snapshot(snap)
        assert rebuilt.video_seq == 1
        assert len(rebuilt.video) == 1
        assert rebuilt.video[0].prompt == "from_snap"
        assert rebuilt.video[0].seed == 99

    def test_qbfstore_append_video_run(self, qbf_store):
        trace = FlowTrace(max_frames=100)
        # Record some video frames
        for i in range(4):
            rgba = bytes([i % 256, 0, 0, 255] * 64)
            trace.record_video_frame(
                rgba=rgba, t=i,
                prompt=f"frame {i}",
                seed=i * 10,
                width=8, height=8,
                h3_latency_ms=float(i),
                w_gate=float(i * 0.5),
            )
        # Also record some node frames for the regular trace
        trace.record_node("c0", "const", {}, {"cv": 1.0}, 100.0, t=0)
        m = qbf_store.append_run(trace, note="video test")
        assert m["n_video"] == 4
        assert m["vseq"] == 4

    def test_qbfstore_load_run_with_video(self, qbf_store):
        trace = FlowTrace(max_frames=100)
        for i in range(3):
            rgba = bytes([0, i % 256, 0, 255] * 32)
            trace.record_video_frame(rgba=rgba, t=i, prompt=f"p{i}",
                                     seed=i, width=8, height=8)
        trace.record_node("c0", "const", {}, {"cv": 1.0}, 100.0, t=0)
        qbf_store.append_run(trace)
        d = qbf_store.load_run(0)
        assert d["manifest"]["n_video"] == 3
        assert len(d["video"]) == 3
        assert d["video"][0]["prompt"] == "p0"
        assert d["video"][0]["seed"] == 0

    def test_qbfstore_flow_trace_with_video(self, qbf_store):
        trace = FlowTrace(max_frames=100)
        for i in range(2):
            rgba = bytes([i, i + 64, i + 128, 255] * 16)
            trace.record_video_frame(rgba=rgba, t=i, prompt=f"q{i}",
                                     seed=i * 5, width=4, height=4)
        trace.record_node("c0", "const", {}, {"cv": 1.0}, 50.0, t=0)
        qbf_store.append_run(trace)
        ft = qbf_store.flow_trace(0)
        assert ft.video_seq == 2
        assert len(ft.video) == 2
        assert ft.video[0].prompt == "q0"

    def test_qbfstore_replay_run_video_trace(self, qbf_store):
        trace = FlowTrace(max_frames=100)
        for i in range(2):
            rgba = bytes([i, 0, 255 - i, 255] * 16)
            trace.record_video_frame(rgba=rgba, t=i, prompt=f"r{i}",
                                     seed=i, width=4, height=4)
        trace.record_node("c0", "const", {}, {"cv": 1.0}, 50.0, t=0)
        qbf_store.append_run(trace)
        ft = qbf_store.flow_trace(0)
        assert ft.video_seq == 2
        assert ft.video[0].seed == 0
        assert ft.video[1].seed == 1
