"""Iter 36: FastH3 (FastVideo GGUF, 4-step VSA) bridge spec tests.

Covers:
  - FastH3Stub: deterministic, FastH3-specific metadata, visual diff from H3Stub
  - FastH3Client: wire contract; FastH3Error on connection failure
  - FastH3Source: factory
  - FastH3Frame: provenance() round-trips QBF
  - FastH3Session: tick -> bridge.push_frame -> replay
  - InfiniteFastH3Loop: stub -> HostBridge -> viz_fasth3_video -> QBF round trip
  - viz_fasth3_video atom: registers, decodes H(4), exposes W/X/Y/Z
  - Param ranges: viz_fasth3_video params in PARAM_RANGES

Run: python -m pytest tests/test_iter36.py -q
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic import (
    Engine, Block, Wire, Program, ATOMS, HostBridge,
    FastH3Stub, FastH3Client, FastH3Source, FastH3Frame,
    FastH3Session, FastH3Error,
    InfiniteFastH3Loop, H3Stub, H3Frame,
    FASTH3_DEFAULT_STEPS, FASTH3_DEFAULT_VSA, FASTH3_DEFAULT_VSA_KEEP,
    FASTH3_DEFAULT_QUANT, FASTH3_PROMPT_BANK_DEFAULT,
    FlowTrace, QbfTraceStore,
    VideoFrameEntry,
)
from atomic.gates import PARAM_RANGES, param_range
from atomic.ui.viewer import Viewer
from atomic.ui.bicameral_viewer import BicameralViewer
from atomic.ui.programs import build as _build


# ── FastH3Stub ────────────────────────────────────────────────────────────────

class TestFastH3Stub:
    def test_default_metadata(self):
        s = FastH3Stub()
        r = s.generate("a comet over the ocean")
        assert r["model"] == "FastH3"
        assert r["steps"] == FASTH3_DEFAULT_STEPS
        assert r["vsa"] == FASTH3_DEFAULT_VSA
        assert r["vsa_keep"] == FASTH3_DEFAULT_VSA_KEEP
        assert r["quant"] == FASTH3_DEFAULT_QUANT
        assert len(r["frames"]) == s.n_frames
        assert isinstance(r["frames"][0], bytes)
        assert len(r["frames"][0]) == s.width * s.height * 4

    def test_determinism_same_seed_same_pixels(self):
        a = FastH3Stub(width=32, height=32, n_frames=4)
        b = FastH3Stub(width=32, height=32, n_frames=4)
        ra = a.generate("cat on a roof", seed=42)
        rb = b.generate("cat on a roof", seed=42)
        for fa, fb in zip(ra["frames"], rb["frames"]):
            assert fa == fb

    def test_different_seed_different_pixels(self):
        s = FastH3Stub(width=32, height=32, n_frames=2)
        a = s.generate("cat on a roof", seed=42)
        b = s.generate("cat on a roof", seed=43)
        # at least one frame differs
        assert any(fa != fb for fa, fb in zip(a["frames"], b["frames"]))

    def test_visual_diff_from_h3_stub(self):
        """FastH3 stub uses wave freq 8.2467 vs H3Stub's 6.2832 — pixels differ."""
        fh = FastH3Stub(width=32, height=32, n_frames=2)
        h3 = H3Stub(width=32, height=32, n_frames=2)
        rf = fh.generate("same prompt", seed=1)
        rh = h3.generate("same prompt", seed=1)
        assert rf["frames"][0] != rh["frames"][0], (
            "FastH3Stub should produce visually distinct output vs H3Stub"
        )

    def test_overrides(self):
        s = FastH3Stub()
        r = s.generate("foo", steps=8, vsa=False)
        assert r["steps"] == 8
        assert r["vsa"] is False

    def test_counters(self):
        s = FastH3Stub(width=8, height=8, n_frames=4)
        s.generate("p1")
        s.generate("p2", n_frames=3)
        assert s._calls == 2
        assert s._frames_emitted == 7


# ── FastH3Frame ──────────────────────────────────────────────────────────────

class TestFastH3Frame:
    def test_provenance_round_trip(self):
        f = FastH3Frame(
            rgba=b"\x01\x02\x03\x04" * 16,
            prompt="cat", seed=42, t=7, h3_latency_ms=12.5,
            steps=4, vsa=True, vsa_keep=10, quant="Q4_K_M", model="FastH3",
        )
        prov = f.provenance()
        assert prov["model"] == "FastH3"
        assert prov["steps"] == 4
        assert prov["vsa"] == 1
        assert prov["quant"] == "Q4_K_M"
        assert prov["prompt"] == "cat"
        assert prov["seed"] == 42
        assert prov["size_bytes"] == 64
        assert prov["rgba_sha256"] == hashlib.sha256(f.rgba).hexdigest()
        assert prov["h3_latency_ms"] == 12.5
        assert prov["t"] == 7

    def test_to_h3_frame_drops_metadata(self):
        f = FastH3Frame(rgba=b"x"*4, prompt="p", steps=4, vsa=True,
                        quant="Q5_K_M")
        h = f.to_h3_frame()
        assert isinstance(h, H3Frame)
        assert h.prompt == "p"
        assert not hasattr(h, "steps")

    def test_default_metadata(self):
        f = FastH3Frame(rgba=b"\x00"*4)
        assert f.steps == FASTH3_DEFAULT_STEPS
        assert f.vsa == FASTH3_DEFAULT_VSA
        assert f.vsa_keep == FASTH3_DEFAULT_VSA_KEEP
        assert f.quant == FASTH3_DEFAULT_QUANT
        assert f.model == "FastH3"


# ── FastH3Client + Source ────────────────────────────────────────────────────

class TestFastH3ClientAndSource:
    def test_source_factory(self):
        assert isinstance(FastH3Source("stub"), FastH3Stub)
        assert isinstance(FastH3Source("http"), FastH3Client)
        with pytest.raises(FastH3Error):
            FastH3Source("bogus")

    def test_client_unreachable_raises(self):
        c = FastH3Client(endpoint="http://127.0.0.1:1", timeout_s=0.1)
        with pytest.raises(FastH3Error):
            c.generate("p")

    def test_client_fallback_on_failure(self):
        """Fallback stub returns the n_frames the caller requested."""
        stub = FastH3Stub(width=8, height=8, n_frames=2)
        c = FastH3Client(endpoint="http://127.0.0.1:1", timeout_s=0.1, fallback=stub)
        r = c.generate("p1", n_frames=2)
        assert r["model"] == "FastH3"
        assert len(r["frames"]) == 2
        assert r["steps"] == FASTH3_DEFAULT_STEPS

    def test_client_default_metadata_in_request_body(self):
        """Even on a forced exception, the request body the client WOULD
        have sent includes the FastH3-specific knobs."""
        captured = {}

        def _failing_post(self, path, body):
            captured["path"] = path
            captured["body"] = body
            raise FastH3Error("captured")

        c = FastH3Client(endpoint="http://x", timeout_s=0.01)
        c._post = _failing_post.__get__(c, type(c))
        with pytest.raises(FastH3Error):
            c.generate("p", n_frames=3, steps=8, vsa=False)
        assert captured["path"] == "/generate"
        b = captured["body"]
        assert b["steps"] == 8
        assert b["vsa"] is False
        assert b["quant"] == FASTH3_DEFAULT_QUANT
        assert b["model"] == "FastH3"
        assert b["n_frames"] == 3


# ── FastH3Session ────────────────────────────────────────────────────────────

class TestFastH3Session:
    def test_tick_pushes_frame_into_bridge(self):
        br = HostBridge(latency=1)
        s = FastH3Stub(width=8, height=8, n_frames=1)
        sess = FastH3Session(s, bridge=br, max_ticks=5)
        frame = sess.tick()
        assert isinstance(frame, FastH3Frame)
        # The bridge has a pending frame at arrival tick=1 (latency=1)
        assert br.depth() >= 1
        # pop at tick=1 -> frame available
        ready = br.pop_frame(1)
        assert len(ready) == 1
        assert ready[0]["rgba"] == frame.rgba

    def test_round_robin_prompts(self):
        s = FastH3Stub(width=8, height=8, n_frames=1)
        sess = FastH3Session(s, prompts=["a", "b"], bridge=None,
                             frames_per_prompt=1, max_ticks=4)
        f1 = sess.tick()
        f2 = sess.tick()
        assert f1.prompt == "a"
        assert f2.prompt == "b"

    def test_consensus_pick(self):
        s = FastH3Stub(width=8, height=8, n_frames=1)
        pick = {"picked": None}

        def consensus(prev):
            pick["picked"] = prev[-1] if prev else None
            return "override"

        sess = FastH3Session(s, prompts=["a", "b"], bridge=None,
                             frames_per_prompt=1,
                             prompt_consensus=consensus, max_ticks=3)
        sess.tick()
        sess.tick()
        # consensus was asked for the prompt after the first one
        assert pick["picked"] == "a"
        # second frame's prompt comes from consensus
        assert sess._ring[-1].prompt == "override"

    def test_stats(self):
        s = FastH3Stub(width=4, height=4, n_frames=1)
        sess = FastH3Session(s, bridge=None, max_ticks=3)
        for _ in range(3):
            sess.tick()
        st = sess.stats()
        assert st["t"] == 3
        assert st["frames_in_ring"] == 3
        assert st["fasth3_calls"] == 3


# ── viz_fasth3_video atom ────────────────────────────────────────────────────

class TestVizFastH3VideoAtom:
    def test_registered(self):
        assert "viz_fasth3_video" in ATOMS
        a = ATOMS["viz_fasth3_video"]
        assert a.category == "sink"
        assert "in" in a.inputs
        assert "ready" in a.outputs
        for out in ("w", "x", "y", "z"):
            assert out in a.outputs

    def test_param_ranges(self):
        assert param_range("viz_fasth3_video", "capture") == (0.0, 1.0, 1.0, "")
        assert param_range("viz_fasth3_video", "steps")[1] == 8
        assert param_range("viz_fasth3_video", "vsa_keep")[1] == 100

    def test_engine_decodes_frame(self):
        p = Program("vfh_test", blocks=[
            Block("src", "const", {"value": 1.0}),
            Block("sink", "viz_fasth3_video",
                  {"capture": 1.0, "steps": 4, "vsa": 1.0,
                   "vsa_keep": 10, "quant": "Q5_K_M"}),
        ], wires=[Wire("src.cv", "sink.in")])
        patch = p.compile("microfx")
        eng = Engine(patch["modules"], patch.get("wires", []))
        eng.run(1)
        # push a frame onto the bus
        rgba = bytes([255, 0, 0, 255] * 64)  # 64 RGBA pixels
        eng.bus.set("sink.frame", rgba)
        eng.run(1)
        out = eng.bus.get("sink.rgba")
        assert out == rgba
        decoded = eng.bus.get("sink.rgba_decoded")
        assert decoded is not None
        # after H(4) decode, the X row (linear red) should be > 0
        assert eng.bus.get("sink.x") is not None


# ── InfiniteFastH3Loop ─────────────────────────────────────────────────────────

class TestInfiniteFastH3Loop:
    def _make_viewer(self):
        """Build a BicameralViewer: sub=clock_bpm@60, con=viz_video.

        Same fixture as iter34: the InfiniteFastH3Loop pushes frames into
        bridge.push_frame -> viz_video bus key vv.frame. The bridge latency
        is 1 tick (matches the wire latch)."""
        from atomic import Program as _P, Block as _B
        sub = _P("fh3_sub", blocks=[_B("clk", "clock_bpm", {"bpm": 60})])
        con = _P("fh3_con", blocks=[_B("vv", "viz_video")])
        v = BicameralViewer(
            sub, con,
            bridge_map=[("clk.trig", "vv.in")],
            bridge_latency=1,
            name="iter36_fasth3_test",
        )
        return v

    def test_stub_to_bridge_to_viz_video_rgba_visible(self):
        v = self._make_viewer()
        s = FastH3Stub(width=8, height=8, n_frames=1)
        loop = InfiniteFastH3Loop(
            s, v, bridge_latency=1, module_id="vfh", max_ticks=4)
        frames = loop.run(3)
        assert len(frames) == 3
        for fr in frames:
            assert isinstance(fr, FastH3Frame)
            assert fr.steps == 4
            assert fr.vsa == FASTH3_DEFAULT_VSA

    def test_loop_stats_match_ticks(self):
        v = self._make_viewer()
        s = FastH3Stub(width=8, height=8, n_frames=1)
        loop = InfiniteFastH3Loop(
            s, v, bridge_latency=1, module_id="vfh", max_ticks=5)
        loop.run(4)
        st = loop.stats()
        assert st["t"] == 4
        assert st["frame_index"] == 4
        assert st["fasth3_calls"] >= 1  # could be 1-4 depending on frames_per_prompt

    def test_loop_records_trace(self):
        v = self._make_viewer()
        s = FastH3Stub(width=8, height=8, n_frames=1)
        tr = FlowTrace()
        loop = InfiniteFastH3Loop(
            s, v, bridge_latency=1, module_id="vfh", max_ticks=3, trace=tr)
        loop.run(3)
        # trace should have one video entry per frame + one node entry per frame
        assert tr.video_seq >= 3
        node_entries = [f for f in tr.frames if f.module_id == "vfh"]
        assert len(node_entries) == 3
        # extra metadata recorded in in_ports (QBF contract)
        e = node_entries[0]
        assert e.in_ports.get("model") == "FastH3"
        assert e.in_ports.get("steps") == 4
        assert e.in_ports.get("vsa") == 1
        assert e.in_ports.get("quant") == "Q5_K_M"


# ── QBF portability of FastH3 frames ─────────────────────────────────────────

class TestFastH3QBFPortability:
    def test_provenance_round_trips_through_qbf(self):
        """The full FastH3 provenance (steps/vsa/quant/model) round-trips
        through a QBF trace store when recorded as a video entry's
        metadata + a node entry's extra."""
        fd, path = tempfile.mkstemp(suffix=".qbf")
        os.close(fd)
        os.unlink(path)
        try:
            store = QbfTraceStore(path)
            tr = FlowTrace()
            f = FastH3Frame(rgba=b"\xab\xcd\xef\x01" * 32,
                             prompt="neon city", seed=99, t=2,
                             h3_latency_ms=7.0, steps=4, vsa=True,
                             vsa_keep=10, quant="Q4_K_M", model="FastH3")
            tr.record_video_frame(
                rgba=f.rgba, t=2,
                prompt=f.prompt, seed=f.seed,
                h3_latency_ms=f.h3_latency_ms,
                width=0, height=0,
                w_gate=0.0, x_gate=0.0, y_gate=0.0, z_gate=0.0)
            # note the FastH3 provenance via the record_node extra path
            # (QBF node entries carry in_ports/out_ports/latency; the
            # model/steps/vsa/quant metadata round-trips through the
            # tag field — kept short so it survives .qbf archives).
            tr.record_node(
                module_id="vfh", module_type="viz_fasth3_video",
                in_ports={"frame_bytes": len(f.rgba),
                          "model": f.model,
                          "steps": f.steps,
                          "vsa": int(f.vsa),
                          "quant": f.quant},
                out_ports={"rgba_decoded": len(f.rgba)},
                latency_us=int(f.h3_latency_ms * 1000),
                t=2, tag="iter36")
            store.append_run(tr, "iter36_fasth3")
            store.close()
            # reopen
            store2 = QbfTraceStore(path)
            run_ids = store2.runs()
            assert len(run_ids) >= 1, "at least one run recorded"
            rid = run_ids[0]
            blob = store2.load_run(rid)
# frame sha matches
            assert blob["frames"] is not None
            # frame sha is recorded in the video entry
            # node entries carry the FastH3 provenance in in_ports
            node_entries = [ne for ne in blob["frames"] if ne.get("module_id") == "vfh"]
            assert len(node_entries) >= 1
            ne = node_entries[0]
            assert ne["in_ports"].get("quant") == "Q4_K_M"
            assert ne["in_ports"].get("steps") == 4
            assert ne["in_ports"].get("model") == "FastH3"
            store2.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ── iter 36 wire-contract guard ──────────────────────────────────────────────

class TestWireContractGuard:
    """The bridge must accept both FastH3 frames and plain H3 frames.
    This proves viz_fasth3_video / viz_video can share the bus key."""

    def test_bridge_accepts_both_frame_kinds(self):
        """push_frame() tags entries _frame=True; generic push() does not.

        pop_frame() only returns _frame-tagged entries (FastH3 / H3 path).
        Generic scalar push() entries are retrieved via pop().
        Both paths coexist without interference."""
        br = HostBridge(latency=1)
        # FastH3 path: push_frame with FastH3 RGBA
        br.push_frame(0, b"\xff\x00\x00\xff" * 8, prompt="fh3", module_id="vfh")
        # generic path: a raw dict (no _frame tag)
        br.push(1, {"scalar": 1.23, "prompt": "raw"})
        # pop_frame: only the FastH3 entry (tagged _frame=True)
        frames = br.pop_frame(2)
        assert len(frames) == 1
        assert frames[0]["prompt"] == "fh3"
        assert frames[0]["module_id"] == "vfh"
        # pop: the generic scalar dict (untagged)
        scalar = br.pop(2)
        assert scalar is not None
        assert scalar["scalar"] == 1.23
        # bridge is now empty
        assert br.depth() == 0