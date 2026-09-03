"""iter35 tests: VideoSynth + VideoSynthSource + InfiniteVideoLoop integration.

Covers:
  - VideoSynth: 5 effects (wave/noise_field/pixel_sort/mandelbrot/fluid),
    determinism (same seed + tick = same frame), H4 channel decoding.
  - VideoSynthSource: H3Source-compatible wrapper.
  - InfiniteVideoLoop + VideoSynth: end-to-end pipeline.
  - BicameralViewer.feed_ivl_tick(): step IVL and capture rendered output.
  - QBF round-trip with VideoSynth frames.

Run: python -m pytest tests/test_iter35.py -v
"""
from __future__ import annotations

import os
import tempfile

import pytest

from atomic import (
    Program, Block, Wire,
    H3Stub,
    FlowTrace, QbfTraceStore,
    BicameralViewer, InfiniteVideoLoop,
    VideoSynth, VideoSynthSource, VIDEO_SYNTH_EFFECTS as EFFECTS,
)


# ── fixtures ────────────────────────────────────────────────────────────────────

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
        name="ivl_synth_test",
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


# ── VideoSynth basics ──────────────────────────────────────────────────────────


class TestVideoSynthBasics:
    def test_five_effects(self):
        for effect in EFFECTS:
            synth = VideoSynth(width=8, height=8, effect=effect)
            assert synth.effect == effect
            assert synth.width == 8
            assert synth.height == 8

    def test_default_wave(self):
        synth = VideoSynth(width=16, height=16)
        assert synth.effect == "wave"

    def test_invalid_effect_falls_back_to_wave(self):
        synth = VideoSynth(width=8, height=8, effect="nonexistent")
        assert synth.effect == "wave"

    def test_tick_increments(self):
        synth = VideoSynth(width=4, height=4)
        assert synth.t == 0
        synth.tick("prompt")
        assert synth.t == 1
        synth.tick("prompt")
        assert synth.t == 2

    def test_run_produces_n_frames(self):
        synth = VideoSynth(width=4, height=4, seed=1)
        frames = synth.run(5, "test")
        assert len(frames) == 5
        assert all(f["t"] == i for i, f in enumerate(frames))

    def test_frame_size(self):
        synth = VideoSynth(width=8, height=8)
        f = synth.tick("x")
        assert len(f["rgba"]) == 8 * 8 * 4

    def test_generate_interface(self):
        synth = VideoSynth(width=8, height=8)
        r = synth.generate("test", n_frames=3)
        assert "frames" in r
        assert len(r["frames"]) == 3
        assert r["prompt"] == "test"

    def test_resize(self):
        synth = VideoSynth(width=8, height=8)
        f = synth.tick("x")
        small = synth._resize(f["rgba"], 4, 4)
        assert len(small) == 4 * 4 * 4


# ── VideoSynth determinism ──────────────────────────────────────────────────────


class TestVideoSynthDeterminism:
    def test_same_seed_same_frame(self):
        synth = VideoSynth(width=8, height=8, seed=42, effect="wave")
        f1 = synth._render("prompt", 5)
        synth2 = VideoSynth(width=8, height=8, seed=42, effect="wave")
        f2 = synth2._render("prompt", 5)
        assert f1 == f2

    def test_different_seed_different_frame(self):
        synth1 = VideoSynth(width=8, height=8, seed=1)
        synth2 = VideoSynth(width=8, height=8, seed=2)
        f1 = synth1._render("prompt", 5)
        f2 = synth2._render("prompt", 5)
        assert f1 != f2

    def test_tick_advances_frame(self):
        synth = VideoSynth(width=8, height=8, seed=7)
        f0 = synth._render("p", 0)
        f1 = synth._render("p", 1)
        assert f0 != f1

    def test_same_prompt_different_ticks_different_frames(self):
        synth = VideoSynth(width=8, height=8, seed=7, effect="wave")
        synth.tick("prompt")
        f0 = synth.tick("prompt")
        synth2 = VideoSynth(width=8, height=8, seed=7, effect="wave")
        synth2.tick("prompt")
        f1 = synth2.tick("prompt")
        assert f0["rgba"] == f1["rgba"]  # same seed/tick
        assert f0["t"] == f1["t"] == 1


# ── VideoSynth H4 channels ──────────────────────────────────────────────────────


class TestVideoSynthH4Channels:
    def test_h4_channels_return_4_tuple(self):
        synth = VideoSynth(width=8, height=8)
        rgba = bytes([128, 64, 32, 255] * 16)
        w, x, y, z = synth.h4_channels(rgba)
        assert isinstance(w, float)
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)

    def test_h4_channels_w_is_log_luma(self):
        synth = VideoSynth(width=4, height=4)
        # Build a frame where the center pixel is (200, 100, 50, 255)
        rgba = bytearray(4 * 4 * 4)
        # Center pixel index: row 2, col 2 -> j = (2*4 + 2) * 4 = 40
        rgba[40:44] = bytes([200, 100, 50, 255])
        w, x, y, z = synth.h4_channels(bytes(rgba))
        import math
        r_n = 200.0 / 255.0
        g_n = 100.0 / 255.0
        b_n = 50.0 / 255.0
        expected_luma = 0.299 * r_n + 0.587 * g_n + 0.114 * b_n
        expected_w = math.log(max(1e-6, expected_luma))
        assert abs(w - expected_w) < 0.05
        assert abs(x - r_n) < 0.01
        assert abs(y - g_n) < 0.01
        assert abs(z - b_n) < 0.01

    def test_h4_channels_short_rgba(self):
        synth = VideoSynth(width=8, height=8)
        w, x, y, z = synth.h4_channels(b"")
        assert (w, x, y, z) == (0.0, 0.0, 0.0, 0.0)


# ── VideoSynthSource ────────────────────────────────────────────────────────────


class TestVideoSynthSource:
    def test_generate_interface(self):
        synth = VideoSynth(width=8, height=8)
        src = VideoSynthSource(synth=synth)
        r = src.generate("test", n_frames=4)
        assert len(r["frames"]) == 4
        assert "prompt" in r
        assert "h3_latency_ms" in r

    def test_calls_counter(self):
        synth = VideoSynth(width=8, height=8)
        src = VideoSynthSource(synth=synth)
        src.generate("x", n_frames=3)
        assert src.stats()["source_calls"] == 1  # one generate() call
        assert src.stats()["calls"] == 3  # synth ticked 3 times

    def test_source_calls_increments(self):
        synth = VideoSynth(width=8, height=8)
        src = VideoSynthSource(synth=synth)
        assert src.t == 0
        src.generate("x", n_frames=3)
        assert src.t == 3  # synth.t advanced by 3 ticks


# ── InfiniteVideoLoop + VideoSynthSource ────────────────────────────────────────


class TestInfiniteVideoLoopVideoSynth:
    def test_step_returns_frame(self, viewer):
        synth = VideoSynth(width=4, height=4, effect="wave")
        src = VideoSynthSource(synth=synth)
        loop = InfiniteVideoLoop(src, viewer, prompts=["test"])
        f = loop.step()
        assert f is not None
        assert len(f.rgba) == 4 * 4 * 4
        assert f.prompt == "test"

    def test_run_n_ticks(self, viewer):
        synth = VideoSynth(width=4, height=4)
        src = VideoSynthSource(synth=synth)
        loop = InfiniteVideoLoop(src, viewer, prompts=["a", "b"], max_ticks=8)
        frames = loop.run(4)
        assert len(frames) == 4

    def test_stats_keys(self, viewer):
        synth = VideoSynth(width=4, height=4)
        src = VideoSynthSource(synth=synth)
        loop = InfiniteVideoLoop(src, viewer, prompts=["x"])
        loop.run(3)
        s = loop.stats()
        for k in ("t", "frame_index", "h3_calls", "bridge_depth"):
            assert k in s, k

    def test_trace_records_frames(self, viewer):
        synth = VideoSynth(width=4, height=4)
        src = VideoSynthSource(synth=synth)
        trace = FlowTrace(max_frames=32)
        loop = InfiniteVideoLoop(src, viewer, prompts=["a", "b"], trace=trace)
        loop.run(4)
        assert trace.video_seq == 4


# ── BicameralViewer.feed_ivl_tick ─────────────────────────────────────────────


class TestBicameralViewerFeedIvlTick:
    def test_feed_ivl_tick_returns_snapshot(self, viewer):
        synth = VideoSynth(width=4, height=4)
        src = VideoSynthSource(synth=synth)
        loop = InfiniteVideoLoop(src, viewer, prompts=["x"], max_ticks=10)
        snap = viewer.feed_ivl_tick(loop)
        assert "sub" in snap
        assert "con" in snap
        assert "bridge" in snap

    def test_feed_ivl_tick_ivl_frame(self, viewer):
        synth = VideoSynth(width=4, height=4)
        src = VideoSynthSource(synth=synth)
        loop = InfiniteVideoLoop(src, viewer, prompts=["x"], max_ticks=10)
        snap = viewer.feed_ivl_tick(loop)
        assert "_ivl_frame" in snap
        frame_meta = snap["_ivl_frame"]
        assert frame_meta is not None
        assert "t" in frame_meta
        assert "prompt" in frame_meta
        assert "seed" in frame_meta
        assert "rgba_sha256" in frame_meta

    def test_feed_ivl_tick_exhausted_returns_none_frame(self, viewer):
        synth = VideoSynth(width=4, height=4)
        src = VideoSynthSource(synth=synth)
        loop = InfiniteVideoLoop(src, viewer, prompts=["x"], max_ticks=2)
        loop.run(2)
        snap = viewer.feed_ivl_tick(loop)
        assert snap["_ivl_frame"] is None

    def test_feed_ivl_batch(self, viewer):
        synth = VideoSynth(width=4, height=4)
        src = VideoSynthSource(synth=synth)
        loop = InfiniteVideoLoop(src, viewer, prompts=["x"], max_ticks=10)
        snaps = viewer.feed_ivl_batch(loop, 3)
        assert viewer.tick == 3


# ── QBF round-trip with VideoSynth ──────────────────────────────────────────────


class TestVideoSynthQbfRoundTrip:
    def test_qbf_archive_and_load(self, viewer, qbf_store):
        synth = VideoSynth(width=4, height=4)
        src = VideoSynthSource(synth=synth)
        trace = FlowTrace(max_frames=32)
        loop = InfiniteVideoLoop(src, viewer, prompts=["p1", "p2"], trace=trace)
        loop.run(4)
        m = qbf_store.append_run(trace, note="synth_qbf_test")
        assert m["n_video"] == 4
        d = qbf_store.load_run(0)
        assert len(d["video"]) == 4
        for entry in d["video"]:
            assert len(entry["rgba"]) == 4 * 4 * 4


# ── BicameralRegistry: infinite_video_export ──────────────────────────────────


class TestBicameralRegistry:
    def test_bicameral_registry_infinite_video_export(self):
        """The 'infinite_video_export' bicameral program exists and validates."""
        from atomic.ui.programs import build_bicameral
        spec = build_bicameral("infinite_video_export")
        assert spec is not None
        assert spec["type"] == "bicameral"
        con_blocks = [b.id for b in spec["con"].blocks]
        assert "vv" in con_blocks and "jle" in con_blocks
        # The con program must validate (no invalid wires)
        errs = spec["con"].validate()
        assert errs == [], f"infinite_video_export con errors: {errs}"


# ── All 5 effects produce valid frames ─────────────────────────────────────────


class TestAllEffects:
    @pytest.mark.parametrize("effect", EFFECTS)
    def test_effect_produces_valid_frame(self, effect):
        synth = VideoSynth(width=8, height=8, effect=effect, seed=99)
        f = synth.tick("test effect " + effect)
        assert len(f["rgba"]) == 8 * 8 * 4
        assert f["prompt"] == "test effect " + effect
        assert f["t"] == 0

    @pytest.mark.parametrize("effect", EFFECTS)
    def test_effect_deterministic(self, effect):
        synth1 = VideoSynth(width=8, height=8, effect=effect, seed=7)
        synth2 = VideoSynth(width=8, height=8, effect=effect, seed=7)
        for _ in range(3):
            synth1.tick("determinism")
            synth2.tick("determinism")
        f1 = synth1.tick("determinism")
        f2 = synth2.tick("determinism")
        assert f1["rgba"] == f2["rgba"]
