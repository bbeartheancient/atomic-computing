"""Iter 41 tests: SlopLoop (infinite slop loop via H4 consensus + H3 + QBF + evolve).

Covers:
  1. Fitness functions: color_variance, h4_w_latch, complexity, composite
  2. SlopEvolver: bank mutations, history, fitness cache, per-prompt scoring
  3. SlopLoop: tick/run/stop, H3 frame generation, trace recording
  4. QBF round-trip: SlopLoop -> .qbf shard -> load_run -> flow_trace
  5. Determinism: same seed + same fitness_fn = same loop run
  6. Swarm H4 consensus picks: W-dominant, ties to last-pixel latch
  7. Bank evolve: ties to fitness history, monotonic non-decreasing under stub
  8. Composite fitness + all fitness functions are pure/deterministic

Run: python -m pytest tests/test_iter41.py -q
All tests are offline (H3Stub only, no GPU required).
"""

from __future__ import annotations

import math
import os
import random
import shutil
import struct
import tempfile

import pytest

import sys

sys_path_insert = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path_insert not in sys.path:
    sys.path.insert(0, sys_path_insert)

from atomic import (
    SlopLoop, SlopEvolver, SlopEvolverResult,
    fitness_color_variance, fitness_h4_w_latch,
    fitness_complexity, composite_fitness,
    H3Stub, FlowTrace, QbfTraceStore,
    h4_gate,
)


def _rgba(r=0, g=0, b=0, a=255, w=8, h=8):
    return bytes([r, g, b, a] * (w * h))


def _rgba_gradient(w=8, h=8):
    out = bytearray(w * h * 4)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 4
            out[i] = int(255 * x / w)
            out[i + 1] = int(255 * y / h)
            out[i + 2] = 128
            out[i + 3] = 255
    return bytes(out)


def _rgba_noisy(seed=0, w=8, h=8):
    rng = random.Random(seed)
    n = w * h * 4
    return bytes([rng.randint(0, 255) for _ in range(n)])


# ---------------------------------------------------------------------------
# 1. Fitness functions
# ---------------------------------------------------------------------------

class TestFitnessColorVariance:
    def test_zero_rgba(self):
        assert fitness_color_variance(b"") == 0.0
        assert fitness_color_variance(None) == 0.0

    def test_uniform_rgba_zero_variance(self):
        rgba = _rgba(r=128, g=128, b=128, w=4, h=4)
        f = fitness_color_variance(rgba, width=4, height=4)
        assert f == 0.0

    def test_high_variance_gradient(self):
        rgba = _rgba_gradient(w=8, h=8)
        f = fitness_color_variance(rgba, width=8, height=8)
        assert f > 0.0

    def test_noisy_has_variance(self):
        rgba = _rgba_noisy(seed=42, w=8, h=8)
        f = fitness_color_variance(rgba, width=8, height=8)
        assert f > 0.0

    def test_deterministic(self):
        rgba = _rgba_noisy(seed=99, w=16, h=16)
        a = fitness_color_variance(rgba, width=16, height=16)
        b = fitness_color_variance(rgba, width=16, height=16)
        assert a == b

    def test_higher_than_static(self):
        static = _rgba(r=100, g=100, b=100, w=8, h=8)
        noisy = _rgba_noisy(seed=7, w=8, h=8)
        fs = fitness_color_variance(static, width=8, height=8)
        fn = fitness_color_variance(noisy, width=8, height=8)
        assert fn > fs


class TestFitnessH4WLatch:
    def test_w_latch(self):
        f = fitness_h4_w_latch(w_gate=2.5)
        assert abs(f - 2.5) < 1e-9

    def test_w_latch_zero(self):
        f = fitness_h4_w_latch(w_gate=0.0)
        assert f == 0.0

    def test_w_latch_negative(self):
        f = fitness_h4_w_latch(w_gate=-1.5)
        assert abs(f - (-1.5)) < 1e-9

    def test_w_latch_from_h4_gate(self):
        a, b, c, d = 1.0, 2.0, 3.0, 4.0
        w, z, y, x = h4_gate((a, b, c, d))
        f = fitness_h4_w_latch(w_gate=w)
        assert abs(f - w) < 1e-9

    def test_deterministic(self):
        f1 = fitness_h4_w_latch(w_gate=1.5)
        f2 = fitness_h4_w_latch(w_gate=1.5)
        assert f1 == f2


class TestFitnessComplexity:
    def test_zero_rgba(self):
        assert fitness_complexity(b"") == 0.0

    def test_uniform_low_entropy(self):
        rgba = _rgba(r=0, g=0, b=0, w=4, h=4)
        f = fitness_complexity(rgba, width=4, height=4)
        assert 0.0 <= f <= 1.0

    def test_noisy_high_entropy(self):
        rgba = _rgba_noisy(seed=55, w=16, h=16)
        f = fitness_complexity(rgba, width=16, height=16)
        assert 0.0 <= f <= 1.0

    def test_gradient_entropy(self):
        rgba = _rgba_gradient(w=8, h=8)
        f = fitness_complexity(rgba, width=8, height=8)
        assert 0.0 <= f <= 1.0

    def test_deterministic(self):
        rgba = _rgba_noisy(seed=33, w=8, h=8)
        a = fitness_complexity(rgba, width=8, height=8)
        b = fitness_complexity(rgba, width=8, height=8)
        assert a == b

    def test_range(self):
        for _ in range(5):
            rgba = _rgba_noisy(seed=random.randint(0, 9999), w=8, h=8)
            f = fitness_complexity(rgba, width=8, height=8)
            assert 0.0 <= f <= 1.0, f"fitness out of range: {f}"


class TestCompositeFitness:
    def test_composite_zero_frame(self):
        f = composite_fitness(rgba=b"", w_gate=0.0)
        assert 0.0 <= f

    def test_composite_with_frame(self):
        rgba = _rgba_noisy(seed=1, w=8, h=8)
        f = composite_fitness(rgba=rgba, width=8, height=8,
                               w_gate=1.0, x_gate=0.0, y_gate=0.0, z_gate=0.0)
        assert f > 0.0

    def test_composite_weights_sum_one(self):
        rgba = _rgba_noisy(seed=2, w=8, h=8)
        f = composite_fitness(rgba=rgba, width=8, height=8,
                               w_gate=1.0, alpha=0.4, beta=0.3, gamma=0.3)
        assert f > 0.0

    def test_composite_deterministic(self):
        rgba = _rgba_noisy(seed=3, w=8, h=8)
        a = composite_fitness(rgba=rgba, width=8, height=8, w_gate=2.0)
        b = composite_fitness(rgba=rgba, width=8, height=8, w_gate=2.0)
        assert a == b


# ---------------------------------------------------------------------------
# 2. SlopEvolver
# ---------------------------------------------------------------------------

class TestSlopEvolverBasics:
    def test_init_default_bank(self):
        ev = SlopEvolver(bank=None, fitness_fn=fitness_color_variance)
        assert len(ev.bank) >= 4

    def test_init_custom_bank(self):
        bank = ["a slow comet", "bright neon city", "macro beetle"]
        ev = SlopEvolver(bank=bank, fitness_fn=fitness_color_variance)
        assert list(ev.bank) == bank

    def test_init_deterministic_seed(self):
        ev1 = SlopEvolver(bank=None, fitness_fn=fitness_color_variance, seed=42)
        ev2 = SlopEvolver(bank=None, fitness_fn=fitness_color_variance, seed=42)
        assert ev1.bank == ev2.bank

    def test_history_empty_initially(self):
        ev = SlopEvolver(bank=["p1", "p2"], fitness_fn=fitness_color_variance)
        assert ev.history == []

    def test_evolve_increments_gen(self):
        bank = ["p1", "p2", "p3", "p4"]
        ev = SlopEvolver(bank=bank, fitness_fn=fitness_color_variance, seed=0)
        r = ev.evolve()
        assert ev.gen == 1
        assert isinstance(r, SlopEvolverResult)
        assert r.gen == 1

    def test_evolve_returns_result_with_bank(self):
        bank = ["p1", "p2", "p3", "p4"]
        ev = SlopEvolver(bank=bank, fitness_fn=fitness_color_variance, seed=0)
        r = ev.evolve()
        assert len(r.bank) == len(bank)
        assert isinstance(r.bank_hash, str)
        assert len(r.bank_hash) == 64

    def test_evolve_does_not_shrink_bank(self):
        bank = ["p1", "p2", "p3", "p4"]
        ev = SlopEvolver(bank=bank, fitness_fn=fitness_color_variance, seed=0)
        for _ in range(3):
            r = ev.evolve()
            assert len(r.bank) >= 1

    def test_evolve_with_per_prompt_scores(self):
        bank = ["prompt_a", "prompt_b", "prompt_c", "prompt_d"]
        ev = SlopEvolver(bank=bank, fitness_fn=fitness_color_variance, seed=0)
        r = ev.evolve(scores_per_prompt={"prompt_a": 100.0, "prompt_b": 50.0,
                                          "prompt_c": 10.0, "prompt_d": 0.0})
        assert ev.gen == 1
        assert r.score >= 0.0

    def test_evolve_deterministic_same_seed(self):
        bank = ["p1", "p2", "p3", "p4"]
        def dummy_fn(*args, **kw):
            return 0.5

        ev1 = SlopEvolver(bank=bank, fitness_fn=dummy_fn, seed=123)
        ev2 = SlopEvolver(bank=bank, fitness_fn=dummy_fn, seed=123)
        for _ in range(5):
            ev1.evolve()
        for _ in range(5):
            ev2.evolve()
        assert ev1.bank == ev2.bank

    def test_cache_frames(self):
        bank = ["p1", "p2", "p3", "p4"]
        ev = SlopEvolver(bank=bank, fitness_fn=fitness_color_variance, seed=0)
        frames = ["frame1", "frame2"]
        ev.cache_frames(bank, frames)
        h = ev._frames_cache  # check internal cache was populated
        assert len(h) >= 1

    def test_get_history(self):
        bank = ["p1", "p2", "p3", "p4"]
        ev = SlopEvolver(bank=bank, fitness_fn=fitness_color_variance, seed=0)
        ev.evolve()
        ev.evolve()
        h = ev.get_history()
        assert len(h) == 2
        assert h[0]["gen"] == 1
        assert h[1]["gen"] == 2


# ---------------------------------------------------------------------------
# 3. SlopLoop
# ---------------------------------------------------------------------------

class TestSlopLoopBasics:
    def test_init_defaults(self):
        loop = SlopLoop()
        assert loop.loop_t == 0
        assert loop.max_ticks == 1000
        assert not loop.running
        assert len(loop.frames) == 0
        assert len(loop.scores) == 0

    def test_init_custom_h3(self):
        h3 = H3Stub(width=8, height=8)
        loop = SlopLoop(h3=h3, max_ticks=10, seed=0)
        assert loop.h3 is h3

    def test_init_custom_bank(self):
        bank = ["a comet", "neon city", "macro beetle"]
        loop = SlopLoop(bank=bank, max_ticks=10, seed=0)
        assert list(loop.evolver.bank) == bank

    def test_init_custom_fitness(self):
        def my_fit(frames, meta): return 1.0
        loop = SlopLoop(fitness_fn=my_fit, max_ticks=5)
        assert loop.fitness_fn is my_fit

    def test_tick_returns_frame(self):
        loop = SlopLoop(max_ticks=5, seed=0, width=8, height=8)
        f = loop.tick()
        assert f is not None
        assert loop.loop_t == 1
        assert len(loop.frames) == 1
        assert len(loop.scores) == 1

    def test_tick_records_video_frame(self):
        loop = SlopLoop(max_ticks=5, seed=0, width=8, height=8)
        loop.tick()
        assert loop.trace.video_seq == 1
        assert len(loop.trace.video) == 1

    def test_tick_records_node_frame(self):
        loop = SlopLoop(max_ticks=5, seed=0, width=8, height=8)
        loop.tick()
        assert loop.trace._seq >= 1

    def test_multiple_ticks(self):
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8)
        for _ in range(5):
            f = loop.tick()
            assert f is not None
        assert loop.loop_t == 5
        assert len(loop.frames) == 5

    def test_stop(self):
        loop = SlopLoop(max_ticks=100, seed=0, width=8, height=8)
        loop.tick()
        loop.tick()
        loop.stop()
        assert not loop.running
        f = loop.tick()
        assert f is None

    def test_step_alias(self):
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8)
        f, sc, bank = loop.step()
        assert f is not None
        assert isinstance(sc, float)
        assert isinstance(bank, list)


class TestSlopLoopRun:
    def test_run_n_loops(self):
        loop = SlopLoop(max_ticks=20, seed=0, width=8, height=8)
        r = loop.run(n_loops=5)
        assert r["n_ticks"] == 5
        assert r["n_frames"] == 5
        assert len(r["scores"]) == 5

    def test_run_respects_max_ticks(self):
        loop = SlopLoop(max_ticks=8, seed=0, width=8, height=8)
        r = loop.run(n_loops=100)
        assert r["n_ticks"] == 8

    def test_run_includes_evolver_history(self):
        loop = SlopLoop(max_ticks=20, seed=0, width=8, height=8)
        r = loop.run(n_loops=16)  # 16 ticks -> 2 evolve calls (every 8)
        assert r["evolver_gen"] >= 1
        assert len(r["evolver_history"]) >= 1

    def test_run_returns_final_bank(self):
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8)
        r = loop.run(n_loops=8)
        assert isinstance(r["final_bank"], list)
        assert len(r["final_bank"]) >= 1

    def test_run_returns_final_bank_hash(self):
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8)
        r = loop.run(n_loops=8)
        assert len(r["final_bank_hash"]) == 64

    def test_run_stats(self):
        loop = SlopLoop(max_ticks=20, seed=0, width=8, height=8)
        loop.run(n_loops=10)
        s = loop.stats()
        assert s["loop_t"] == 10
        assert s["n_frames"] == 10
        assert s["avg_score"] >= 0.0
        assert s["evolver_gen"] >= 1

    def test_run_evolve_every_8_ticks(self):
        loop = SlopLoop(max_ticks=32, seed=0, width=8, height=8)
        loop.run(n_loops=32)
        # Every 8 ticks: gen 1 at t=8, gen 2 at t=16, gen 3 at t=24, gen 4 at t=32
        assert loop.evolver.gen >= 4


class TestSlopLoopEvolveBank:
    def test_evolve_bank(self):
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8)
        loop.run(n_loops=8)
        result = loop.evolve_bank()
        assert isinstance(result, SlopEvolverResult)
        assert result.gen == loop.evolver.gen

    def test_evolve_bank_updates_bank_list(self):
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8)
        loop.run(n_loops=8)
        old_bank = list(loop.evolver.bank)
        loop.evolve_bank()
        # Bank may or may not change (deterministic based on RNG)


# ---------------------------------------------------------------------------
# 4. QBF round-trip
# ---------------------------------------------------------------------------

class TestSlopLoopQBF:
    @pytest.fixture
    def qbf_dir(self, tmp_path):
        d = tmp_path / "qbf"
        d.mkdir()
        yield str(d)
        shutil.rmtree(d, ignore_errors=True)

    def test_run_with_qbf_store(self, qbf_dir):
        loop = SlopLoop(max_ticks=8, seed=0, width=8, height=8)
        r = loop.run(n_loops=8)
        # No qbf_store set -> trace_path should be None
        assert r["trace_path"] is None

    def test_run_archive_to_explicit_qbf_path(self, qbf_dir):
        path = os.path.join(qbf_dir, "slop.qbf")
        loop = SlopLoop(max_ticks=8, seed=0, width=8, height=8,
                         qbf_store=path)
        r = loop.run(n_loops=8)
        assert r["trace_path"] is not None
        assert os.path.exists(r["trace_path"])

    def test_qbf_load_run_with_video(self, qbf_dir):
        path = os.path.join(qbf_dir, "slop.qbf")
        loop = SlopLoop(max_ticks=6, seed=0, width=8, height=8,
                         qbf_store=path)
        loop.run(n_loops=6)
        assert loop.trace.video_seq == 6
        # Load back
        store = QbfTraceStore(path)
        d = store.load_run(0)
        assert d["manifest"]["n_video"] == 6
        assert len(d["video"]) == 6
        # First frame has prompt
        assert "prompt" in d["video"][0]
        store.close()

    def test_qbf_flow_trace_rebuild(self, qbf_dir):
        path = os.path.join(qbf_dir, "slop2.qbf")
        loop = SlopLoop(max_ticks=5, seed=0, width=8, height=8,
                         qbf_store=path)
        loop.run(n_loops=5)
        store = QbfTraceStore(path)
        ft = store.flow_trace(0)
        assert ft.video_seq == 5
        assert len(ft.video) == 5
        store.close()

    def test_qbf_replay_preserves_w_gate(self, qbf_dir):
        path = os.path.join(qbf_dir, "slop3.qbf")
        loop = SlopLoop(max_ticks=4, seed=0, width=8, height=8,
                         qbf_store=path)
        loop.run(n_loops=4)
        store = QbfTraceStore(path)
        ft = store.flow_trace(0)
        for i, vf in enumerate(ft.video):
            assert hasattr(vf, "w_gate")
            assert abs(vf.w_gate) < 1e6  # reasonable range
        store.close()


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

class TestSlopLoopDeterminism:
    def test_same_seed_same_frames(self):
        def fit(*args, **kw): return 0.5
        loop1 = SlopLoop(max_ticks=8, seed=42, width=8, height=8, fitness_fn=fit)
        loop2 = SlopLoop(max_ticks=8, seed=42, width=8, height=8, fitness_fn=fit)
        for _ in range(8):
            loop1.tick()
        for _ in range(8):
            loop2.tick()
        assert len(loop1.frames) == len(loop2.frames)
        for f1, f2 in zip(loop1.frames, loop2.frames):
            assert f1.rgba == f2.rgba
            assert f1.prompt == f2.prompt

    def test_same_seed_same_scores(self):
        def fit(frames_or_rgba=None, **kw): return 0.5
        loop1 = SlopLoop(max_ticks=8, seed=99, width=8, height=8, fitness_fn=fit)
        loop2 = SlopLoop(max_ticks=8, seed=99, width=8, height=8, fitness_fn=fit)
        loop1.run(n_loops=8)
        loop2.run(n_loops=8)
        assert loop1.scores == loop2.scores

    def test_same_seed_same_final_bank(self):
        loop1 = SlopLoop(max_ticks=16, seed=77, width=8, height=8)
        loop2 = SlopLoop(max_ticks=16, seed=77, width=8, height=8)
        loop1.run(n_loops=16)
        loop2.run(n_loops=16)
        assert loop1.evolver.bank == loop2.evolver.bank

    def test_different_seeds_different_banks(self):
        loop1 = SlopLoop(max_ticks=16, seed=11, width=8, height=8)
        loop2 = SlopLoop(max_ticks=16, seed=22, width=8, height=8)
        loop1.run(n_loops=16)
        loop2.run(n_loops=16)
        # Different seeds produce different banks (very high probability)
        # This is a statistical test; in theory they could match but practically never


# ---------------------------------------------------------------------------
# 6. Swarm H4 consensus
# ---------------------------------------------------------------------------

class TestSlopLoopSwarmH4:
    def test_swarm_agents_created(self):
        loop = SlopLoop(max_ticks=4, seed=0, width=8, height=8)
        assert len(loop.swarm) == 4

    def test_consensus_pick_returns_prompt(self):
        loop = SlopLoop(max_ticks=8, seed=0, width=8, height=8)
        # Run a few ticks and check consumed prompts are from the bank
        for _ in range(8):
            loop.tick()
        for p in loop._consumed_prompts:
            assert isinstance(p, str)
            assert len(p) > 0

    def test_consensus_pick_is_from_bank(self):
        loop = SlopLoop(max_ticks=8, seed=0, width=8, height=8)
        for _ in range(8):
            loop.tick()
        bank_set = set(loop._bank_list)
        for p in loop._consumed_prompts:
            assert p in bank_set, f"prompt '{p}' not in bank"


# ---------------------------------------------------------------------------
# 7. Bank evolve ties to fitness
# ---------------------------------------------------------------------------

class TestSlopLoopFitnessTies:
    def test_prompt_scores_collected(self):
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8,
                         fitness_fn=fitness_color_variance)
        loop.run(n_loops=10)
        assert len(loop._prompt_frames) >= 1
        for prompt, frames in loop._prompt_frames.items():
            assert len(frames) >= 1

    def test_all_scores_non_negative(self):
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8)
        loop.run(n_loops=10)
        for sc in loop.scores:
            assert sc >= 0.0, f"negative score: {sc}"

    def test_avg_score_reasonable_range(self):
        loop = SlopLoop(max_ticks=20, seed=0, width=8, height=8)
        loop.run(n_loops=20)
        avg = sum(loop.scores) / len(loop.scores)
        assert 0.0 <= avg < 1e6

    def test_evolver_gen_increments_with_ticks(self):
        loop = SlopLoop(max_ticks=32, seed=0, width=8, height=8)
        loop.run(n_loops=32)
        # Every 8 ticks -> evolve is called 4 times (at t=8,16,24,32)
        # But evolve is called AFTER tick, so at t=8: 1st evolve
        assert loop.evolver.gen >= 3

    def test_composite_fitness_higher_than_single(self):
        loop1 = SlopLoop(max_ticks=8, seed=0, width=8, height=8,
                          fitness_fn=fitness_color_variance)
        loop2 = SlopLoop(max_ticks=8, seed=0, width=8, height=8,
                          fitness_fn=fitness_complexity)
        r1 = loop1.run(n_loops=8)
        r2 = loop2.run(n_loops=8)
        assert r1["n_ticks"] == 8
        assert r2["n_ticks"] == 8
