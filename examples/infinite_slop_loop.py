"""Infinite Slop Loop demo (iter 41): H4 consensus + H3Stub + QBF trace + bank evolution.

End-to-end demonstration of the atomic-computing analog of infinite-livestream::

    4-agent Swarm (H4 W-consensus) -> H3Stub (per-tick RGBA frame)
    -> FlowTrace (video frames) -> Fitness (color variance / H4 latches)
    -> SlopEvolver (mutate bank) -> QBF shard (portable archive)

The loop runs `n_ticks` iterations (default 32), evolves the prompt bank
every 8 ticks, and archives the trace to a .qbf shard for replay.

Fitness options:
  color_variance  -- higher = more visual activity (default)
  h4_w_latch     -- H4 W-channel from last-pixel H4 gate
  complexity      -- histogram entropy of R/G/B
  composite       -- weighted: alpha*var + beta*complex + gamma*w_latch

Usage::

    python -m examples.infinite_slop_loop            # defaults: 32 ticks, color_variance
    python -m examples.infinite_slop_loop --ticks 64 --fitness complexity
    python -m examples.infinite_slop_loop --fitness composite --seed 42
    python -m examples.infinite_slop_loop --h3-stub  # always uses H3Stub (no GPU)
    python -m examples.infinite_slop_loop --no-fallback  # live ComfyUI, no stub fallback (iter 44)

Swap H3Stub for H3Client (real H3 on GPU1) by changing the h3= argument
in the SlopLoop constructor. The trace archive is portable regardless.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import shutil

from atomic import (
    SlopLoop,
    fitness_color_variance, fitness_h4_w_latch,
    fitness_complexity, composite_fitness,
    QbfTraceStore,
    FastH3Client, FastH3Stub,
)


def _make_fitness(name):
    if name == "color_variance":
        return fitness_color_variance
    if name == "h4_w_latch":
        return fitness_h4_w_latch
    if name == "complexity":
        return fitness_complexity
    if name == "composite":
        return composite_fitness
    raise ValueError("unknown fitness: %r" % name)


def demo(n_ticks=32, fitness="color_variance", seed=0, width=64, height=64,
         archive=True, verbose=True, h3=None, h3_endpoint=None,
         h3_steps=4, no_fallback=False):
    """Run the infinite slop loop.

    Args:
        n_ticks:   Number of ticks to run (default 32).
        fitness:    Fitness function name (default "color_variance").
        seed:      RNG seed for determinism.
        width:     Frame width in pixels (default 64).
        height:    Frame height in pixels (default 64).
        archive:   Whether to write a QBF shard (default True).
        verbose:   Print stats to stdout.
        h3:        H3 backend (FastH3Client / FastH3Stub). Default: FastH3Stub.
        h3_endpoint: ComfyUI URL (only used when h3 is None). Default: 8188.
        h3_steps:  FastH3 steps (only used when h3 is None). Default 4.
        no_fallback: If True, use FastH3Client with fallback=None (iter 44);
                     raises on ComfyUI failure instead of swapping in stub.

    Returns the SlopLoop result dict.
    """
    fit_fn = _make_fitness(fitness)

    qbf_dir = tempfile.mkdtemp(prefix="slop_loop_")
    qbf_path = os.path.join(qbf_dir, "slop.qbf") if archive else None

    if h3 is None:
        endpoint = h3_endpoint or "http://127.0.0.1:8188"
        fallback = None if no_fallback else FastH3Stub(
            width=width, height=width, n_frames=1)
        h3 = FastH3Client(
            endpoint=endpoint,
            fallback=fallback,
            mode="comfyui",
            steps=h3_steps,
        )

    if verbose:
        print("[infinite_slop_loop] starting")
        print("  n_ticks:    %d" % n_ticks)
        print("  fitness:    %s" % fitness)
        print("  seed:       %d" % seed)
        print("  width:      %d" % width)
        print("  height:     %d" % height)
        print("  archive:    %s" % qbf_path)
        print("  h3:         %s" % type(h3).__name__)
        if hasattr(h3, "endpoint"):
            print("  h3_endpoint: %s" % h3.endpoint)

    loop = SlopLoop(
        h3=h3,
        max_ticks=n_ticks,
        seed=seed,
        width=width,
        height=height,
        fitness_fn=fit_fn,
        qbf_store=qbf_path,
    )

    r = loop.run(n_loops=n_ticks)

    if verbose:
        print("\n--- results ---")
        print("  n_ticks:          %d" % r["n_ticks"])
        print("  n_frames:         %d" % r["n_frames"])
        print("  avg_score:        %.4f" % r["avg_score"])
        print("  evolver_gen:      %d" % r["evolver_gen"])
        print("  final_bank_hash:  %s" % r["final_bank_hash"][:16] + "...")
        print("  trace_video_seq:  %d" % r["trace_video_seq"])
        print("  trace_path:       %s" % (r["trace_path"] or "(none)"))
        print("\n  per-tick scores:")
        for i, sc in enumerate(r["scores"]):
            print("    tick %3d: %.4f" % (i, sc))
        print("\n  final prompt bank (%d prompts):" % len(r["final_bank"]))
        for i, p in enumerate(r["final_bank"]):
            print("    [%d] %s" % (i, p))

    # Verify QBF archive
    if archive and r["trace_path"] and os.path.exists(r["trace_path"]):
        store = QbfTraceStore(r["trace_path"])
        d = store.load_run(0)
        assert d["manifest"]["n_video"] == r["n_frames"], \
            "QBF n_video mismatch: %d != %d" % (d["manifest"]["n_video"], r["n_frames"])
        if verbose:
            print("\n  QBF archive verified: %d frames preserved" % d["manifest"]["n_video"])
        store.close()

    if verbose:
        print("\n[infinite_slop_loop] done")

    # Cleanup temp dir
    shutil.rmtree(qbf_dir, ignore_errors=True)

    return r


def main():
    p = argparse.ArgumentParser(description="Iter 41: Infinite Slop Loop demo")
    p.add_argument("--ticks", type=int, default=32,
                   help="Number of ticks to run (default 32)")
    p.add_argument("--fitness", default="color_variance",
                   choices=["color_variance", "h4_w_latch", "complexity", "composite"],
                   help="Fitness function (default: color_variance)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default 0)")
    p.add_argument("--width", type=int, default=64,
                   help="Frame width in pixels (default 64)")
    p.add_argument("--height", type=int, default=64,
                   help="Frame height in pixels (default 64)")
    p.add_argument("--no-archive", action="store_true",
                   help="Skip QBF shard archive")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Suppress verbose output")
    p.add_argument("--h3-stub", action="store_true",
                   help="Use FastH3Stub (no GPU; offline). Default: FastH3Client "
                        "pointing at 8188 with stub fallback.")
    p.add_argument("--h3-endpoint", default="http://127.0.0.1:8188",
                   help="ComfyUI endpoint for FastH3Client "
                        "(default: http://127.0.0.1:8188)")
    p.add_argument("--h3-steps", type=int, default=4,
                   help="FastH3 steps (default 4)")
    p.add_argument("--no-fallback", action="store_true",
                   help="Use FastH3Client with fallback=None against "
                        "h3-endpoint; raises on failure instead of stub "
                        "fallback (iter 44: live ComfyUI vsa path)")

    args = p.parse_args()
    h3 = FastH3Stub(width=args.width, height=args.height, n_frames=1) \
        if args.h3_stub else None
    demo(n_ticks=args.ticks, fitness=args.fitness, seed=args.seed,
         width=args.width, height=args.height,
         archive=not args.no_archive, verbose=not args.quiet,
         h3=h3, h3_endpoint=args.h3_endpoint, h3_steps=args.h3_steps,
         no_fallback=args.no_fallback)


if __name__ == "__main__":
    main()
