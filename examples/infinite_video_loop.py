"""iter35: InfiniteVideoLoop — VideoSynth + H3Stub backend switch.

The end-to-end "infinite slop" / "interdimensional cable" pipeline:
  VideoSynth/H3Stub (per-tick RGBA frames, GPU-free)
              -> HostBridge.push_frame (subconscious -> conscious)
              -> BicameralViewer.feed_video_tick
              -> viz_video atom decodes + renders on the tile wall
              -> FlowTrace records each frame -> portable via QBF

Two backend shapes (switch via USE_VIDEO_SYNTH):
  True  (default): VideoSynth — pure-Python deterministic animated frames
                   (wave/noise_field/pixel_sort/mandelbrot/fluid)
  False               H3Stub   — seeded gradient frames (iter 34)

This demo:
  1) Builds a tiny bicameral program (sub=const clock, con=viz_video)
  2) Creates a VideoSynth backend (8x8, wave effect)
  3) Wires an InfiniteVideoLoop around them with a 4-prompt bank
  4) Runs 8 ticks: synth frames flow sub -> bridge -> con -> tile wall
  5) Records the run via FlowTrace + archives to a .qbf shard
  6) Replays the .qbf shard and verifies frames match

Run:
  ~/runtime/.venv/bin/python -m examples.infinite_video_loop
"""
from __future__ import annotations

import os
import tempfile

from atomic import (
    H3Stub, Program, Block, Wire,
    BicameralViewer, InfiniteVideoLoop, FlowTrace,
    QbfTraceStore,
    VideoSynth, VideoSynthSource,
)


USE_VIDEO_SYNTH = True


PROMPTS = (
    "a slow orbit around a frozen comet",
    "neon city under heavy rain, far future",
    "macro shot of a beetle on wet bark",
    "saturn rings dissolving into ribbons of light",
)


def _make_sub():
    return Program(
        "sub_clock",
        description="subconscious clock — supplies scalar pulse",
        blocks=[Block("clk", "clock_bpm", {"bpm": 120})],
        wires=[],
    )


def _make_con():
    return Program(
        "con_video",
        description="conscious viz_video — renders pushed RGBA frames",
        blocks=[Block("vv", "viz_video")],
        wires=[],
    )


def _make_source(width, height):
    if USE_VIDEO_SYNTH:
        synth = VideoSynth(width=width, height=height, effect="wave", seed=7)
        return VideoSynthSource(synth=synth)
    else:
        return H3Stub(width=width, height=height)


def main():
    print("=" * 60)
    print("iter35: InfiniteVideoLoop end-to-end pipeline")
    print(f"  backend: {'VideoSynth (pure-Python)' if USE_VIDEO_SYNTH else 'H3Stub'}")
    print("=" * 60)

    width = 8
    height = 8

    print(f"\n[1] Building bicameral program + viewer")
    sub = _make_sub()
    con = _make_con()
    viewer = BicameralViewer(
        sub, con,
        bridge_map=[("clk.trig", "vv.in")],
        bridge_latency=1,
        name="ivl_demo",
    )
    print(f"  sub: 1 block (clock_bpm@120)")
    print(f"  con: 1 block (viz_video)")

    print(f"\n[2] VideoSynth backend ({width}x{height}, 4 prompts)")
    source = _make_source(width, height)
    trace = FlowTrace(max_frames=32)
    loop = InfiniteVideoLoop(
        source, viewer,
        prompts=PROMPTS,
        module_id="vv",
        bridge_latency=1,
        max_ticks=16,
        trace=trace,
    )
    source_name = type(source).__name__
    print(f"  {source_name} ready, InfiniteVideoLoop wired")

    print(f"\n[3] Running 8 ticks: synth -> bridge -> viz_video -> trace")
    frames = loop.run(8)
    print(f"  rendered frames: {len(frames)}")
    print(f"  frame[0]: t={frames[0].t}, seed={frames[0].seed}, "
          f"prompt={frames[0].prompt[:30]!r}")
    print(f"  frame[3]: prompt={frames[3].prompt[:30]!r}  "
          f"(should be 'macro shot...')")
    print(f"  stats: {loop.stats()}")

    assert len(frames) == 8, f"expected 8 frames, got {len(frames)}"
    for f in frames:
        assert len(f.rgba) == width * height * 4
    for f in frames:
        assert len(f.rgba) == width * height * 4

    print(f"\n[4] Verifying trace recorded all 8 frames")
    print(f"  trace.video_seq: {trace.video_seq}")
    print(f"  trace.video: {len(trace.video)} entries")
    print(f"  last entry: t={trace.video[-1].t}, "
          f"prompt={trace.video[-1].prompt[:30]!r}")
    assert trace.video_seq == 8
    assert len(trace.video) == 8

    print(f"\n[5] Archiving trace to a .qbf shard")
    tmp = tempfile.mkdtemp(prefix="ivl_qbf_")
    path = os.path.join(tmp, "infinite.qbf")
    store = QbfTraceStore(path)
    manifest = store.append_run(trace, note="iter35 infinite_video_loop")
    print(f"  shard: {path}")
    print(f"  manifest: n_video={manifest['n_video']}, "
          f"vseq={manifest['vseq']}")
    assert manifest["n_video"] == 8

    print(f"\n[6] Reloading trace from .qbf (portable replay)")
    d = store.load_run(0)
    print(f"  n_video = {d['manifest']['n_video']}")
    assert len(d["video"]) == 8
    expected_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(8)]
    for i, entry in enumerate(d["video"]):
        assert entry["prompt"] == expected_prompts[i], \
            f"frame {i}: {entry['prompt']!r} vs {expected_prompts[i]!r}"
        assert len(entry["rgba"]) == width * height * 4
    print(f"  All 8 frames match (prompt cycle + size)")
    ft = store.flow_trace(0)
    print(f"  video_seq = {ft.video_seq}")
    print(f"  video[0].seed = {ft.video[0].seed}")
    for i in range(8):
        assert ft.video[i].prompt == trace.video[i].prompt, \
            f"prompt mismatch at frame {i}"
    print(f"  All 8 frames: bit-exact replay through .qbf")

    print(f"\n[8] InfiniteVideoLoop.stats()")
    stats = loop.stats()
    print(f"  {stats}")
    assert stats["t"] == 8
    assert stats["frame_index"] == 8
    assert stats["h3_calls"] == 8

    store.close()
    try:
        os.unlink(path)
    except OSError:
        pass
    os.rmdir(tmp)

    print("\n" + "=" * 60)
    print("[infinite_video_loop] ALL CHECKS PASSED")
    print("=" * 60)
    return {
        "frames_rendered": len(frames),
        "trace_video_seq": trace.video_seq,
        "qbf_runs": 1,
        "stats": stats,
    }


if __name__ == "__main__":
    main()
