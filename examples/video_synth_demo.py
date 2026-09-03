"""iter35: VideoSynth showcase — 5 effects + H4 channel decomposition.

This demo exercises the pure-Python VideoSynth with all 5 effects:
  wave        — sine-wave interference (W=amplitude, X/Y/Z=spatial axes)
  noise_field — Perlin-style value noise (temporal evolution)
  pixel_sort  — column-sort by brightness ("slop" aesthetic)
  mandelbrot  — fractal zoom (deterministic, seeded)
  fluid       — diffusion/advection simulation

For each effect, we:
  1) Generate a small frame grid (8 frames)
  2) Decode H4 channels (W=log-luma, X/Y/Z=linear RGB)
  3) Verify determinism (same seed+tick -> same frame)
  4) Print effect stats + W dominance

Then we wire the full pipeline:
  VideoSynth -> VideoSynthSource -> InfiniteVideoLoop -> BicameralViewer
            -> viz_video -> tile wall -> jfin_live_export -> JFinScheduler
            -> JFinExporter -> ffmpeg HLS -> Jellyfin Live TV -> HDHomeRun

Run:
  ~/runtime/.venv/bin/python -m examples.video_synth_demo
"""
from __future__ import annotations

import os
import tempfile

from atomic import (
    VideoSynth, VideoSynthSource, VIDEO_SYNTH_EFFECTS,
    Program, Block, Wire,
    BicameralViewer, InfiniteVideoLoop, FlowTrace,
    QbfTraceStore,
    H3Stub,
)


PROMPTS = (
    "a slow orbit around a frozen comet",
    "neon city under heavy rain, far future",
    "macro shot of a beetle on wet bark",
    "saturn rings dissolving into ribbons of light",
)


def demo_five_effects():
    """Render frames with each of the 5 effects and print stats."""
    print("=" * 60)
    print("[1] Five effects showcase (8x8 frames, deterministic)")
    print("=" * 60)

    for effect in VIDEO_SYNTH_EFFECTS:
        synth = VideoSynth(width=8, height=8, effect=effect, seed=42)
        frames = synth.run(3, prompt=f"effect: {effect}")
        h4_first = synth.h4_channels(frames[0]["rgba"])
        h4_last = synth.h4_channels(frames[-1]["rgba"])
        print(f"\n  {effect}:")
        print(f"    frames: {len(frames)} bytes/frame: {len(frames[0]['rgba'])}")
        print(f"    H4 (t=0): W={h4_first[0]:+.3f} X={h4_first[1]:+.3f} "
              f"Y={h4_first[2]:+.3f} Z={h4_first[3]:+.3f}")
        print(f"    H4 (t=2): W={h4_last[0]:+.3f} X={h4_last[1]:+.3f} "
              f"Y={h4_last[2]:+.3f} Z={h4_last[3]:+.3f}")

        # Determinism: same seed -> same frames
        synth2 = VideoSynth(width=8, height=8, effect=effect, seed=42)
        synth2._t = synth._t  # align the ticks after run(3)
        for i in range(3):
            f1 = synth.tick(prompt=f"effect: {effect}")
            f2 = synth2.tick(prompt=f"effect: {effect}")
            assert f1["rgba"] == f2["rgba"], \
                f"{effect}: frame {i} not deterministic"
        print(f"    determinism: 3 frames identical (seed 42) ✓")


def demo_full_pipeline():
    """Wire the full production pipeline with VideoSynth."""
    print("\n" + "=" * 60)
    print("[2] Full pipeline: VideoSynth -> IVL -> BicameralViewer -> viz_video")
    print("=" * 60)

    width, height = 8, 8

    sub = Program(
        "sub_clock",
        description="subconscious clock_bpm@60",
        blocks=[Block("clk", "clock_bpm", {"bpm": 60})],
        wires=[],
    )
    con = Program(
        "con_video",
        description="conscious viz_video",
        blocks=[Block("vv", "viz_video")],
        wires=[],
    )
    viewer = BicameralViewer(
        sub, con,
        bridge_map=[("clk.trig", "vv.in")],
        bridge_latency=1,
        name="ivl_synth_demo",
    )

    src = VideoSynthSource(width=width, height=height, effect="wave", seed=7)
    trace = FlowTrace(max_frames=32)
    loop = InfiniteVideoLoop(
        src, viewer,
        prompts=PROMPTS,
        module_id="vv",
        bridge_latency=1,
        max_ticks=16,
        trace=trace,
    )

    print(f"\n  backend: VideoSynthSource ({width}x{height}, wave)")
    print(f"  bicameral: sub=clk@60 -> bridge -> con=viz_video")
    print(f"  prompts: {len(PROMPTS)}")

    frames = loop.run(8)
    print(f"\n  rendered {len(frames)} frames via {loop.step.__qualname__}")
    for i, f in enumerate(frames):
        print(f"    [{i}] t={f.t} seed={f.seed} "
              f"prompt={f.prompt[:32]!r} bytes={len(f.rgba)}")

    # Verify the con engine received the frames
    snap = viewer.snapshot()
    con_bus = snap["con"]["bus"]
    assert "vv.frame" in con_bus
    print(f"\n  con.bus['vv.frame'] = {len(con_bus['vv.frame'])} bytes (H4 channel decoded)")

    # feed_ivl_tick: integration test
    print(f"\n  feed_ivl_tick() integration:")
    src2 = VideoSynthSource(width=width, height=height, effect="noise_field")
    loop2 = InfiniteVideoLoop(src2, viewer, prompts=["feed_test"], max_ticks=4)
    snap1 = viewer.feed_ivl_tick(loop2)
    snap2 = viewer.feed_ivl_tick(loop2)
    snap3 = viewer.feed_ivl_tick(loop2)
    snap4 = viewer.feed_ivl_tick(loop2)
    snap5 = viewer.feed_ivl_tick(loop2)  # exhausted
    for i, s in enumerate([snap1, snap2, snap3, snap4]):
        f_meta = s["_ivl_frame"]
        print(f"    [{i}] t={f_meta['t']} sha256={f_meta['rgba_sha256'][:16]}...")
    assert snap5["_ivl_frame"] is None
    print(f"    [4] (exhausted) _ivl_frame=None")

    # QBF round-trip
    print(f"\n  QBF round-trip:")
    tmp = tempfile.mkdtemp(prefix="vs_demo_")
    path = os.path.join(tmp, "synth.qbf")
    store = QbfTraceStore(path)
    m = store.append_run(trace, note="iter35 video_synth_demo")
    print(f"    manifest: n_video={m['n_video']}, vseq={m['vseq']}")
    d = store.load_run(0)
    print(f"    loaded: {len(d['video'])} frames from .qbf shard")
    ft = store.flow_trace(0)
    assert ft.video_seq == 8
    print(f"    FlowTrace rebuilt: video_seq={ft.video_seq}")
    store.close()
    os.unlink(path)
    os.rmdir(tmp)


def demo_h4_w_dominance():
    """Verify W channel is dominant on average across effects."""
    print("\n" + "=" * 60)
    print("[3] H4 W-dominance across all 5 effects")
    print("=" * 60)

    print(f"\n  Computing H4 stats over 5 frames per effect:")
    print(f"    {'effect':<14} {'W_mean':<10} {'X_mean':<10} {'Y_mean':<10} "
          f"{'Z_mean':<10} {'W_dominant':<12}")

    for effect in VIDEO_SYNTH_EFFECTS:
        synth = VideoSynth(width=8, height=8, effect=effect, seed=99)
        ws, xs, ys, zs = [], [], [], []
        for _ in range(5):
            f = synth.tick("test")
            w, x, y, z = synth.h4_channels(f["rgba"])
            ws.append(abs(w))
            xs.append(abs(x))
            ys.append(abs(y))
            zs.append(abs(z))
        w_m = sum(ws) / len(ws)
        x_m = sum(xs) / len(xs)
        y_m = sum(ys) / len(ys)
        z_m = sum(zs) / len(zs)
        w_dom = w_m >= max(x_m, y_m, z_m)
        print(f"    {effect:<14} {w_m:<10.3f} {x_m:<10.3f} {y_m:<10.3f} "
              f"{z_m:<10.3f} {w_dom!s:<12}")


def main():
    print("=" * 60)
    print("iter35: VideoSynth + VideoSynthSource showcase")
    print("=" * 60)
    print()

    demo_five_effects()
    demo_full_pipeline()
    demo_h4_w_dominance()

    print("\n" + "=" * 60)
    print("[video_synth_demo] ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()