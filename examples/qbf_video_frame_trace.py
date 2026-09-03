"""iter33 Aspect 4: QBF trace of H3 frames (record/replay through .qbf).

End-to-end demo: an H3 stub generates a sequence of RGBA frames, the
FlowTrace captures them (one VideoFrameEntry per tick), the QBF shard
archives the whole run, and a fresh trace + replay rebuild the exact
same H4-decoded frames.

Pipeline topology:
  H3Stub (deterministic)
      -> H3Session.tick() per tick
      -> H4 RGBA decoder (H4 gate: W=log_A, Z/Y/X=linear BGR)
      -> FlowTrace.record_video_frame (per-tick ring, max_frames)
      -> QbfTraceStore.append_run (one shard, n_video blobs)
      -> QbfTraceStore.load_run -> flow_trace() (replay-ready)
      -> FrameEntry.video_seq + prompts + seeds match

This is the "H3 trace is portable" guarantee: every recorded frame
travels through the .qbf shard with its prompt + seed + w/x/y/z gate
latches so a future H3 session can replay the same H4-decoded colors.

Usage:
  ~/runtime/.venv/bin/python -m examples.qbf_video_frame_trace
"""
from __future__ import annotations

import os
import tempfile

from atomic import (
    FlowTrace, VideoFrameEntry,
    QbfTraceStore, H3Stub, H3Session, h4_gate,
)


PROMPTS = [
    "a slow orbit around a frozen comet",
    "neon city under heavy rain, far future",
    "macro shot of a beetle on wet bark",
    "saturn rings dissolving into ribbons of light",
]


def _decode_h4(rgba: bytes, width: int, height: int):
    """Decode one frame via the H4 gate (same as viz_video atom).

    Returns (decoded_rgba, w_gate, x_gate, y_gate, z_gate) for the LAST pixel.
    """
    import math
    n = len(rgba) // 4
    out = bytearray(n * 4)
    last_w, last_x, last_y, last_z = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        j = i * 4
        a_raw = rgba[j + 3]
        r_raw = rgba[j]
        g_raw = rgba[j + 1]
        b_raw = rgba[j + 2]
        a_log = math.log(max(1, a_raw))
        w_row, z_row, y_row, x_row = h4_gate(
            (a_log, float(b_raw), float(g_raw), float(r_raw)))
        out[j] = max(0, min(255, int(round(x_row))))
        out[j + 1] = max(0, min(255, int(round(y_row))))
        out[j + 2] = max(0, min(255, int(round(z_row))))
        out[j + 3] = a_raw
        last_w, last_x, last_y, last_z = w_row, x_row, y_row, z_row
    return bytes(out), last_w, last_x, last_y, last_z


def main():
    print("=" * 60)
    print("iter33 Aspect 4: QBF trace of H3 frames")
    print("=" * 60)

    width = 16
    height = 16

    print(f"\n[1] Building H3Stub (deterministic, {width}x{height})")
    h3 = H3Stub(width=width, height=height)
    session = H3Session(h3, prompts=PROMPTS, frames_per_prompt=1)
    print(f"  H3Stub: {width}x{height}, 4 prompts in bank")

    print(f"\n[2] Recording 8 ticks: H3 -> H4 -> FlowTrace.video ring")
    trace = FlowTrace(max_frames=20)
    for t in range(8):
        frame = session.tick()
        if frame is None:
            continue
        # Decode via H4 gate, get last-pixel gate latches
        _, w_gate, x_gate, y_gate, z_gate = _decode_h4(
            frame.rgba, width, height)
        # Record in the trace's video ring
        trace.record_video_frame(
            rgba=frame.rgba, t=t,
            prompt=frame.prompt,
            seed=frame.seed,
            h3_latency_ms=frame.h3_latency_ms,
            width=width, height=height,
            w_gate=w_gate, x_gate=x_gate,
            y_gate=y_gate, z_gate=z_gate,
        )
        # Also record a synthetic node frame for the regular trace
        trace.record_node("c0", "const", {}, {"cv": 1.0}, 50.0, t=t,
                          tag=f"tick_{t}")
    print(f"  video_seq = {trace.video_seq}, "
          f"frames = {trace._seq}")
    assert trace.video_seq == 8
    assert len(trace.video) == 8
    # Verify the last recorded video frame
    last = trace.video[-1]
    print(f"  last frame: t={last.t}, prompt={last.prompt[:30]!r}, "
          f"seed={last.seed}, W={last.w_gate:.2f}")

    print(f"\n[3] Snapshot the trace")
    snap = trace.snapshot()
    print(f"  n_video = {snap['n_video']}, vseq = {snap['vseq']}")
    print(f"  video[0] keys: {list(snap['video'][0].keys())}")

    print(f"\n[4] Archiving to a .qbf shard")
    tmp = tempfile.mkdtemp(prefix="qbf_video_")
    path = os.path.join(tmp, "frames.qbf")
    store = QbfTraceStore(path)
    manifest = store.append_run(trace, note="8-frame H3 demo")
    print(f"  shard: {path}")
    print(f"  manifest: n_frames={manifest['n_frames']}, "
          f"n_video={manifest['n_video']}, vseq={manifest['vseq']}")
    assert manifest["n_video"] == 8
    assert manifest["vseq"] == 8

    print(f"\n[5] load_run() with video")
    d = store.load_run(0)
    print(f"  manifest.n_video = {d['manifest']['n_video']}")
    print(f"  video entries: {len(d['video'])}")
    assert len(d["video"]) == 8
    # Verify rgba bytes were decoded
    rgba0 = d["video"][0].get("rgba")
    assert isinstance(rgba0, bytes), \
        f"expected bytes, got {type(rgba0)}"
    assert len(rgba0) == width * height * 4
    print(f"  video[0].rgba: {len(rgba0)} bytes "
          f"(width*height*4 = {width*height*4})")
    print(f"  video[0].prompt: {d['video'][0]['prompt'][:30]!r}")
    print(f"  video[0].w_gate: {d['video'][0]['w_gate']:.2f}")

    print(f"\n[6] flow_trace() rebuild (replay-ready)")
    ft = store.flow_trace(0)
    print(f"  video_seq = {ft.video_seq}")
    assert ft.video_seq == 8
    assert len(ft.video) == 8
    # Verify prompts/seeds/gates preserved
    for i in range(8):
        orig = trace.video[i]
        new = ft.video[i]
        assert orig.t == new.t
        assert orig.prompt == new.prompt
        assert orig.seed == new.seed
        assert abs(orig.w_gate - new.w_gate) < 1e-6
        assert orig.rgba == new.rgba, f"frame {i} rgba mismatch"
    print(f"  All 8 frames match bit-exact (prompt, seed, w_gate, rgba)")

    print(f"\n[7] Determinism: rebuild trace from snapshot == original")
    rebuilt = FlowTrace.from_snapshot(trace.snapshot())
    assert rebuilt.video_seq == trace.video_seq
    assert len(rebuilt.video) == len(trace.video)
    for i in range(8):
        assert rebuilt.video[i].prompt == trace.video[i].prompt
        assert rebuilt.video[i].rgba == trace.video[i].rgba
    print(f"  snapshot round-trip OK (8 frames preserved)")

    print(f"\n[8] Multi-run: 2 sessions in one shard")
    # Build a second H3 session (fresh max_ticks counter)
    h3_2 = H3Stub(width=width, height=height)
    session_2 = H3Session(h3_2, prompts=PROMPTS[:2],
                           frames_per_prompt=1, max_ticks=100)
    trace2 = FlowTrace(max_frames=10)
    for t in range(3):
        frame = session_2.tick()
        if frame is None:
            continue
        _, w_gate, _, _, _ = _decode_h4(frame.rgba, width, height)
        trace2.record_video_frame(
            rgba=frame.rgba, t=t,
            prompt=frame.prompt, seed=frame.seed,
            h3_latency_ms=frame.h3_latency_ms,
            width=width, height=height, w_gate=w_gate,
        )
        trace2.record_node("c0", "const", {}, {"cv": 1.0}, 50.0, t=t)
    m2 = store.append_run(trace2, note="3-frame second session")
    runs = store.runs()
    print(f"  runs: {runs}")
    print(f"  run 1: n_video=8, run 2: n_video={m2['n_video']}")
    assert len(runs) == 2
    assert m2["n_video"] == 3

    store.close()
    try:
        os.unlink(path)
    except OSError:
        pass
    os.rmdir(tmp)

    print("\n" + "=" * 60)
    print("[qbf_video_frame_trace] ALL CHECKS PASSED")
    print("=" * 60)
    return {
        "n_video_per_run": [8, 3],
        "total_runs": 2,
    }


if __name__ == "__main__":
    main()
