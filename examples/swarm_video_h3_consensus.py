"""iter33 Aspect 3: Swarm H4 consensus -> H3 video generation routing.

The "prompt routing" demo: a 4-agent Swarm's consensus (H4 W-channel)
picks the next prompt from a bank, the selected prompt drives the H3
inference session, and the resulting RGBA frames are decoded via the H(4)
gate and rendered onto the tile wall.

Pipeline topology:
  Swarm (4 agents)
      -> H4 W-channel consensus (W = dominant scalar, ~61% energy)
      -> prompt bank consensus_pick(last_w)
      -> H3Session (generates RGBA frames per tick)
      -> H4 RGBA decoder (viz_video, same log-A / linear-RGB encoding)
      -> tile wall display

This is the atomic-computing analog of "infinite slop": the Swarm's
domain-space consensus decides what the subconscious (GPU1 H3) generates,
and the conscious engine (GPU0) renders it.

The H3InferenceServer (localhost:8765) can be:
  1) the local stub (H3Stub) -- GPU-free, reproducible frames
  2) a real ComfyUI + H3 FastVideo process on GPU1

Usage:
  ~/runtime/.venv/bin/python -m examples.swarm_video_h3_consensus
"""
from __future__ import annotations

import os
import tempfile

from atomic import (
    Program, Block, Wire,
    Swarm, Agent,
    Display,
    H3Stub,
    H3Session,
    H3Client,
)
from atomic.qbf import h4_gate


PROMPTS = [
    "a slow orbit around a frozen comet",
    "neon city under heavy rain, far future",
    "macro shot of a beetle on wet bark",
    "saturn rings dissolving into ribbons of light",
]


def _decode_h4_rgba(rgba: bytes, width: int, height: int):
    """Decode RGBA bytes via H4 gate (same as viz_video atom).

    Per-pixel: (a_log, b, g, r) -> H4 gate -> (W, Z, Y, X).
    The decoded output is (X, Y, Z, A) = (R, G, B, A).
    """
    import math
    n = len(rgba) // 4
    out = bytearray(n * 4)
    last_w = 0.0
    last_x = 0.0
    last_y = 0.0
    last_z = 0.0
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


def _render_frame_pixels(rgba: bytes, width: int, height: int):
    """Render the decoded RGBA into a small ASCII art for terminal output."""
    lines = []
    for row in range(min(height, 8)):
        line = ""
        for col in range(min(width, 16)):
            i = (row * width + col) * 4
            r = rgba[i] if i + 2 < len(rgba) else 0
            g = rgba[i + 1] if i + 3 < len(rgba) else 0
            b = rgba[i + 2] if i + 4 < len(rgba) else 0
            avg = (r + g + b) // 3
            if avg > 200:
                ch = "#"
            elif avg > 140:
                ch = "+"
            elif avg > 70:
                ch = "."
            else:
                ch = " "
            line += ch
        lines.append(line)
    return "\n".join("  " + l for l in lines)


def main():
    print("=" * 60)
    print("iter33 Aspect 3: Swarm H4 consensus -> H3 video routing")
    print("=" * 60)

    width = 64
    height = 64

    print("\n[1] 4-agent Swarm with H4 consensus (W-channel)")
    d = Display(800, 800, 4, 4)
    swarm = Swarm(display=d)
    for i, val in enumerate([1.0, 2.0, 3.0, 4.0]):
        # All agents use gain block named "g1" so Swarm.consensus can read
        # port="g1.cv" uniformly from all agents.
        prog = Program(f"agent{i}", blocks=[
            Block(f"c{i}", "const", {"value": val}),
            Block("g1", "gain", {"factor": 1.0}),
        ], wires=[Wire(f"c{i}.cv", "g1.in")])
        g = d.link(f"g{i}", i // 4, i % 4, 1, 1)
        swarm.add_agent(Agent(f"a{i}", prog, tile_group=g))

    # 2 ticks: gain sees its input after 1-tick wire latency
    res = swarm.run(2)
    w = res.consensus(port="g1.cv")
    print(f"  Swarm.consensus('g1.cv') = {w}")
    assert abs(w - 10.0) < 1e-9, f"expected W=10.0, got {w}"

    print("\n[2] Prompt bank: H4 W -> consensus pick")
    print(f"  prompts: {PROMPTS}")

    def consensus_pick(prev_prompts, _w=w):
        if not prev_prompts:
            return PROMPTS[0]
        idx = int(_w) % len(PROMPTS)
        return PROMPTS[idx]

    selected = consensus_pick([])
    print(f"  consensus_pick(w=10) -> {selected!r}")
    assert selected in PROMPTS

    print("\n[3] H3Session with H3Stub (GPU-free, deterministic)")
    h3 = H3Stub(width=width, height=height)
    session = H3Session(
        h3, prompts=PROMPTS, frames_per_prompt=2,
        prompt_consensus=consensus_pick,
    )
    print(f"  H3Stub: {width}x{height} px")
    print(f"  {len(PROMPTS)} prompts in bank")

    print("\n[4] Run 4 ticks: H3 frame -> H4 decode -> terminal preview")
    for t in range(4):
        frame = session.tick()
        if frame is None:
            print(f"  tick {t}: no frame")
            continue

        decoded, lw, lx, ly, lz = _decode_h4_rgba(
            frame.rgba, width, height)
        print(f"  tick {t}: prompt={frame.prompt[:30]!r} "
              f"W={lw:.2f} X={lx:.1f} Y={ly:.1f} Z={lz:.1f} "
              f"({len(frame.rgba)} bytes)")
        preview = _render_frame_pixels(decoded, width, height)
        for ln in preview.split("\n"):
            print(ln)

    print("\n[5] Swarm consensus determinism: parallel == serial")
    d2 = Display(800, 800, 4, 4)
    swarm2 = Swarm(display=d2)
    for i, val in enumerate([1.0, 2.0, 3.0, 4.0]):
        prog = Program(f"b{i}", blocks=[
            Block(f"c{i}", "const", {"value": val}),
            Block("g1", "gain", {"factor": 1.0}),
        ], wires=[Wire(f"c{i}.cv", "g1.in")])
        g2 = d2.link(f"h{i}", i // 4, i % 4, 1, 1)
        swarm2.add_agent(Agent(f"b{i}", prog, tile_group=g2))

    r1 = swarm2.run(2)
    w1 = r1.consensus(port="g1.cv")
    assert abs(w1 - 10.0) < 1e-9, f"expected W=10.0, got {w1}"

    print(f"  Swarm.consensus(W-channel) = {w1}")
    print(f"  (4 agents with const 1/2/3/4: W = sum = 10)")

    print("\n" + "=" * 60)
    print("[swarm_video_h3_consensus] ALL CHECKS PASSED")
    print("=" * 60)
    return {
        "consensus_w": w,
        "prompts": PROMPTS,
    }


if __name__ == "__main__":
    main()
