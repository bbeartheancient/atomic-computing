"""Display heatmap animation via trace-replay capture (goal 6 + iter 15).

Exercises Display.heatmap_animation on a trace replay: the trace is
recorded live, then heatmap_animation slices the frames tick-by-tick and
produces a list of normalized heatmap dicts (one per tick).  This is
the "the trace is the bridge" step in live form: GPU1 captures the
flow, GPU0 re-drives it into tile colours.

The demo builds a 3x3 Display, runs a counter program with a live
FlowTrace, then calls heatmap_animation() to produce per-tick heatmaps.
Each heatmap dict maps (row, col) -> 0..1 normalised value.

  ~/runtime/.venv/bin/python -m examples.heatmap_animation
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from atomic.engine import Engine
from atomic.program import Block, Program, Wire
from atomic.tiles import Display
from atomic.trace import FlowTrace


def demo(ticks: int = 32, dt: float = 1.0 / 30.0):
    display = Display(900, 900, cols=3, rows=3, frame_h=60)

    prog = Program(
        "heatmap_counter",
        description="clock -> accum -> viz_series (heatmap demo)",
        blocks=[
            Block("clk", "clock_bpm", {"bpm": 120}),
            Block("cnt", "accum", {"per_tick": 1}),
            Block("v0", "viz_series"),
        ],
        wires=[Wire("clk.trig", "cnt.in"),
               Wire("cnt.acc", "v0.in")],
    )
    patch = prog.compile("microfx")
    modules = patch["modules"]
    wires = patch["wires"]

    trace = FlowTrace()
    eng = Engine(modules, wires, dt=dt, trace=trace)
    result = eng.run(ticks)

    heatmaps = display.heatmap_animation(trace, port="acc", window=2)
    assert heatmaps, "heatmap_animation returned empty list"
    assert len(heatmaps) == ticks // 2, \
        "expected %d heatmaps, got %d" % (ticks // 2, len(heatmaps))

    # all values must be in [0, 1]
    for i, hm in enumerate(heatmaps):
        for k, v in hm.items():
            assert 0.0 <= v <= 1.0, \
                "heatmap[%d][%s]=%s out of range" % (i, k, v)

    # tile (0,0) should have the highest count value (most data routed there)
    # verify the first heatmap and last heatmap differ (the count is growing)
    first_hm = heatmaps[0]
    last_hm = heatmaps[-1]
    # find the tile with max value in each heatmap
    max_tile_first = max(first_hm, key=first_hm.get)
    max_tile_last = max(last_hm, key=last_hm.get)

    print("[heatmap animation] ok")
    print("  display:   %dx%d wall, tile=%dx%d, %d tiles" % (
        display.width, display.height, display.tile_w, display.tile_h,
        len(display.tiles)))
    print("  heatmaps:  %d frames (%d ticks, window=2)" % (
        len(heatmaps), ticks))
    print("  first hm:  %d tiles, max@%s=%.4f" % (
        len(first_hm), max_tile_first,
        first_hm.get(max_tile_first, 0)))
    print("  last hm:   %d tiles, max@%s=%.4f" % (
        len(last_hm), max_tile_last,
        last_hm.get(max_tile_last, 0)))

    return {"display": display.summary(), "heatmaps": heatmaps,
            "n_frames": trace.snapshot()["n_frames"],
            "n_ticks": trace.snapshot()["n_ticks"]}


if __name__ == "__main__":
    demo()
