"""Programs: built-in demo programs the UI can launch by name.

Each `build(name)` returns a Program.  Names are short, lowercase, and
also become the URL path on the server (e.g. /run/clock_counter).
"""
from __future__ import annotations

from typing import Callable

from ..program import Block, Program, Wire


def _clock_counter() -> Program:
    return Program(
        "clock_counter",
        description="clock_bpm -> accum -> viz_series",
        blocks=[
            Block("clk", "clock_bpm", {"bpm": 120}),
            Block("cnt", "accum", {"per_tick": 1}),
            Block("v0", "viz_series"),
        ],
        wires=[Wire("clk.trig", "cnt.in"),
               Wire("cnt.acc", "v0.in")],
    )


def _gated_clock_counter() -> Program:
    return Program(
        "gated_clock_counter",
        description="clock_bpm AND tap -> accum -> viz_series",
        blocks=[
            Block("clk", "clock_bpm", {"bpm": 60}),
            Block("t1", "tap"),
            Block("a1", "and"),
            Block("cnt", "accum", {"per_tick": 1}),
            Block("v0", "viz_series"),
        ],
        wires=[Wire("clk.trig", "a1.a"),
               Wire("t1.trig", "a1.b"),
               Wire("a1.q", "cnt.in"),
               Wire("cnt.acc", "v0.in")],
    )


def _sine_lfo_scope() -> Program:
    return Program(
        "sine_lfo_scope",
        description="sine_lfo -> gain -> viz_series",
        blocks=[
            Block("src", "sine_lfo", {"hz": 1, "amp": 1}),
            Block("g1", "gain", {"gain": 1}),
            Block("v0", "viz_series"),
        ],
        wires=[Wire("src.cv", "g1.in"),
               Wire("g1.cv", "v0.in")],
    )


def _hadamard_wxyz() -> Program:
    return Program(
        "hadamard_wxyz",
        description="const -> h4_slide -> 4 viz_series (W/X/Y/Z)",
        blocks=[
            Block("src", "const", {"value": 1}),
            Block("h4", "h4_slide"),
            Block("vw", "viz_series"),
            Block("vx", "viz_series"),
            Block("vy", "viz_series"),
            Block("vz", "viz_series"),
        ],
        wires=[Wire("src.cv", "h4.in"),
               Wire("h4.w", "vw.in"),
               Wire("h4.x", "vx.in"),
               Wire("h4.y", "vy.in"),
               Wire("h4.z", "vz.in")],
    )


def _xy_pad() -> Program:
    return Program(
        "xy_pad",
        description="two sine_lfos -> viz_xy",
        blocks=[
            Block("sx", "sine_lfo", {"hz": 0.5, "amp": 1}),
            Block("sy", "sine_lfo", {"hz": 0.7, "amp": 1}),
            Block("v0", "viz_xy"),
        ],
        wires=[Wire("sx.cv", "v0.x"),
               Wire("sy.cv", "v0.y")],
    )


def _wxyz3d_demo() -> Program:
    return Program(
        "wxyz3d_demo",
        description="4 sine_lfos -> viz_wxyz3d (W/X/Y/Z)",
        blocks=[
            Block("sw", "sine_lfo", {"hz": 0.3, "amp": 1}),
            Block("sx", "sine_lfo", {"hz": 0.5, "amp": 1}),
            Block("sy", "sine_lfo", {"hz": 0.7, "amp": 1}),
            Block("sz", "sine_lfo", {"hz": 0.9, "amp": 1}),
            Block("v0", "viz_wxyz3d"),
        ],
        wires=[Wire("sw.cv", "v0.w"),
               Wire("sx.cv", "v0.x"),
               Wire("sy.cv", "v0.y"),
               Wire("sz.cv", "v0.z")],
    )


def _heatmap_demo() -> Program:
    return Program(
        "heatmap_demo",
        description="accum -> viz_series (drives per-tile heat via bus)",
        blocks=[
            Block("clk", "clock_bpm", {"bpm": 120}),
            Block("cnt", "accum", {"per_tick": 1}),
            Block("v0", "viz_series"),
        ],
        wires=[Wire("clk.trig", "cnt.in"),
               Wire("cnt.acc", "v0.in")],
    )


def _heatmap_live() -> Program:
    return Program(
        "heatmap_live",
        description="const -> accum -> viz_heatmap (drives per-tile heatmap live)",
        blocks=[
            Block("src", "const", {"value": 1}),
            Block("h4", "h4_slide"),
            Block("cnt", "accum", {"per_tick": 1}),
            Block("vh", "viz_heatmap"),
        ],
        wires=[Wire("src.cv", "h4.in"),
               Wire("h4.w", "cnt.in"),
               Wire("cnt.acc", "vh.in")],
    )


_REGISTRY: dict[str, Callable[[], Program]] = {
    "clock_counter": _clock_counter,
    "gated_clock_counter": _gated_clock_counter,
    "sine_lfo_scope": _sine_lfo_scope,
    "hadamard_wxyz": _hadamard_wxyz,
    "xy_pad": _xy_pad,
    "wxyz3d_demo": _wxyz3d_demo,
    "heatmap_demo": _heatmap_demo,
    "heatmap_live": _heatmap_live,
}


def build(name: str) -> Program | None:
    fn = _REGISTRY.get(name)
    if fn is None:
        return None
    return fn()


def all_programs() -> list[str]:
    return sorted(_REGISTRY.keys())
