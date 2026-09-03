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


def _jfin_export_demo() -> Program:
    return Program(
        "jfin_export_demo",
        description="const -> viz_video -> jfin_live_export (H3 -> HLS/DASH -> Jellyfin)",
        blocks=[
            Block("src", "const", {"value": 1}),
            Block("vv", "viz_video"),
            Block("jle", "jfin_live_export",
                  {"scheduler_key": "atomic-01", "width": 64, "height": 64}),
        ],
        wires=[Wire("src.cv", "vv.in"),
               Wire("src.cv", "jle.in")],
    )


def _video_live() -> Program:
    return Program(
        "video_live",
        description="viz_video_h3 (polls localhost:8765) -> tile wall",
        blocks=[
            Block("vh3", "viz_video_h3",
                  {"server_url": "http://localhost:8765",
                   "width": 64, "height": 64,
                   "prompt": "a slow orbit around a frozen comet",
                   "n_frames": 1,
                   "timeout_s": 5.0,
                   "max_retries": 3,
                   "enabled": 1.0}),
        ],
        wires=[],
    )


def _feed_video_live() -> Program:
    return Program(
        "feed_video_live",
        description="viz_video (server-push frames via /api/feed_video) -> tile wall",
        blocks=[
            Block("vv", "viz_video"),
        ],
        wires=[],
    )


def _fasth3_video_live() -> Program:
    """iter 36: FastH3 (4-step VSA GGUF) program — renders via viz_fasth3_video.

    The actual FastH3 frames are served by FastH3Session / InfiniteFastH3Loop
    via HostBridge.push_frame into the conscious engine's bus[vfh.frame] key.
    Params mirror FastH3Stub defaults: steps=4, vsa=ON (10% kept), Q5_K_M.
    """
    return Program(
        "fasth3_video_live",
        description="viz_fasth3_video (FastH3 4-step VSA, server-push) -> tile wall",
        blocks=[
            Block("vfh", "viz_fasth3_video",
                  {"capture": 1.0, "steps": 4, "vsa": 1.0,
                   "vsa_keep": 10, "quant": "Q5_K_M"}),
        ],
        wires=[],
    )


def _infinite_slop_loop() -> Program:
    """iter 46: SlopLoop program — feeds viz_video via server-push.

    The SlopLoop wires directly to this program's viz_video block
    (module_id="vv") via Viewer.feed_video_tick(). Switch to this program,
    then start the slop loop from the Slop control bar.

    Pipeline: Swarm H4 consensus -> H3Stub -> Viewer.feed_video_tick(vv)
             -> viz_video renders RGBA -> tile wall
    """
    return Program(
        "infinite_slop_loop",
        description="SlopLoop (H4 consensus + H3Stub + evolve) -> viz_video -> tile wall",
        blocks=[
            Block("vv", "viz_video"),
        ],
        wires=[],
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
    "jfin_export_demo": _jfin_export_demo,
    "video_live": _video_live,
    "feed_video_live": _feed_video_live,
    "fasth3_video_live": _fasth3_video_live,
    "infinite_slop_loop": _infinite_slop_loop,
}


def build(name: str) -> Program | None:
    fn = _REGISTRY.get(name)
    if fn is None:
        return None
    return fn()


def all_programs() -> list[str]:
    return sorted(_REGISTRY.keys())


# ── bicameral programs ─────────────────────────────────────────────────────────


def _bicameral_clock() -> dict:
    return {
        "type": "bicameral",
        "sub": Program(
            "clock_source",
            description="clock_bpm@60 — subconscious clock",
            blocks=[Block("clk", "clock_bpm", {"bpm": 60})],
            wires=[],
        ),
        "con": Program(
            "counter_display",
            description="accum -> smooth -> viz_series — conscious counter",
            blocks=[
                Block("cnt", "accum", {"per_tick": 1}),
                Block("sm", "smooth", {"alpha": 0.1}),
                Block("v0", "viz_series"),
            ],
            wires=[
                Wire("cnt.acc", "sm.in"),
                Wire("sm.cv", "v0.in"),
            ],
        ),
        "bridge_map": [("clk.trig", "cnt.in")],
        "bridge_latency": 1,
        "use_h4": False,
    }


def _infinite_video_bicameral() -> dict:
    """iter34: H3 -> HostBridge.push_frame -> viz_video (conscious sink).

    Subconscious is just a clock_bpm@60 (the prompt-generation cadence).
    Conscious renders the pushed RGBA frames via viz_video.
    Bridge traffic is H3 frames (sub -> con) + the clock scalar (sub -> con).
    """
    return {
        "type": "bicameral",
        "sub": Program(
            "infinite_video_sub",
            description="clock_bpm@60 — H3 cadence",
            blocks=[Block("clk", "clock_bpm", {"bpm": 60})],
            wires=[],
        ),
        "con": Program(
            "infinite_video_con",
            description="viz_video — renders pushed RGBA frames",
            blocks=[Block("vv", "viz_video")],
            wires=[],
        ),
        "bridge_map": [("clk.trig", "vv.in")],
        "bridge_latency": 1,
        "use_h4": False,
    }


_BICAMERAL_REGISTRY: dict[str, callable] = {
    "bicameral_clock": _bicameral_clock,
    "infinite_video_bicameral": _infinite_video_bicameral,
}


def _infinite_slop_bicameral() -> dict:
    """iter 46: bicameral SlopLoop — Swarm H4 consensus on GPU1 -> viz_video on GPU0.

    The SlopLoop drives both the BicameralViewer (via feed_video_tick on the
    conscious engine's bus[vv.frame]) and the swarm H4 consensus on the
    subconscious side. Operator hits Start on the Slop bar to begin.

    Architecture (mirrors the SlopLoop pipeline):
      sub=clock_bpm@60 (H3 cadence) + swarm H4
      con=viz_video (renders the pushed RGBA frame)
    """
    return {
        "type": "bicameral",
        "sub": Program(
            "slop_sub",
            description="clock_bpm@60 — slop loop cadence + swarm H4 consensus",
            blocks=[Block("clk", "clock_bpm", {"bpm": 60})],
            wires=[],
        ),
        "con": Program(
            "slop_con",
            description="viz_video — renders slop-loop pushed frames",
            blocks=[Block("vv", "viz_video")],
            wires=[],
        ),
        "bridge_map": [("clk.trig", "vv.in")],
        "bridge_latency": 1,
        "use_h4": False,
    }


_BICAMERAL_REGISTRY["infinite_slop_bicameral"] = _infinite_slop_bicameral


def _infinite_video_export() -> dict:
    """iter35: VideoSynth -> InfiniteVideoLoop -> BicameralViewer -> viz_video + jfin_live_export.

    Pipeline:
      sub=clock_bpm@60 (cadence)
      con=viz_video + jfin_live_export (frames injected via bridge bus, no wires needed)
           -> JFinScheduler -> JFinExporter -> ffmpeg HLS
           -> Jellyfin Live TV -> HDHomeRun M3U -> LAN clients

    Frames are injected into the con engine's bus[vv.frame] key by the
    InfiniteVideoLoop. The jfin_live_export block reads bus[jle.frame]
    directly. No intra-program wires needed for frame traffic.
    """
    return {
        "type": "bicameral",
        "sub": Program(
            "video_export_sub",
            description="clock_bpm@60 — cadence for video export",
            blocks=[Block("clk", "clock_bpm", {"bpm": 60})],
            wires=[],
        ),
        "con": Program(
            "video_export_con",
            description="viz_video + jfin_live_export — tile wall + Jellyfin stream",
            blocks=[
                Block("vv", "viz_video"),
                Block("jle", "jfin_live_export",
                      {"scheduler_key": "atomic-01",
                       "width": 64, "height": 64}),
            ],
            wires=[],
        ),
        "bridge_map": [],  # frame traffic uses bridge.push_frame/pop_frame; clock not needed
        "bridge_latency": 1,
        "use_h4": False,
    }


_BICAMERAL_REGISTRY["infinite_video_export"] = _infinite_video_export


def _infinite_fasth3_bicameral() -> dict:
    """iter 36: FastH3 (4-step VSA GGUF) -> HostBridge -> viz_fasth3_video.

    Subconscious is a clock_bpm@60 (H3 / FastH3 frame cadence).
    Conscious renders the pushed RGBA frames via viz_fasth3_video.
    Bridge traffic is FastH3 frames (sub -> con); the clock scalar is
    not used (frame cadence is one-per-tick).

    This is the FastH3 twin of iter 34's infinite_video_bicameral: same
    wire contract, FastH3 metadata (steps=4, vsa=ON, Q5_K_M) on every
    frame. The InfiniteFastH3Loop populates the bus[vfh.frame] key and
    drives the conscious engine exactly like InfiniteVideoLoop does for
    base H3.
    """
    return {
        "type": "bicameral",
        "sub": Program(
            "infinite_fasth3_sub",
            description="clock_bpm@60 — FastH3 cadence",
            blocks=[Block("clk", "clock_bpm", {"bpm": 60})],
            wires=[],
        ),
        "con": Program(
            "infinite_fasth3_con",
            description="viz_fasth3_video — renders pushed FastH3 RGBA frames",
            blocks=[
                Block("vfh", "viz_fasth3_video",
                      {"capture": 1.0, "steps": 4, "vsa": 1.0,
                       "vsa_keep": 10, "quant": "Q5_K_M"}),
            ],
            wires=[],
        ),
        "bridge_map": [("clk.trig", "vfh.in")],
        "bridge_latency": 1,
        "use_h4": False,
    }


_BICAMERAL_REGISTRY["infinite_fasth3_bicameral"] = _infinite_fasth3_bicameral


def build_bicameral(name: str) -> dict | None:
    fn = _BICAMERAL_REGISTRY.get(name)
    if fn is None:
        return None
    return fn()


def all_bicameral_programs() -> list[str]:
    return sorted(_BICAMERAL_REGISTRY.keys())
