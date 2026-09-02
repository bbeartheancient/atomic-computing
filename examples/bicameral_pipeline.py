"""Bicameral pipeline demo: two engines wired through HostBridge (goal B).

The subconscious (GPU1) runs a clock_bpm@60; the conscious (GPU0)
runs accum->smooth->viz_series. The HostBridge sits between them:
subconscious ticks -> sample sub bus -> push to bridge -> pop from
bridge -> inject into conscious node inputs -> conscious ticks.

Demonstrates:
  - BicameralPipeline: sub_program + con_program wired through HostBridge
  - Bridge depth over time: the queue accumulates beats between push/pop
  - Subconscious clock fires at 60 bpm (1 Hz): beats at tick 30, 60, 90...
  - Conscious accum counts beats with 1-tick input latency
  - Final accum value = 2 (two beats processed in 90 ticks)
  - Bridge depth peaks at the beat arrival

~/runtime/.venv/bin/python -m examples.bicameral_pipeline
"""

from __future__ import annotations

from atomic.bridge import BicameralPipeline, HostBridge
from atomic.program import Block, Program, Wire


def sub_program() -> Program:
    return Program(
        "clock_source",
        description="clock_bpm@60 — the subconscious clock",
        blocks=[
            Block("clk", "clock_bpm", {"bpm": 60}),
        ],
        wires=[],
    )


def con_program() -> Program:
    return Program(
        "counter_display",
        description="accum -> smooth -> viz_series — the conscious counter",
        blocks=[
            Block("cnt", "accum", {"per_tick": 1}),
            Block("sm", "smooth", {"alpha": 0.1}),
            Block("v0", "viz_series"),
        ],
        wires=[
            Wire("cnt.acc", "sm.in"),
            Wire("sm.cv", "v0.in"),
        ],
    )


def demo(ticks: int = 90, dt: float = 1.0 / 30.0):
    sub = sub_program()
    con = con_program()

    bridge_map = [("clk.trig", "cnt.in")]

    pipe = BicameralPipeline(
        sub, con,
        bridge_map=bridge_map,
        bridge_latency=1,
        use_h4=False,
        dt=dt,
    )

    bridge_depths = []
    for _ in range(ticks):
        pipe.tick()
        depth = pipe.bridge.depth()
        bridge_depths.append(depth)

    result = pipe.run(0)

    sub_bus = result.sub["bus"]
    con_bus = result.con["bus"]
    max_depth = max(bridge_depths)
    beats_detected = sum(1 for d in bridge_depths if d > 0)

    acc = con_bus.get("cnt.acc")
    assert acc is not None, "accum output missing from final bus"
    assert acc == 2.0, f"expected accum=2.0 after 90 ticks (two beats at 60bpm), got {acc}"

    sm = con_bus.get("sm.cv")
    assert sm is not None, "smooth output missing from final bus"
    assert 0.0 < sm < acc, f"smooth {sm} not in (0, {acc})"

    print("[bicameral pipeline] ok")
    print("  sub program:       %s" % sub.name)
    print("  con program:       %s" % con.name)
    print("  bridge map:        %s" % bridge_map)
    print("  bridge latency:    %d tick" % pipe.bridge.latency)
    print("  max bridge depth:  %d" % max_depth)
    print("  beats queued:      %d" % beats_detected)
    print("  final accum:       %.1f" % acc)
    print("  final smooth:      %.4f" % sm)
    print("  bridge depths:     %s" % str(bridge_depths[:15]) + ("..." if len(bridge_depths) > 15 else ""))

    return {
        "sub_program": sub,
        "con_program": con,
        "bridge_map": bridge_map,
        "result": result,
        "bridge_depths": bridge_depths,
        "max_depth": max_depth,
        "beats_detected": beats_detected,
        "accum": acc,
        "smooth": sm,
    }


if __name__ == "__main__":
    demo()
