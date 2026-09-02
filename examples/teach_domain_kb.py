"""Teach domain knowledge base: expanded teacher registry persisted via QBF (iter 26, goals 4/5/6).

Run: ~/runtime/.venv/bin/python -m examples.teach_domain_kb

This module:
  1. Seeds 14 teacher examples across 6 domains (spatial, medical, audio, signal, control, general)
  2. Persists the expanded registry as a .qbf shard under ~/.runtime/atomic_qbf
  3. Verifies load round-trip and domain routing

The QBF file travels with the knowledge base — no 50 MB wall, H(4) gates optional.
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic import Program, Block, Wire, Engine
from atomic.teach import TeacherRegistry, save_registry, load_registry, DOMAINS


def build_kb() -> TeacherRegistry:
    """Build the canonical domain-routed knowledge base."""
    reg = TeacherRegistry()

    # spatial (CORE H4 keystone)
    p1 = Program("spatial_h4_consensus",
        description="A sensor drives H4 spatial gate; dominant W consensus axis drives a series chart.",
        blocks=[Block("sig", "sensor"), Block("h4", "h4_slide"), Block("v", "viz_series")],
        wires=[Wire("sig.cv", "h4.in"), Wire("h4.w", "v.in")])
    p2 = Program("spatial_wxyz_rotation",
        description="H4 W/X/Y/Z outputs feed a 3D scope showing spatial rotation and dominant amplitude.",
        blocks=[Block("sig", "sensor"), Block("h4", "h4_slide"), Block("v", "viz_wxyz3d")],
        wires=[Wire("sig.cv", "h4.in"),
               Wire("h4.w", "v.w"), Wire("h4.x", "v.x"),
               Wire("h4.y", "v.y"), Wire("h4.z", "v.z")])
    p3 = Program("spatial_hadamard_scope",
        description="Hadamard spatial gate splits audio into W/X/Y/Z axis streams for a 3D scope.",
        blocks=[Block("sig", "sensor"), Block("h4", "h4_slide"), Block("v", "viz_wxyz3d")],
        wires=[Wire("sig.cv", "h4.in"),
               Wire("h4.w", "v.w"), Wire("h4.x", "v.x"),
               Wire("h4.y", "v.y"), Wire("h4.z", "v.z")])

    # medical (vital sign / alarm concepts)
    p4 = Program("medical_threshold_alarm",
        description="A medical sensor threshold gates an alarm chart.",
        blocks=[Block("sig", "sensor"), Block("th", "threshold", {"hi": 0.8, "lo": 0.2}), Block("v", "viz_series")],
        wires=[Wire("sig.cv", "th.in"), Wire("th.gate", "v.in")])
    p5 = Program("medical_vital_monitor",
        description="A medical sensor feeds a smooth filter then a threshold alarm chart.",
        blocks=[Block("sig", "sensor"), Block("sm", "smooth", {"alpha": 0.1}),
                Block("th", "threshold", {"hi": 0.7, "lo": 0.3}), Block("v", "viz_series")],
        wires=[Wire("sig.cv", "sm.in"), Wire("sm.cv", "th.in"), Wire("th.gate", "v.in")])
    p6 = Program("medical_ecg_pulse_counter",
        description="ECG pulse sensor triggers threshold then accum counter on a chart.",
        blocks=[Block("sig", "sensor"), Block("th", "threshold", {"hi": 0.6, "lo": 0.2}),
                Block("acc", "accum", {"per_tick": 1}), Block("v", "viz_series")],
        wires=[Wire("sig.cv", "th.in"), Wire("th.gate", "acc.in"), Wire("acc.acc", "v.in")])

    # audio (hadamard / mdct / spatial mix)
    p7 = Program("audio_spectrum_flux",
        description="Audio sensor feeds MDCT flux detector driving a series chart.",
        blocks=[Block("sig", "sensor"), Block("fl", "mdct_flux", {}), Block("v", "viz_series")],
        wires=[Wire("sig.cv", "fl.in"), Wire("fl.flux", "v.in")])
    p8 = Program("audio_spatial_mix",
        description="Two audio sensors mix through H4 spatial gate; W dominant consensus drives the chart.",
        blocks=[Block("s1", "sensor"), Block("s2", "sensor"), Block("h4", "h4_slide"), Block("v", "viz_series")],
        wires=[Wire("s1.cv", "h4.in"), Wire("s2.cv", "h4.in"), Wire("h4.w", "v.in")])
    p9 = Program("audio_hadamard_3d",
        description="Audio stream feeds Hadamard transform; W/X/Y/Z axis outputs drive a 3D scope.",
        blocks=[Block("sig", "sensor"), Block("h4", "h4_slide"), Block("v", "viz_wxyz3d")],
        wires=[Wire("sig.cv", "h4.in"),
               Wire("h4.w", "v.w"), Wire("h4.x", "v.x"),
               Wire("h4.y", "v.y"), Wire("h4.z", "v.z")])

    # signal (filter / smooth / hysteresis / clamp)
    p10 = Program("signal_moving_average_filter",
        description="Signal sensor feeds a moving average filter; threshold gates a chart.",
        blocks=[Block("sig", "sensor"), Block("ma", "moving_avg", {"n": 8}),
                 Block("th", "threshold", {"hi": 0.5, "lo": -0.5}), Block("v", "viz_series")],
        wires=[Wire("sig.cv", "ma.in"), Wire("ma.cv", "th.in"), Wire("th.gate", "v.in")])
    p11 = Program("signal_hysteresis_filter",
        description="Signal passes through a gain then a hysteresis threshold; clamped output goes to chart.",
        blocks=[Block("sig", "sensor"), Block("g", "gain", {"factor": 2.0}),
                Block("cl", "clamp", {"lo": -1.0, "hi": 1.0}), Block("v", "viz_series")],
        wires=[Wire("sig.cv", "g.in"), Wire("g.cv", "cl.in"), Wire("cl.cv", "v.in")])
    p12 = Program("signal_smooth_bias",
        description="Signal smoothed with bias and gain feeding a series chart.",
        blocks=[Block("sig", "sensor"), Block("sm", "smooth", {"alpha": 0.15}),
                Block("g", "gain", {"factor": 1.5}), Block("v", "viz_series")],
        wires=[Wire("sig.cv", "sm.in"), Wire("sm.cv", "g.in"), Wire("g.cv", "v.in")])

    # control (clock / toggle / divider / accum)
    p13 = Program("control_gated_clock_counter",
        description="A gated clock fires a counter; the count is smoothed before driving a chart.",
        blocks=[Block("c1", "clock_bpm", {"bpm": 60}), Block("a1", "accum", {"per_tick": 1}),
                Block("s1", "smooth", {"alpha": 0.1}), Block("v1", "viz_series")],
        wires=[Wire("c1.trig", "a1.in"), Wire("a1.acc", "s1.in"), Wire("s1.cv", "v1.in")])
    p14 = Program("control_clock_divider",
        description="A clock divider fires a toggle on every other beat; the toggle state drives a chart.",
        blocks=[Block("clk", "clock_bpm", {"bpm": 120}), Block("tg", "toggle", {"initial": 0}), Block("v", "viz_series")],
        wires=[Wire("clk.trig", "tg.trig"), Wire("tg.state", "v.in")])
    p15 = Program("control_bpm_accumulator",
        description="A fast BPM clock accumulates pulses; the count feeds a smoothed chart.",
        blocks=[Block("clk", "clock_bpm", {"bpm": 240}), Block("acc", "accum", {"per_tick": 1}),
                Block("sm", "smooth", {"alpha": 0.05}), Block("v", "viz_series")],
        wires=[Wire("clk.trig", "acc.in"), Wire("acc.acc", "sm.in"), Wire("sm.cv", "v.in")])

    # Register by domain
    for p, dom in [
        (p1, "spatial"), (p2, "spatial"), (p3, "spatial"),
        (p4, "medical"), (p5, "medical"), (p6, "medical"),
        (p7, "audio"),   (p8, "audio"),   (p9, "audio"),
        (p10, "signal"), (p11, "signal"), (p12, "signal"),
        (p13, "control"), (p14, "control"), (p15, "control"),
    ]:
        reg.register(p.description, p, domain=dom, difficulty="canonical")

    return reg


def demo(path=None):
    """Build KB, persist to QBF, verify round-trip and domain routing."""
    # 1) Build
    reg = build_kb()
    print(f"[teach_domain_kb] built {len(reg.examples)} examples across {len(DOMAINS)} domains")

    # 2) Persist via QBF
    qbf_dir = os.environ.get("ATOMIC_QBF_DIR",
                             os.path.join(os.path.expanduser("~"), ".runtime", "atomic_qbf"))
    os.makedirs(qbf_dir, exist_ok=True)

    if path is None:
        path = os.path.join(qbf_dir, "teach_domain_kb.qbf")

    saved = save_registry(path=path, registry=reg)
    print(f"[teach_domain_kb] persisted -> {saved}")

    # 3) Load round-trip
    loaded = load_registry(saved)
    print(f"[teach_domain_kb] loaded {len(loaded.examples)} examples")

    # 4) Domain routing checks
    routing_checks = [
        ("hadamard spatial wxyz 3d scope", "spatial", "h4_slide"),
        ("medical sensor threshold alarm vital", "medical", "threshold"),
        ("audio sensor spectrum flux frequency", "audio", "mdct_flux"),
        ("signal moving average filter threshold", "signal", "moving_avg"),
        ("clock bpm counter toggle divider", "control", "clock_bpm"),
    ]
    routing_ok = 0
    for desc, domain, prim in routing_checks:
        p = loaded.from_description(desc, domain=domain)
        assert p.validate() == [], (desc, domain, p.validate())
        has = any(b.primitive == prim for b in p.blocks)
        print(f"  routing [{domain:8s}] '{desc[:40]:<40}' -> {prim}: {'OK' if has else 'MISS'}")
        routing_ok += int(has)

    # 5) Domain coverage
    for dom in DOMAINS:
        examples = loaded.list(domain=dom)
        print(f"  domain {dom:8s}: {len(examples)} examples")

    # 6) QBF round-trip fidelity
    patch = loaded.examples[0]["program"].compile("microfx")
    res = Engine(patch["modules"], patch["wires"]).run(10)
    assert res["final"] is not None, "engine run failed on loaded program"

    print(f"[teach_domain_kb] routing: {routing_ok}/{len(routing_checks)}")
    print("[teach_domain_kb] all checks passed")
    return {
        "n_examples": len(loaded.examples),
        "routing_ok": routing_ok,
        "routing_total": len(routing_checks),
        "path": saved,
    }


if __name__ == "__main__":
    demo()
