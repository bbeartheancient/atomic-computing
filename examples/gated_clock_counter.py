"""Gated clock counter: from_description + engine run (goal 4/5 demo).

The canonical "LLM assembles apps by matching function paths through
gates" demo: clock_bpm fires an accum counter; the count feeds smooth;
smooth feeds a viz_series chart.  The description matches the seeded
"gated_clock_counter" registry entry in the "control" domain, so
from_description() returns it directly (no LLM needed).

Verifies:
  - from_description matches the control-domain example.
  - Clock fires at 60 bpm = 1 Hz: at dt=1/30 s, first beat at tick 29.
  - accum increments each beat.
  - The final g1.cv is the accumulated count.
  - The compiled microfx patch runs and produces non-trivial output.

  ~/runtime/.venv/bin/python -m examples.gated_clock_counter
"""

from __future__ import annotations

from atomic.engine import Engine
from atomic.teach import from_description, match, domain_vocab


def demo(ticks: int = 90, dt: float = 1.0 / 30.0):
    # 1) Domain routing: control domain keyword weights.
    vocab = domain_vocab("control")
    assert vocab.get("clock", 0) > 0, "control vocab missing clock"
    assert vocab.get("accum", 0) > 0, "control vocab missing accum"

    # 2) from_description picks up the seeded control-domain example.
    desc = "A gated clock fires a counter; the count is smoothed before driving a chart."
    prog = from_description(desc, domain="control")
    assert prog is not None, "from_description returned None"
    assert prog.name == "gated_clock_counter", "wrong program matched: %s" % prog.name

    # 3) match() also finds it.
    hit = match(desc, domain="control")
    assert hit is not None, "match returned None"
    assert hit.name == "gated_clock_counter"

    # 4) Compile and run the matched program.  Explicit views so the
    #    engine populates series arrays for the visualizers.
    patch = prog.compile("microfx")
    assert len(patch["modules"]) == 4, len(patch["modules"])
    assert len(patch["wires"]) == 3, len(patch["wires"])
    views = [{"module": "a1", "as": "series", "output": "acc"},
             {"module": "s1", "as": "series", "output": "cv"},
             {"module": "v1", "as": "series", "output": "cv"}]
    eng = Engine(patch["modules"], patch["wires"], views=views, dt=dt)
    result = eng.run(ticks)
    bus = result["bus"]

    # 5) clock_bpm at 60 bpm fires at 1 Hz: dt=1/30 -> beat at tick 30, 60.
    #    accum has 1-tick input latency, so a beat at tick N is observed
    #    at tick N+1. Over 90 ticks both beats are processed: acc=2.
    acc = bus.get("a1.acc")
    assert acc is not None, "accum output missing from final bus"
    assert acc == 2.0, f"expected accum=2.0 after 90 ticks (two beats at 60bpm), got {acc}"

    # 6) smooth alpha=0.1 converges toward the accum; s1.cv is smooth output.
    s1 = bus.get("s1.cv")
    assert s1 is not None, "smooth output missing from final bus"
    # smooth lags the accum: it should be > 0 and < 2 after 90 ticks
    assert 0.0 < s1 < acc, f"smooth {s1} not in (0, {acc})"

    print("[gated clock counter] ok")
    print("  description: %s" % desc)
    print("  matched:     %s" % prog.name)
    print("  blocks:      %s" % [b.id for b in prog.blocks])
    print("  wires:       %s" % ["%s->%s" % (w.src, w.dst) for w in prog.wires])
    print("  final:       a1.acc=%s  s1.cv=%.4f" % (acc, s1))
    print("  domain:      control")
    print("  control vocab: %s" % {k: v for k, v in vocab.items()
                                   if v > 0})

    return {"program": prog, "patch": patch, "final": bus,
            "accum": acc, "smooth": s1}


if __name__ == "__main__":
    demo()
