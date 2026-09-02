"""iter-6 parity (oracle MODE 2): the resident node driver.

STEP 1 -- driver loop == evaluatePatch. For each of the 6 static
conformance patches (re-pinned in test_parity), the resident driver's
re-implementation of the pinned per-tick loop (no live feeds) must
reproduce evaluatePatch's {final, series}. Both sides run under node
with the same V8 Math, so the comparison uses the patch's own
tolerance (0.0, or 1e-9 for the mdct_flux V8-vs-CPython seam --
harmless here). This pins the re-implementation as a faithful twin
BEFORE any live hook is used.

STEP 2 -- driver (live) == engine (live). The same resident driver is
fed a per-tick feed (live params, live taps) and must match the Python
Engine fed with the SAME feed. Q1 (iter 2) is the enabler: the patch
proxy reads m.params AT EVAL TIME, so a between-tick mutation is seen
by the next tick -- something the batch evaluatePatch cannot express
(per-run config is static). The driver mutates m.params; the engine
merges into node.params; the IR keeps keys lowercase so the two agree.

Live seams exercised (one primary per patch):
  const      live 'value'     a SOURCE param re-timed mid-run
  gain       live 'factor'    the affine gain, three steps
  sine_lfo   live 'rate_hz'   a Math.sin body -> 1e-9 ulp tolerance
  smooth      live 'alpha'    a stateful convergence coefficient
  threshold  live 'hi'        the hysteresis bound flips state 1->0->1
  clock_bpm  live 'bpm'       the HOST-source accumulator re-timed
  counter    live taps         feed.taps overrides the (empty) static taps
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from atomic import Engine, LiveOracle
from atomic.oracle import run as oracle_run
from test_parity import PATCHES, _diff

DT = 1 / 30


# ---------------------------------------------------------------------------
# STEP 1: driver loop == evaluatePatch (the re-implementation is faithful)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,ticks,ui_taps,tol,patch", PATCHES,
                         ids=[p[0] for p in PATCHES])
def test_driver_loop_matches_evaluatepatch(label, ticks, ui_taps, tol, patch):
    with LiveOracle() as live:
        drv_final, drv_series = live.run(patch, ticks, dt=DT, ui_taps=ui_taps)
    ref_final, ref_series = oracle_run(patch, ticks, dt=DT, ui_taps=ui_taps)
    problems = _diff(ref_final, drv_final, tol)
    assert not problems, \
        "driver loop diverges from evaluatePatch for %s:\n%s" % (
            label, "\n".join(problems))
    for key, ref_arr in ref_series.items():
        drv_arr = drv_series.get(key, [])
        assert len(ref_arr) == len(drv_arr), \
            "%s: series[%s] len %d != %d" % (label, key, len(ref_arr), len(drv_arr))
        for i, (rv, dv) in enumerate(zip(ref_arr, drv_arr)):
            if rv is None:
                continue
            ok = float(rv) == float(dv) if tol <= 0.0 else \
                abs(float(rv) - float(dv)) <= tol
            assert ok, "%s: series[%s][%d] js=%r drv=%r" % (
                label, key, i, rv, dv)


# ---------------------------------------------------------------------------
# STEP 2: driver (live feeds) == engine (same feeds)
# ---------------------------------------------------------------------------

LIVE = [
    # (label, ticks, ui_taps, patch, feeds, tol)
    ("const live value", 20, None,
     {"modules": [
         {"id": "c1", "primitive": "const", "params": {"value": 5}},
         {"id": "g1", "primitive": "gain", "params": {"factor": 2}}],
      "wires": [{"from": "c1.cv", "to": "g1.in"}], "views": []},
     {5: {"params": {"c1": {"value": 7}}},
      12: {"params": {"c1": {"value": 3}}}}, 0.0),
    ("gain live factor", 20, None,
     {"modules": [
         {"id": "c1", "primitive": "const", "params": {"value": 5}},
         {"id": "g1", "primitive": "gain", "params": {"factor": 1}}],
      "wires": [{"from": "c1.cv", "to": "g1.in"}], "views": []},
     {5: {"params": {"g1": {"factor": 3}}},
      10: {"params": {"g1": {"factor": 4}}},
      15: {"params": {"g1": {"factor": 2}}}}, 0.0),
    ("sine_lfo live rate", 100, None,
     {"modules": [
         {"id": "l", "primitive": "sine_lfo",
          "params": {"offset": 0.5, "amp": 1.0, "rate_hz": 1.0}}],
      "wires": [], "views": []},
     {10: {"params": {"l": {"rate_hz": 2.0}}},
      40: {"params": {"l": {"rate_hz": 0.5}}}}, 1e-9),
    ("smooth live alpha", 60, None,
     {"modules": [
         {"id": "k", "primitive": "const", "params": {"value": 7}},
         {"id": "s1", "primitive": "smooth", "params": {"alpha": 0.5}}],
      "wires": [{"from": "k.cv", "to": "s1.in"}], "views": []},
     {10: {"params": {"s1": {"alpha": 0.9}}},
      30: {"params": {"s1": {"alpha": 0.1}}}}, 0.0),
    ("threshold live hi", 20, None,
     {"modules": [
         {"id": "k", "primitive": "const", "params": {"value": 4}},
         {"id": "th", "primitive": "threshold",
          "params": {"hi": 3, "lo": 6}}],
      "wires": [{"from": "k.cv", "to": "th.in"}], "views": []},
     {10: {"params": {"th": {"hi": 5}}},
      16: {"params": {"th": {"hi": 2}}}}, 0.0),
    ("clock_bpm live bpm", 60, None,
     {"modules": [
         {"id": "clk", "primitive": "clock_bpm", "params": {"bpm": 60}},
         {"id": "acc", "primitive": "accum", "params": {"per_tick": 1}}],
      "wires": [{"from": "clk.trig", "to": "acc.in"}], "views": []},
     {10: {"params": {"clk": {"bpm": 120}}}}, 0.0),
    ("counter live taps", 60, None,
     {"modules": [
         {"id": "ui", "primitive": "tap", "params": {}},
         {"id": "acc", "primitive": "accum", "params": {"per_tick": 1}}],
      "wires": [{"from": "ui.tap", "to": "acc.in"}], "views": []},
     {5: {"taps": [5]}, 30: {"taps": [30]}}, 0.0),
]


@pytest.mark.parametrize("label,ticks,ui_taps,patch,feeds,tol", LIVE,
                         ids=[l[0] for l in LIVE])
def test_live_driver_matches_engine(label, ticks, ui_taps, patch, feeds, tol):
    with LiveOracle() as live:
        drv_final, drv_series = live.run(
            patch, ticks, dt=DT, ui_taps=ui_taps, feeds=feeds)
    eng = Engine(patch["modules"], patch["wires"],
                 views=patch.get("views") or [], dt=DT,
                 ui_taps=ui_taps, feeds=feeds)
    res = eng.run(ticks)
    problems = _diff(drv_final, res["final"], tol)
    assert not problems, \
        "live driver/engine diverge for %s:\n%s" % (
            label, "\n".join(problems))
    for key, drv_arr in drv_series.items():
        eng_arr = res["series"].get(key, [])
        assert len(drv_arr) == len(eng_arr), \
            "%s: series[%s] len %d != %d" % (label, key, len(drv_arr), len(eng_arr))
        for i, (dv, ev) in enumerate(zip(drv_arr, eng_arr)):
            if dv is None:
                continue
            ok = float(dv) == float(ev) if tol <= 0.0 else \
                abs(float(dv) - float(ev)) <= tol
            assert ok, "%s: series[%s][%d] drv=%r eng=%r" % (
                label, key, i, dv, ev)


def test_counter_live_taps_counts_two():
    """The conformance's own check, now driven by LIVE feeds: taps at
    tick 5 and 30 (delivered via feed.taps, not the static ui_taps)
    still count exactly 2 by the end -- and the driver agrees."""
    label, ticks, ui_taps, patch, feeds, tol = LIVE[-1]
    eng = Engine(patch["modules"], patch["wires"], views=[],
                 dt=DT, ui_taps=ui_taps, feeds=feeds)
    res = eng.run(ticks)
    assert res["final"]["acc.acc"] == 2
    with LiveOracle() as live:
        drv_final, _ = live.run(patch, ticks, dt=DT,
                                 ui_taps=ui_taps, feeds=feeds)
    assert drv_final["acc.acc"] == 2
