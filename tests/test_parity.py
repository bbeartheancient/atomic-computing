"""Parity baseline (build-order step 2): the Python engine must match
the node jsfx oracle on the 6 conformance patches of
fabric/tests/jsfx_conformance.js:165-217 (vectors re-pinned here).

Tolerance policy (pinned in ATOMIC-PC-STATE.md): bit-exact on
integer/affine paths (tol 0.0); 1e-9 where a body calls Math.*
(mdct_flux: V8 Math.sin/cos vs CPython math.sin/cos last-ulp seam).
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic import Engine, oracle  # noqa: E402

DT = 1 / 30

# (label, ticks, ui_taps, tol, patch) -- mirrors jsfx_conformance.js
PATCHES = [
    ("const->gain", 120, None, 0.0,
     {"modules": [
         {"id": "c1", "primitive": "const", "params": {"value": 5}},
         {"id": "g1", "primitive": "gain", "params": {"factor": 2}}],
      "wires": [{"from": "c1.cv", "to": "g1.in"}],
      "views": [{"module": "g1", "as": "series"}]}),
    ("smooth converges", 200, None, 0.0,
     {"modules": [
         {"id": "k", "primitive": "const", "params": {"value": 7}},
         {"id": "s1", "primitive": "smooth", "params": {"alpha": 0.5}}],
      "wires": [{"from": "k.cv", "to": "s1.in"}], "views": []}),
    ("mdct_flux finite", 150, None, 1e-9,
     {"modules": [
         {"id": "k2", "primitive": "const", "params": {"value": 3}},
         {"id": "f", "primitive": "mdct_flux", "params": {}}],
      "wires": [{"from": "k2.cv", "to": "f.in"}], "views": []}),
    # counter app: two taps -> edge counter counts 2 (tap@5/@30; the
    # consumer sees each pulse ONE TICK later -> 2 total)
    ("counter via taps", 60, [5, 30], 0.0,
     {"modules": [
         {"id": "ui", "primitive": "tap", "params": {}},
         {"id": "acc", "primitive": "accum", "params": {"per_tick": 1}}],
      "wires": [{"from": "ui.tap", "to": "acc.in"}], "views": []}),
    # Rack stackable inputs: two cables into one input SUM
    ("stacked inputs sum", 5, None, 0.0,
     {"modules": [
         {"id": "c1", "primitive": "const", "params": {"value": 5}},
         {"id": "c2", "primitive": "const", "params": {"value": 3}},
         {"id": "g", "primitive": "gain", "params": {"factor": 1}}],
      "wires": [{"from": "c1.cv", "to": "g.in"},
                {"from": "c2.cv", "to": "g.in"}], "views": []}),
    # fan-out: one output feeds two nodes freely
    ("fan-out", 5, None, 0.0,
     {"modules": [
         {"id": "c", "primitive": "const", "params": {"value": 5}},
         {"id": "g1", "primitive": "gain", "params": {"factor": 1}},
         {"id": "g2", "primitive": "gain", "params": {"factor": 1}}],
      "wires": [{"from": "c.cv", "to": "g1.in"},
                {"from": "c.cv", "to": "g2.in"}], "views": []}),
]


def _diff(js_final, py_final, tol):
    """JS-undefined == missing key (JSON drops undefined values)."""
    problems = []
    for k in sorted(set(js_final) | set(py_final)):
        jv = js_final.get(k)
        pv = py_final.get(k)
        if (jv is None) != (pv is None):
            problems.append("%s: js=%r py=%r" % (k, jv, pv))
            continue
        if jv is None:
            continue
        if isinstance(jv, str) or isinstance(pv, str):
            if jv != pv:
                problems.append("%s: %r != %r" % (k, jv, pv))
            continue
        jf, pf = float(jv), float(pv)
        if tol <= 0.0:
            if jf != pf:
                problems.append("%s: js=%r py=%r (bit mismatch)" % (k, jv, pv))
        elif abs(jf - pf) > tol:
            problems.append("%s: js=%r py=%r (diff %.3e > %g)" %
                             (k, jv, pv, abs(jf - pf), tol))
    return problems


@pytest.mark.parametrize("label,ticks,ui_taps,tol,patch", PATCHES,
                         ids=[p[0] for p in PATCHES])
def test_engine_matches_oracle(label, ticks, ui_taps, tol, patch):
    js_final, js_series = oracle.run(patch, ticks, dt=DT, ui_taps=ui_taps)
    eng = Engine(patch["modules"], patch["wires"],
                 views=patch.get("views") or [], dt=DT, ui_taps=ui_taps)
    res = eng.run(ticks)
    problems = _diff(js_final, res["final"], tol)
    assert not problems, "final bus diverges for %s:\n%s" % (
        label, "\n".join(problems))
    for key, js_arr in js_series.items():
        py_arr = res["series"].get(key, [])
        assert len(js_arr) == len(py_arr), \
            "%s: series[%s] len %d != %d" % (label, key, len(js_arr), len(py_arr))
        for i, (jv, pv) in enumerate(zip(js_arr, py_arr)):
            if jv is None:
                continue
            ok = float(jv) == float(pv) if tol <= 0.0 \
                else abs(float(jv) - float(pv)) <= tol
            assert ok, "%s: series[%s][%d] js=%r py=%r" % (
                label, key, i, jv, pv)


def test_counter_value_is_two():
    """The conformance's own check: acc.acc === 2 after taps @5/@30."""
    patch = PATCHES[3][4]
    eng = Engine(patch["modules"], patch["wires"], dt=DT, ui_taps=[5, 30])
    res = eng.run(60)
    assert res["final"]["acc.acc"] == 2
