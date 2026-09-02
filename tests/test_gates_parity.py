"""iter-4 parity: the gate catalog vs the node oracle (MODE 1).

Logic (10 gate_* macros): two 20-tick passes, per patch 1 + 10*(2 toggles
+ 1 gate) = 31 modules. Both toggles ride the shared ui tap (taps [5,15])
and feed the gate with ONE tick of wire latency (gate input at tick t =
toggle state at t-1):
  pass A (ia=ib=0): (a,b) = (0,0)@t0-6, (1,1)@t7-16, (0,0)@t17-19
  pass B (ia=1,ib=0): (a,b) = (0,0)@t0, (1,0)@t1-6, (0,1)@t7-16, (1,0)@t17-19
The absolute truth-table series below pin the ORACLE (so we are not just
matching two implementations that are wrong the same way).

Quantum (bipolar +/-1): 12-tick run, tap [5]. The bipolar feed is a
THREE-HOP chain (tg.state -> g2 gain -> b1 bias -> gate.q); the t6 state
flip lands at g2@t7, b1@t8, gate@t9: q(t) = 0@t0, -1@t1-8, +1@t9-11.

alogic: six families in parallel, each on base + factor*state (2 hops:
const/gain latch one tick each): v = base@t0-7, base+factor@t8-11.
fam4 (CML) is pinned to always output 1 (the faithful
0.5+d-in-[0.25,0.75] quantize quirk).

h4_slide is harness-only (NOT in the oracle fixture) -> pinned inside the
engine against hoa64.sylvester(4), plus the documented label swap vs
fabric hadamard4 (y<->x, z<->x... w shared).

Tolerance 0.0 (bit-exact) everywhere: gate bodies are integer/affine IEEE
and the JSON bridge round-trips doubles losslessly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from atomic.engine import Engine
from atomic.gates import ATOMS, Atom
from atomic.oracle import run as oracle_run

DT = 1.0 / 30.0
C = 0.7071067811865476  # hadamard_gate's exact literal (shared with EEL2)

GATES10 = ["gate_buffer", "gate_not", "gate_and", "gate_or", "gate_nand",
           "gate_nor", "gate_xor", "gate_xnor", "gate_imply", "gate_nimply"]
SINGLE_IN = {"gate_buffer", "gate_not"}


def _diff(js, py, tol):
    """Bus-dict diff. A key dropped by JSON (JS undefined) == Python None."""
    problems = []
    for key in sorted(set(js) | set(py)):
        jv, pv = js.get(key), py.get(key)
        if jv is None and pv is None:
            continue
        if jv is None:
            problems.append("%s: oracle undefined, twin %r" % (key, pv))
            continue
        if pv is None:
            problems.append("%s: oracle %r, twin undefined" % (key, jv))
            continue
        if isinstance(jv, str) or isinstance(pv, str):
            if jv != pv:
                problems.append("%s: %r vs %r" % (key, jv, pv))
        elif abs(float(jv) - float(pv)) > tol:
            problems.append("%s: %r vs %r" % (key, jv, pv))
    return problems


def _check_series(label, js_series, py_series, keys, tol=0.0):
    problems = []
    for key in keys:
        js = js_series.get(key) or []
        py = py_series.get(key) or []
        if len(js) != len(py):
            problems.append("%s %s: len %d vs %d" % (label, key, len(js), len(py)))
            continue
        for i, (a, b) in enumerate(zip(js, py)):
            if abs(float(a) - float(b)) > tol:
                problems.append("%s %s[%d]: %r vs %r" % (label, key, i, a, b))
    return problems


# ------------------------------------------------------------------ LOGIC --

def _logic_patch(ia, ib):
    modules = [{"id": "ui", "primitive": "tap", "params": {}}]
    wires = []
    views = []
    for g in GATES10:
        modules += [{"id": g + "_ta", "primitive": "toggle",
                     "params": {"initial": ia}},
                    {"id": g + "_tb", "primitive": "toggle",
                     "params": {"initial": ib}},
                    {"id": g, "primitive": g, "params": {}}]
        wires.append({"from": "ui.tap", "to": g + "_ta.trig"})
        wires.append({"from": "ui.tap", "to": g + "_tb.trig"})
        if g in SINGLE_IN:
            wires.append({"from": g + "_ta.state", "to": g + ".in"})
        else:
            wires.append({"from": g + "_ta.state", "to": g + ".a"})
            wires.append({"from": g + "_tb.state", "to": g + ".b"})
        views.append({"module": g, "as": "series", "output": "q"})
    return {"modules": modules, "wires": wires, "views": views}


# (a,b) phases over 20 ticks: pass A (0,0)/(1,1)/(0,0); pass B
# (0,0)/(1,0)/(0,1)/(1,0). Absolute outputs per gate:
PASS_A = {
    "gate_buffer": [0] * 7 + [1] * 10 + [0] * 3,
    "gate_not":    [1] * 7 + [0] * 10 + [1] * 3,
    "gate_and":    [0] * 7 + [1] * 10 + [0] * 3,
    "gate_or":     [0] * 7 + [1] * 10 + [0] * 3,
    "gate_nand":   [1] * 7 + [0] * 10 + [1] * 3,
    "gate_nor":    [1] * 7 + [0] * 10 + [1] * 3,
    "gate_xor":    [0] * 20,
    "gate_xnor":   [1] * 20,
    "gate_imply":  [1] * 20,
    "gate_nimply": [0] * 20,
}
PASS_B = {
    "gate_buffer": [0] + [1] * 6 + [0] * 10 + [1] * 3,
    "gate_not":    [1] + [0] * 6 + [1] * 10 + [0] * 3,
    "gate_and":    [0] * 20,
    "gate_or":     [0] + [1] * 19,
    "gate_nand":   [1] * 20,
    "gate_nor":    [1] + [0] * 19,
    "gate_xor":    [0] + [1] * 19,
    "gate_xnor":   [1] + [0] * 19,
    "gate_imply":  [1] + [0] * 6 + [1] * 10 + [0] * 3,
    "gate_nimply": [0] + [1] * 6 + [0] * 10 + [1] * 3,
}


def _run_pair(patch, ticks, taps):
    js_final, js_series = oracle_run(patch, ticks, dt=DT, ui_taps=taps)
    res = Engine(patch["modules"], patch["wires"], views=patch["views"],
                 dt=DT, ui_taps=taps).run(ticks)
    return js_final, js_series, res


def test_logic_pass_A():
    js_final, js_series, res = _run_pair(_logic_patch(0, 0), 20, [5, 15])
    assert not _diff(js_final, res["final"], 0.0)
    assert not _check_series("logicA", js_series, res["series"],
                              ["%s.q" % g for g in GATES10])
    for g in GATES10:
        assert res["series"]["%s.q" % g] == PASS_A[g], g


def test_logic_pass_B():
    js_final, js_series, res = _run_pair(_logic_patch(1, 0), 20, [5, 15])
    assert not _diff(js_final, res["final"], 0.0)
    assert not _check_series("logicB", js_series, res["series"],
                              ["%s.q" % g for g in GATES10])
    for g in GATES10:
        assert res["series"]["%s.q" % g] == PASS_B[g], g


# ---------------------------------------------------------------- QUANTUM --

def _quantum_patch():
    modules = [
        {"id": "ui", "primitive": "tap", "params": {}},
        {"id": "tg", "primitive": "toggle", "params": {"initial": 0}},
        {"id": "g2", "primitive": "gain", "params": {"factor": 2}},
        {"id": "b1", "primitive": "bias", "params": {"add": -1}},
        {"id": "one", "primitive": "const", "params": {"value": 1}},
    ]
    for mid, prim in [("p_x", "pauli_x"), ("p_y", "pauli_y"),
                       ("p_z", "pauli_z"), ("hg", "hadamard_gate"),
                       ("ph_s", "phase_s"), ("ph_t", "phase_t")]:
        modules.append({"id": mid, "primitive": prim, "params": {}})
    modules += [{"id": "cn", "primitive": "cnot", "params": {}},
                {"id": "sw", "primitive": "swap_gate", "params": {}},
                {"id": "tf", "primitive": "toffoli", "params": {}}]
    wires = [
        {"from": "ui.tap", "to": "tg.trig"},
        {"from": "tg.state", "to": "g2.in"},
        {"from": "g2.cv", "to": "b1.in"},
        {"from": "b1.cv", "to": "p_x.q"},
        {"from": "b1.cv", "to": "p_y.q"},
        {"from": "b1.cv", "to": "p_z.q"},
        {"from": "b1.cv", "to": "hg.q"},
        {"from": "b1.cv", "to": "ph_s.q"},
        {"from": "b1.cv", "to": "ph_t.q"},
        {"from": "b1.cv", "to": "cn.c"},
        {"from": "b1.cv", "to": "sw.a"},
        {"from": "b1.cv", "to": "tf.c1"},
        {"from": "one.cv", "to": "cn.t"},
        {"from": "one.cv", "to": "sw.b"},
        {"from": "one.cv", "to": "tf.c2"},
        {"from": "one.cv", "to": "tf.t"},
    ]
    views = [{"module": m, "as": "series", "output": "out"}
             for m in ("p_x", "p_y", "p_z", "hg", "ph_s", "ph_t")]
    views += [{"module": "cn", "as": "series", "output": "out"},
              {"module": "cn", "as": "series", "output": "c_out"},
              {"module": "sw", "as": "series", "output": "a_out"},
              {"module": "sw", "as": "series", "output": "b_out"},
              {"module": "tf", "as": "series", "output": "out"},
              {"module": "tf", "as": "series", "output": "c_out"}]
    return {"modules": modules, "wires": wires, "views": views}


# q(t) = 0@t0, -1@t1-8, +1@t9-11  (3-hop feed latency)
QPOS = [0.0] + [1.0] * 8 + [-1.0] * 3
QNEG = [0.0] + [-1.0] * 8 + [1.0] * 3
QC = [0.0] + [-C] * 8 + [C] * 3
QUANTUM_EXPECT = {
    "p_x.out": QPOS, "p_y.out": QPOS, "p_z.out": QPOS,
    "hg.out": QC,
    "ph_s.out": QNEG, "ph_t.out": QNEG,
    "cn.out": QPOS, "cn.c_out": QNEG,
    "sw.a_out": [0.0] + [1.0] * 11, "sw.b_out": QNEG,
    "tf.out": QPOS, "tf.c_out": QNEG,
}


def test_quantum_parity():
    patch = _quantum_patch()
    js_final, js_series, res = _run_pair(patch, 12, [5])
    assert not _diff(js_final, res["final"], 0.0)
    assert not _check_series("quantum", js_series, res["series"],
                              list(QUANTUM_EXPECT))
    for key, expect in QUANTUM_EXPECT.items():
        got = res["series"][key]
        assert len(got) == len(expect), (key, len(got), len(expect))
        for i, (g, e) in enumerate(zip(got, expect)):
            assert abs(float(g) - float(e)) <= 0.0, (key, i, g, e)


# ----------------------------------------------------------------- ALOGIC --

FAMS = [(0, 0.5, 0.2), (1, 0.3, 0.4), (2, 0.3, 0.5), (3, 0.3, 0.4),
        (4, 0.3, 0.4), (5, 0.3, 0.4)]


def _alogic_patch():
    modules = [{"id": "ui", "primitive": "tap", "params": {}},
               {"id": "tg", "primitive": "toggle", "params": {"initial": 0}}]
    wires = [{"from": "ui.tap", "to": "tg.trig"}]
    views = []
    for i, (fam, base, fac) in enumerate(FAMS):
        params = {"family": fam}
        if fam == 5:
            params.update({"cell_b": 1, "cell_c": 0})
        modules += [{"id": "b%d" % i, "primitive": "const",
                     "params": {"value": base}},
                    {"id": "k%d" % i, "primitive": "gain",
                     "params": {"factor": fac}},
                    {"id": "g%d" % i, "primitive": "alogic",
                     "params": params}]
        wires += [{"from": "b%d.cv" % i, "to": "g%d.in" % i},
                  {"from": "tg.state", "to": "k%d.in" % i},
                  {"from": "k%d.cv" % i, "to": "g%d.in" % i}]
        views.append({"module": "g%d" % i, "as": "series", "output": "q"})
    return {"modules": modules, "wires": wires, "views": views}


ALOGIC_EXPECT = {
    0: [0.0] * 8 + [1.0] * 4,
    1: [0.0] * 8 + [1.0] * 4,
    2: [0.0] * 8 + [1.0] * 4,
    3: [0.0] * 8 + [1.0] * 4,
    4: [1.0] * 12,   # CML: 0.5+d always truthy -> pinned quirk
    5: [0.0] * 8 + [1.0] * 4,
}


def test_alogic_parity():
    patch = _alogic_patch()
    js_final, js_series, res = _run_pair(patch, 12, [5])
    assert not _diff(js_final, res["final"], 0.0)
    keys = ["g%d.q" % i for i in range(len(FAMS))]
    assert not _check_series("alogic", js_series, res["series"], keys)
    for i, (fam, _, _) in enumerate(FAMS):
        assert res["series"]["g%d.q" % i] == ALOGIC_EXPECT[fam], fam


# --------------------------------------------------------------- H4 SLIDE --

def _script_atom(seq):
    def host(node, t, dt, bus):
        bus.set(node.id + ".cv", float(seq[t]) if t < len(seq) else 0.0)
    return Atom("script", "scripted source", "source", {}, [], ["cv"], "",
                 host=host)


def _h4_run(seq, ticks, extra=(), extra_wires=()):
    atoms = dict(ATOMS)
    atoms["script"] = _script_atom(list(seq))
    modules = [{"id": "src", "primitive": "script", "params": {}},
               {"id": "h", "primitive": "h4_slide", "params": {}}]
    wires = [{"from": "src.cv", "to": "h.in"}]
    modules += list(extra)
    wires += list(extra_wires)
    return Engine(modules, wires, atoms=atoms).run(ticks)


def test_h4_slide_rows_match_hoa64():
    try:
        import hoa64
        import numpy as np
    except ImportError:
        pytest.skip("hoa64 not importable (run pytest from $HOME)")
    H = np.array(hoa64.sylvester(4), dtype=float)
    outs = []
    for i in range(4):
        seq = [0.0, 0.0, 0.0, 0.0]
        seq[i] = 1.0
        f = _h4_run(seq, 5)["final"]
        outs.append([f["h.w"], f["h.z"], f["h.y"], f["h.x"]])
    got = np.array(outs[::-1])  # window newest-first: e_i lands in row 3-i
    assert (got == H).all(), "h4_slide rows != hoa64.sylvester(4)"


def test_h4_slide_orthogonal_and_w_dominant():
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not importable")
    rows = []
    for i in range(4):
        seq = [0.0, 0.0, 0.0, 0.0]
        seq[i] = 1.0
        f = _h4_run(seq, 5)["final"]
        rows.append([f["h.w"], f["h.z"], f["h.y"], f["h.x"]])
    H = np.array(rows)[::-1]  # back to the sylvester row order
    assert (H @ H.T == 4 * np.eye(4)).all()
    # constant stream: all the energy lands in W (the consensus row)
    f = _h4_run([2.0] * 6, 6)["final"]
    assert f["h.w"] == 8.0
    assert f["h.z"] == 0.0 and f["h.y"] == 0.0 and f["h.x"] == 0.0


def test_h4_slide_vs_fabric_hadamard4_label_swap():
    # documented delta: fabric hadamard4 emits (w, y, x, z) with the y/x
    # patterns SWAPPED vs CORE. h4_slide restores the CORE row order, so:
    #   h4_slide.z == hadamard4.y ; h4_slide.y == hadamard4.x
    #   h4_slide.x == hadamard4.z  (w identical)
    seq = [1.0, 2.0, 3.0, 4.0, 5.0]
    atoms = dict(ATOMS)
    atoms["script"] = _script_atom(seq)
    eng = Engine(
        [{"id": "src", "primitive": "script", "params": {}},
         {"id": "c", "primitive": "hadamard4", "params": {}},
         {"id": "s", "primitive": "h4_slide", "params": {}}],
        [{"from": "src.cv", "to": "c.in"},
         {"from": "src.cv", "to": "s.in"}], atoms=atoms)
    f = eng.run(5)["final"]
    assert f["c.w"] == f["s.w"]
    assert f["s.z"] == f["c.y"]
    assert f["s.y"] == f["c.x"]
    assert f["s.x"] == f["c.z"]
