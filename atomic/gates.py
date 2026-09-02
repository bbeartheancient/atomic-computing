"""Gates: the atom catalog -- CV atoms first (build-order steps 2/3).

Every atom is multi-target: the EEL2 body is IMPORTED from the sibling
fabric catalog (fabric/microfx.py MODULES -- never vendored) and the
Python impl below is what the engine executes. Parity between the two
is pinned by the conformance patches (tests/test_parity.py): bit-exact
on integer/affine paths, <= 1e-9 where a body calls Math.* (the V8
Math vs CPython libm last-ulp seam: sine_lfo sin, mdct_flux cos).

Host sources: jsfx's HOST_SOURCES_SET is {clock_bpm, sensor} -- tap is
NOT in that set; it compiles an EMPTY body in the patch runner (the
fixture source is "") and the runner writes the hardcoded bus key
"ui.tap" itself. The twin models sensor and tap as inert host sources:
bus-observable behavior is identical (the module never writes; the
engine owns "ui.tap" per tick).

Known seams (documented, not chased in goal 1):
- param keys are matched lowercased in the twin; the JS proxy matches
  them verbatim (m.params). The harness IR (program.py) will enforce
  lowercase keys, so patch-level parity is unaffected.
- min/max on a NaN arg: JS Math.min(x, NaN) = NaN, Python min(x, nan)
  = x. Unreachable in the patch context: inputs are finitized at the
  wire latch and the conformance patches feed numeric params only.
- fabric's hadamard4 emits ports (w, y, x, z) with the y/x patterns
  SWAPPED vs CORE's canonical rows (W [++++], Z [+ - + -], Y [++ - -],
  X [+ - - +]). The twin mirrors the fabric module (oracle parity);
  the CORE-canonical gate is the separate h4_slide atom (step 3).
"""

import json
import math
import os
import sys

from . import dsp
from .jsnum import cond_truthy, js_falsy, js_number, js_or0


def _fabric_modules():
    try:
        import fabric.microfx as mf
    except ImportError:
        sys.path.insert(0, os.path.expanduser(os.path.join("~", "M1Multitronic")))
        import fabric.microfx as mf
    return mf.MODULES


_MF = _fabric_modules()


def _fabric_gate_tables():
    # _GATES/_QGATES are module-level tables in microfx (NOT entries of
    # MODULES -- the gates were removed from MODULES, 2026-08-26).
    try:
        import fabric.microfx as mf
    except ImportError:
        sys.path.insert(0, os.path.expanduser(os.path.join("~", "M1Multitronic")))
        import fabric.microfx as mf
    return mf._GATES, mf._QGATES


_GATES, _QGATES = _fabric_gate_tables()


class Atom:
    """One gate: a multi-target spec (EEL2 body + Python impl)."""

    def __init__(self, name, title, category, params, inputs, outputs,
                 source, multi_in=False, init=None, tick=None, host=None):
        self.name = name
        self.title = title
        self.category = category
        self.params = params    # {name: default}, lowercase names
        self.inputs = inputs
        self.outputs = outputs
        self.multi_in = multi_in
        self.source = source    # EEL2 body (the microfx compile target)
        self.init = init        # fn(node) once at t==0 (or None)
        self.tick = tick        # fn(node) per tick (or None)
        self.host = host        # fn(node, t, dt, bus) for host sources

    def __repr__(self):
        return "Atom(%r, %r)" % (self.name, self.category)


ATOMS = {}


def _reg(name, init=None, tick=None, host=None, multi_in=False):
    e = _MF[name]
    params = {p["name"]: p["default"] for p in e.get("params", [])}
    atom = Atom(name, e.get("title", name), e.get("category", "function"),
                params, list(e.get("inputs", [])), list(e.get("outputs", [])),
                e.get("source", ""), multi_in=multi_in,
                init=init, tick=tick, host=host)
    ATOMS[name] = atom
    return atom


# ---------------------------------------------------------------- CV atoms

def _const_tick(n):
    n.output("cv", n.read("value"))


_reg("const", tick=_const_tick)


def _clock_host(node, t, dt, bus):
    # jsfx.js:1113-1120: the dt ACCUMULATOR (not PulseGenerator)
    bpm = js_number(node.params.get("bpm"))
    if js_falsy(bpm):
        bpm = 60.0
    period = 60.0 / max(0.1, bpm)
    acc = js_or0(node.hostState.get("acc")) + dt
    node.hostState["acc"] = acc
    if acc >= period:
        node.hostState["acc"] = acc - period
        bus.set(node.id + ".trig", 1)
    else:
        bus.set(node.id + ".trig", 0)


_reg("clock_bpm", host=_clock_host)


def _sine_init(n):
    n.set_var("phase", 0.0)


def _sine_tick(n):
    # dt = 1/30; phase += rate_hz * dt; cv = offset + amp * sin(...)
    dt = 1.0 / 30.0
    n.set_var("dt", dt)
    ph = n.read("phase")
    ph = ph + n.read("rate_hz") * dt
    n.set_var("phase", ph)
    cv = n.read("offset") + n.read("amp") * math.sin(ph * 6.2832)
    n.set_var("cv", cv)
    n.output("cv", cv)


_reg("sine_lfo", init=_sine_init, tick=_sine_tick)


def _gain_tick(n):
    n.output("cv", n.input("in") * n.read("factor"))


_reg("gain", tick=_gain_tick)


def _bias_tick(n):
    n.output("cv", n.input("in") + n.read("add"))


_reg("bias", tick=_bias_tick)


def _smooth_init(n):
    n.set_var("y", 0.0)


def _smooth_tick(n):
    # y += alpha * (input - y): RHS fully evaluated on the OLD y
    y = n.read("y")
    y = y + n.read("alpha") * (n.input("in") - y)
    n.set_var("y", y)
    n.output("cv", y)


_reg("smooth", init=_smooth_init, tick=_smooth_tick)


def _thresh_init(n):
    n.set_var("state", 0.0)


def _thresh_tick(n):
    # v > hi ? state = 1 : v < lo ? state = 0 : 0  (ternary short-
    # circuits: only the selected branch evaluates)
    v = n.input("in")
    n.set_var("v", v)
    if cond_truthy(n.read("v") > n.read("hi")):
        n.set_var("state", 1.0)
    elif cond_truthy(n.read("v") < n.read("lo")):
        n.set_var("state", 0.0)
    n.output("gate", n.read("state"))


_reg("threshold", init=_thresh_init, tick=_thresh_tick)


def _mavg_init(n):
    n.set_var("buf", 0.0)
    n.set_var("idx", 0.0)
    n.set_var("filled", 0.0)
    n.set_var("acc", 0.0)


def _mavg_tick(n):
    v = n.input("in")
    n.set_var("v", v)
    base = n.var0("buf")
    idx = n.read("idx")
    old = n.mem_read(base, idx)
    n.mem_write(base, idx, v)
    acc = n.read("acc") + (v - old)
    n.set_var("acc", acc)
    idx = idx + 1.0
    npar = n.read("n")
    if cond_truthy(idx >= npar):
        idx = 0.0
    n.set_var("idx", idx)
    filled = n.read("filled")
    if cond_truthy(filled < npar):
        filled = filled + 1.0
        n.set_var("filled", filled)
    # filled ? acc / filled : v   (EEL2 cond: |c| > 1e-5)
    out = (acc / filled) if cond_truthy(filled) else v
    n.output("cv", out)


_reg("moving_avg", init=_mavg_init, tick=_mavg_tick)


def _clamp_tick(n):
    n.output("cv", min(n.read("hi"), max(n.read("lo"), n.input("in"))))


_reg("clamp", tick=_clamp_tick)


def _flux_init(n):
    n.set_var("buf", 0.0)
    n.set_var("idx", 0.0)
    n.set_var("prev_e", 0.0)


def _flux_tick(n):
    base = n.var0("buf")
    idx = n.read("idx")
    n.mem_write(base, idx, n.input("in"))
    idx = idx + 1.0
    if cond_truthy(idx >= 64):
        idx = 0.0
        dsp.mdct(n.mem, 0, 64)
        e = 0.0
        for k in range(32):
            e = e + n.mem_read(0, k) * n.mem_read(0, k)
        prev_e = n.read("prev_e")
        flux = abs(e - prev_e)
        n.set_var("prev_e", e)
        n.set_var("flux", flux)
    n.set_var("idx", idx)
    # first 63 ticks: flux var unassigned -> undefined on the bus
    n.output("flux", n.read("flux"))


_reg("mdct_flux", init=_flux_init, tick=_flux_tick)


def _noop_host(node, t, dt, bus):
    # sensor: values pushed via setValue (DOM path only; the patch
    # runner never feeds it). tap: empty body; bus key "ui.tap" is
    # owned by the engine. Both: inert in the oracle.
    return None


_reg("sensor", host=_noop_host)
_reg("tap", host=_noop_host)


def _tog_init(n):
    n.set_var("state", n.read("initial"))


def _tog_tick(n):
    if cond_truthy(n.input("trig") > 0):
        n.set_var("state", 1.0 - n.read("state"))
    n.output("state", n.read("state"))


_reg("toggle", init=_tog_init, tick=_tog_tick)


def _acc_init(n):
    n.set_var("acc", 0.0)
    n.set_var("prev", 0.0)


def _acc_tick(n):
    # rising-edge counter: v = in > 0 (bool); edge state lives in the
    # ATOM (prev var), not the runner
    v = n.input("in") > 0
    n.set_var("v", v)
    if cond_truthy(v > n.read("prev")):
        n.set_var("acc", n.read("acc") + n.read("per_tick"))
    n.set_var("prev", v)
    if cond_truthy(n.read("wrap") > 0 and n.read("acc") >= n.read("wrap")):
        n.set_var("acc", 0.0)
    n.output("acc", n.read("acc"))


_reg("accum", init=_acc_init, tick=_acc_tick)


def _h4_init(n):
    n.set_var("s0", 0.0)
    n.set_var("s1", 0.0)
    n.set_var("s2", 0.0)


def _h4_tick(n):
    # 4-sample sliding window [v, s0, s1, s2]; all four outputs read
    # the OLD window (reads precede the shift assignments)
    v = n.input("in")
    n.set_var("v", v)
    s0 = n.read("s0")
    s1 = n.read("s1")
    s2 = n.read("s2")
    n.output("w", v + s0 + s1 + s2)
    n.output("y", v - s0 + s1 - s2)
    n.output("x", v + s0 - s1 - s2)
    n.output("z", v - s0 - s1 + s2)
    n.set_var("s2", s1)
    n.set_var("s1", s0)
    n.set_var("s0", v)


_reg("hadamard4", init=_h4_init, tick=_h4_tick)

# ------------------------------------------- logic + quantum + alogic + H4
#
# The gate set (the 10 logic macros, the single/double-qubit quantum
# gates, and cnot/swap/toffoli) was REMOVED from MODULES per operator
# spec (2026-08-26) -- it lives only in the Control library, i.e. in the
# test FIXTURE (fabric/tests/microfx_modules.json). The oracle registers
# exactly those fixture bodies, so these Atoms are built from the
# FIXTURE too (titles/ports from the _GATES/_QGATES catalog; EEL2
# source + params verbatim from the fixture). _reg() cannot be used
# here: it reads MODULES, where the gates no longer exist.

def _fixture():
    p = os.path.expanduser(
        os.path.join("~", "M1Multitronic", "fabric", "tests",
                     "microfx_modules.json"))
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {"source": {}, "params": {}}


_FIX = _fixture()


def _fix_atom(name, title, category, inputs, outputs, tick,
              init=None, multi_in=False):
    src = _FIX["source"].get(name, "")
    prm = {p["name"]: p["default"]
           for p in _FIX.get("params", {}).get(name, [])}
    atom = Atom(name, title, category, prm, list(inputs), list(outputs),
                src, multi_in=multi_in, init=init, tick=tick)
    ATOMS[name] = atom
    return atom


# -- logic macros (the 10 gate_* of fabric _GATES) -------------------------
# EEL2 `> 0.5` yields a 0/1 NUMBER (jsfx.js:404-407) and && / || / ! are
# the numeric EEL2 ops (|v|<1e-5, jsfx.js:323/380/384). After the leading
# threshold every operand is exactly 0 or 1, so the twins collapse to
# plain Python truth tables and are bit-exact (out is 1.0 / 0.0).

def _lg_buf(n):
    n.output("q", 1.0 if n.input("in") > 0.5 else 0.0)


def _lg_not(n):
    n.output("q", 0.0 if n.input("in") > 0.5 else 1.0)


def _lg_and(n):
    n.output("q", 1.0 if (n.input("a") > 0.5 and n.input("b") > 0.5) else 0.0)


def _lg_or(n):
    n.output("q", 1.0 if (n.input("a") > 0.5 or n.input("b") > 0.5) else 0.0)


def _lg_nand(n):
    n.output("q", 0.0 if (n.input("a") > 0.5 and n.input("b") > 0.5) else 1.0)


def _lg_nor(n):
    n.output("q", 0.0 if (n.input("a") > 0.5 or n.input("b") > 0.5) else 1.0)


def _lg_xor(n):
    n.output("q", 1.0 if (n.input("a") > 0.5) != (n.input("b") > 0.5) else 0.0)


def _lg_xnor(n):
    n.output("q", 1.0 if (n.input("a") > 0.5) == (n.input("b") > 0.5) else 0.0)


def _lg_imply(n):
    # EEL2: a && b || !a  (0/1 numbers)  ==  (a and b) or (not a)
    a = n.input("a") > 0.5
    b = n.input("b") > 0.5
    n.output("q", 1.0 if ((a and b) or (not a)) else 0.0)


def _lg_nimply(n):
    # EEL2: a && !b
    a = n.input("a") > 0.5
    b = n.input("b") > 0.5
    n.output("q", 1.0 if (a and (not b)) else 0.0)


_LOGIC_TICKS = {
    "gate_buffer": _lg_buf,
    "gate_not": _lg_not,
    "gate_and": _lg_and,
    "gate_or": _lg_or,
    "gate_nand": _lg_nand,
    "gate_nor": _lg_nor,
    "gate_xor": _lg_xor,
    "gate_xnor": _lg_xnor,
    "gate_imply": _lg_imply,
    "gate_nimply": _lg_nimply,
}

for _name, (_title, _inps, _body) in _GATES.items():
    _fix_atom(_name, _title, "logic", _inps, ["q"], _LOGIC_TICKS[_name],
              multi_in=len(_inps) > 1)

# CORE aliases: the spec names the macros by their short forms; the
# catalog keeps the gate_* ids (same Atom object, so the compiler later
# still emits atom.name == "gate_*").
for _short, _full in [("buffer", "gate_buffer"), ("not", "gate_not"),
                      ("and", "gate_and"), ("or", "gate_or"),
                      ("nand", "gate_nand"), ("nor", "gate_nor"),
                      ("xor", "gate_xor"), ("xnor", "gate_xnor"),
                      ("imply", "gate_imply"), ("nimply", "gate_nimply")]:
    ATOMS[_short] = ATOMS[_full]


# -- quantum set (bipolar +/-1 streams: |0>=+1, |1>=-1) ---------------------

def _pauli_tick(n):
    # computational NOT is a sign flip (Y == -in on real streams)
    n.output("out", -n.input("q"))


def _hadam_tick(n):
    n.output("out", n.input("q") * 0.7071067811865476)


def _phase_tick(n):
    # S/T phase rotations are identity on real amplitudes
    n.output("out", n.input("q"))


def _cnot_tick(n):
    c = n.input("c")
    n.output("out", n.input("t") * (-1.0 if c > 0.5 else 1.0))
    n.output("c_out", c)


def _swap_tick(n):
    n.output("a_out", n.input("b"))
    n.output("b_out", n.input("a"))


def _toffoli_tick(n):
    c1 = n.input("c1")
    c2 = n.input("c2")
    t = n.input("t")
    flip = (c1 > 0.5) and (c2 > 0.5)   # EEL2 c = (c1>0.5 && c2>0.5)
    n.output("out", t * (-1.0 if flip else 1.0))
    n.output("c_out", c1)


_QTICKS = {
    "pauli_x": _pauli_tick,
    "pauli_y": _pauli_tick,
    "pauli_z": _pauli_tick,
    "hadamard_gate": _hadam_tick,
    "phase_s": _phase_tick,
    "phase_t": _phase_tick,
}

for _name, (_title, _inps, _body) in _QGATES.items():
    _fix_atom(_name, _title, "quantum", _inps, ["out"], _QTICKS[_name])

# cnot / swap_gate / toffoli: removed from MODULES; the fixture is
# the canonical home for their bodies (ports per the control library).
_fix_atom("cnot", "CNOT", "quantum", ["c", "t"], ["out", "c_out"],
          _cnot_tick, multi_in=True)
_fix_atom("swap_gate", "SWAP", "quantum", ["a", "b"], ["a_out", "b_out"],
          _swap_tick, multi_in=True)
_fix_atom("toffoli", "Toffoli (CCNOT)", "quantum", ["c1", "c2", "t"],
          ["out", "c_out"], _toffoli_tick, multi_in=True)


# -- alogic (analog logic families) ------------------------------------------

def _alogic_init(n):
    n.set_var("st", 0.0)


def _alogic_tick(n):
    # Mirrors the EEL2 body 1:1: `v = input('in'); t = thresh;` then six
    # independent `family == N ? (...)` ternaries (exactly one active),
    # then the final quantize `q = q ? 1 : 0;`. The q/st interpreter
    # vars PERSIST across ticks exactly as in the JS interpreter (a
    # valid family 0..5 always assigns q before the quantize).
    v = n.input("in")
    t = n.read("thresh")
    fam = n.read("family")
    if fam == 0:                                 # DL: diode drop
        n.set_var("q", 1.0 if v > 0.6 else 0.0)
    elif fam == 1:                               # TDL: tunnel latch
        if v > t + 0.1:
            n.set_var("st", 1.0)
        elif v < t - 0.1:
            n.set_var("st", 0.0)
        n.set_var("q", n.read("st"))
    elif fam == 2:                               # NL: neon strike
        if v > 0.75:
            n.set_var("st", 1.0)
        elif v < 0.4:
            n.set_var("st", 0.0)
        n.set_var("q", n.read("st"))
    elif fam == 3:                               # MOS: sharp
        n.set_var("q", 1.0 if v > t else 0.0)
    elif fam == 4:                               # CML: small swing
        d = v - t
        if d > 0.25:
            d = 0.25
        elif d < -0.25:
            d = -0.25
        n.set_var("q", 0.5 + d)
    elif fam == 5:                               # QCA: majority cell
        a = 1.0 if v > t else 0.0
        b = 1.0 if n.read("cell_b") > 0.5 else 0.0
        c = 1.0 if n.read("cell_c") > 0.5 else 0.0
        n.set_var("q", 1.0 if (a + b + c) >= 2 else 0.0)
    q = n.read("q")
    n.set_var("q", 1.0 if (q is not None and abs(q) > 1e-5) else 0.0)
    n.output("q", n.read("q"))


_fix_atom("alogic", "Analog logic family", "function", ["in"], ["q"],
          _alogic_tick, init=_alogic_init, multi_in=True)


# -- h4_slide: the CORE-canonical H(4) spatial gate --------------------------
# fabric's hadamard4 emits the SAME math with the y/x/z labels SWAPPED;
# this atom carries CORE's canonical row order on the 4-sample window
# [v, s0, s1, s2] (v = newest):
#   Row0 W [+ + + +]   Row1 Z [+ - + -]
#   Row2 Y [+ + - -]   Row3 X [+ - - +]
# i.e. h4_slide == the 4x4 Sylvester-Hadamard (hoa64.sylvester(4)).
# Harness-only: NOT in the oracle fixture, so parity is pinned against
# hoa64 directly (tests/test_gates_parity.py), not via the node oracle.

_H4S_SOURCE = ("@init\n"
               "s0 = 0; s1 = 0; s2 = 0;\n"
               "\n"
               "@tick\n"
               "v = input('in');\n"
               "output('w', v + s0 + s1 + s2);   // Row0 [+ + + +]\n"
               "output('z', v - s0 + s1 - s2);   // Row1 [+ - + -]\n"
               "output('y', v + s0 - s1 - s2);   // Row2 [+ + - -]\n"
               "output('x', v - s0 - s1 + s2);   // Row3 [+ - - +]\n"
               "s2 = s1; s1 = s0; s0 = v;")


def _h4s_init(n):
    n.set_var("s0", 0.0)
    n.set_var("s1", 0.0)
    n.set_var("s2", 0.0)


def _h4s_tick(n):
    # reads the OLD window (v, s0, s1, s2), emits all four rows, THEN
    # shifts -- identical ordering to fabric hadamard4
    v = n.input("in")
    s0 = n.read("s0")
    s1 = n.read("s1")
    s2 = n.read("s2")
    n.output("w", v + s0 + s1 + s2)
    n.output("z", v - s0 + s1 - s2)
    n.output("y", v + s0 - s1 - s2)
    n.output("x", v - s0 - s1 + s2)
    n.set_var("s2", s1)
    n.set_var("s1", s0)
    n.set_var("s0", v)


ATOMS["h4_slide"] = Atom("h4_slide", "H(4) spatial gate (CORE rows)",
                         "spatial", {"domain": "tensor"}, ["in"],
                         ["w", "z", "y", "x"], _H4S_SOURCE,
                         init=_h4s_init, tick=_h4s_tick)


# visualizers: sinks; empty body in the patch runner (rendered from
# views[]) -> no-op atoms
_reg("viz_series")
_reg("viz_xy")
_reg("viz_wxyz3d")
