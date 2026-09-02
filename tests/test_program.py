"""iter-5: program.py -- the AtomicProgram IR (blocks + wires + views),
the strict-subset validation (sibling validate_patch rules + the
self-wire and cycle rejections the runner tolerates), and the five
compile targets.

Pinned here:
  * round-trip: every conformance patch -> Program.from_patch ->
    validate() clean -> to_patch() == the original dict (exact);
  * the IR-compiled counter patch runs on the node oracle and matches
    the engine (the IR is transparent to the parity baseline);
  * each rule violation is rejected (dup id, unknown primitive, bad
    endpoint, undeclared src/dst port, inputless node, 2-input
    function, self-wire, cycle, non-lowercase param/port keys,
    dangling entry);
  * the microfx target is accepted by the sibling validate_patch for
    CV programs, and the documented sibling gap is pinned for gate
    programs (gates are not in fabric MODULES; the oracle is the
    authority there);
  * CORE short aliases normalize to canonical gate_* ids on compile;
  * eel2/python/mermaid/wgsl all emit; the python target is a LIVE
    runner over atomic.engine (executed end-to-end in this suite).
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic import Engine, Program, Block, Wire, ProgramError  # noqa: E402
from atomic.gates import ATOMS, Atom  # noqa: E402
from atomic import oracle  # noqa: E402

DT = 1.0 / 30.0

# the 6 conformance patches (vectors re-pinned from tests/test_parity.py)
PATCHES = [
    ("const->gain", 120, None,
     {"modules": [
         {"id": "c1", "primitive": "const", "params": {"value": 5}},
         {"id": "g1", "primitive": "gain", "params": {"factor": 2}}],
      "wires": [{"from": "c1.cv", "to": "g1.in"}],
      "views": [{"module": "g1", "as": "series"}]}),
    ("smooth converges", 200, None,
     {"modules": [
         {"id": "k", "primitive": "const", "params": {"value": 7}},
         {"id": "s1", "primitive": "smooth", "params": {"alpha": 0.5}}],
      "wires": [{"from": "k.cv", "to": "s1.in"}], "views": []}),
    ("mdct_flux finite", 150, None,
     {"modules": [
         {"id": "k2", "primitive": "const", "params": {"value": 3}},
         {"id": "f", "primitive": "mdct_flux", "params": {}}],
      "wires": [{"from": "k2.cv", "to": "f.in"}], "views": []}),
    ("counter via taps", 60, [5, 30],
     {"modules": [
         {"id": "ui", "primitive": "tap", "params": {}},
         {"id": "acc", "primitive": "accum", "params": {"per_tick": 1}}],
      "wires": [{"from": "ui.tap", "to": "acc.in"}], "views": []}),
    ("stacked inputs sum", 5, None,
     {"modules": [
         {"id": "c1", "primitive": "const", "params": {"value": 5}},
         {"id": "c2", "primitive": "const", "params": {"value": 3}},
         {"id": "g", "primitive": "gain", "params": {"factor": 1}}],
      "wires": [{"from": "c1.cv", "to": "g.in"},
                {"from": "c2.cv", "to": "g.in"}], "views": []}),
    ("fan-out", 5, None,
     {"modules": [
         {"id": "c", "primitive": "const", "params": {"value": 5}},
         {"id": "g1", "primitive": "gain", "params": {"factor": 1}},
         {"id": "g2", "primitive": "gain", "params": {"factor": 1}}],
      "wires": [{"from": "c.cv", "to": "g1.in"},
                {"from": "c.cv", "to": "g2.in"}], "views": []}),
]

COUNTER = PATCHES[3][3]


def _diff(js, py, tol=0.0):
    problems = []
    for key in sorted(set(js) | set(py)):
        jv, pv = js.get(key), py.get(key)
        if (jv is None) != (pv is None):
            problems.append("%s: js=%r py=%r" % (key, jv, pv))
        elif jv is None:
            continue
        if isinstance(jv, str) or isinstance(pv, str):
            if jv != pv:
                problems.append("%s: %r != %r" % (key, jv, pv))
        elif abs(float(jv) - float(pv)) > tol:
            problems.append("%s: %r vs %r" % (key, jv, pv))
    return problems


# ------------------------------------------------------------- round-trip

@pytest.mark.parametrize("label,ticks,taps,patch", PATCHES,
                         ids=[p[0] for p in PATCHES])
def test_ir_roundtrip(label, ticks, taps, patch):
    prog = Program.from_patch(patch, name=label)
    assert prog.validate() == [], (label, prog.validate())
    assert prog.to_patch() == patch, "round-trip lost content for %s" % label


def test_ir_hash_is_stable_and_content_sensitive():
    a = Program.from_patch(COUNTER)
    b = Program.from_patch(COUNTER)
    assert a.hash == b.hash
    c = Program.from_patch(COUNTER)
    c.blocks[1].params["per_tick"] = 2
    c.hash = c.compute_hash()
    assert c.hash != a.hash


# ------------------------------------------- IR -> microfx -> oracle parity

def test_ir_counter_patch_oracle_parity():
    prog = Program.from_patch(COUNTER, name="counter")
    compiled = prog.compile("microfx")
    js_final, _ = oracle.run(compiled, 60, dt=DT, ui_taps=[5, 30])
    res = Engine(compiled["modules"], compiled["wires"],
                 views=compiled.get("views") or [],
                 dt=DT, ui_taps=[5, 30]).run(60)
    assert not _diff(js_final, res["final"]), _diff(js_final, res["final"])
    assert res["final"]["acc.acc"] == 2


# --------------------------------------------------------------- validation

def test_rejects_empty_program():
    assert Program("empty").validate() == ["program has no blocks"]


def test_rejects_duplicate_ids():
    p = Program("dup", blocks=[Block("c1", "const"), Block("c1", "const")])
    errs = p.validate()
    assert any("duplicate block id" in e for e in errs), errs


def test_rejects_unknown_primitive():
    p = Program("x", blocks=[Block("c1", "does_not_exist")])
    assert any("unknown primitive" in e for e in p.validate())


def test_rejects_bad_wire_endpoint():
    p = Program("x", blocks=[Block("c1", "const"), Block("g1", "gain")],
                wires=[Wire("c1", "g1.in")])
    assert any("module.port" in e for e in p.validate())


def test_rejects_source_port_not_a_declared_output():
    p = Program("x", blocks=[Block("g1", "gain"), Block("g2", "gain")],
                wires=[Wire("g1.in", "g2.in")])
    errs = p.validate()
    assert any("not a declared output" in e for e in errs), errs


def test_rejects_dest_port_not_a_declared_input():
    p = Program("x", blocks=[Block("c1", "const"), Block("g1", "gain")],
                wires=[Wire("c1.cv", "g1.cv")])
    errs = p.validate()
    assert any("not a declared input port" in e for e in errs), errs


def test_rejects_wire_into_inputless_node():
    p = Program("x", blocks=[Block("c1", "const"), Block("c2", "const")],
                wires=[Wire("c1.cv", "c2.in")])
    errs = p.validate()
    assert any("not a declared input port" in e for e in errs), errs


def test_rejects_wire_to_unknown_module():
    p = Program("x", blocks=[Block("g1", "gain")],
                wires=[Wire("zz.cv", "g1.in")])
    errs = p.validate()
    assert any("unknown module" in e for e in errs), errs


def test_node_rule_rejects_two_input_function():
    # catalog atoms are self-consistent (multi_in iff >1 input), so the
    # port-level rule needs a hand-built atom to bite:
    ATOMS["fake2in"] = Atom("fake2in", "Fake two-input", "function",
                             {}, ["a", "b"], ["out"], "", multi_in=False)
    try:
        p = Program("x", blocks=[Block("f1", "fake2in")])
        errs = p.validate()
        assert any("must have one input port" in e for e in errs), errs
    finally:
        del ATOMS["fake2in"]


def test_node_rule_exempts_multi_in_and_sinks():
    p = Program("x", blocks=[Block("an", "gate_and"),     # 2 in, multi_in
                              Block("tf", "toffoli"),      # 3 in, multi_in
                              Block("v1", "viz_wxyz3d")])  # 4 in, sink
    assert p.validate() == [], p.validate()


def test_rejects_self_wire():
    p = Program("x", blocks=[Block("g1", "gain")],
                wires=[Wire("g1.cv", "g1.in")])
    assert any("self-wire" in e for e in p.validate())


def test_rejects_cycle():
    p = Program("x",
                blocks=[Block("c1", "const"), Block("g1", "gain"),
                        Block("g2", "gain")],
                wires=[Wire("c1.cv", "g1.in"),
                       Wire("g1.cv", "g2.in"),
                       Wire("g2.cv", "g1.in")])
    errs = p.validate()
    assert any("cycle" in e for e in errs), errs


def test_non_ui_source_with_other_id_is_a_plain_block():
    # 'ui' is the one virtual source; any other tap id is a plain block
    # (dead in oracle mode 1 -- pinned in the contract, not a build error)
    p = Program("x", blocks=[Block("t1", "tap"), Block("ac", "accum")],
                wires=[Wire("t1.trig", "ac.in")])
    assert p.validate() == [], p.validate()


def test_rejects_non_lowercase_param_key():
    p = Program("x", blocks=[Block("c1", "const", {"Value": 5})])
    assert any("must be lowercase" in e for e in p.validate())


def test_rejects_non_lowercase_wire_ports():
    p = Program("x", blocks=[Block("c1", "const"), Block("g1", "gain")],
                wires=[Wire("c1.CV", "g1.in")])
    assert any("lowercase" in e for e in p.validate())


def test_rejects_dangling_entry():
    p = Program("x", blocks=[Block("c1", "const")], entry="zz")
    assert any("entry block" in e for e in p.validate())


def test_compile_raises_program_error_when_invalid():
    p = Program("x", blocks=[Block("c1", "const"), Block("c1", "const")])
    with pytest.raises(ProgramError):
        p.compile("microfx")


# ------------------------------------------------------------------ targets

def _cv_program():
    return Program("cv_app", description="const -> gain -> smooth -> chart",
                   blocks=[Block("c1", "const", {"value": 5}),
                           Block("g1", "gain", {"factor": 2}),
                           Block("s1", "smooth", {"alpha": 0.2}),
                           Block("v1", "viz_series")],
                   wires=[Wire("c1.cv", "g1.in"), Wire("g1.cv", "s1.in"),
                          Wire("s1.cv", "v1.in")],
                   views=[{"module": "s1", "output": "cv", "as": "series"}],
                   tags=["cv", "test"])


def test_microfx_target_shape_and_fabric_accepts():
    patch = _cv_program().compile("microfx")
    assert set(patch) == {"modules", "wires", "views"}
    assert patch["modules"][0] == {
        "id": "c1", "primitive": "const", "params": {"value": 5}}
    assert patch["wires"] == [{"from": "c1.cv", "to": "g1.in"},
                               {"from": "g1.cv", "to": "s1.in"},
                               {"from": "s1.cv", "to": "v1.in"}]
    import fabric.microfx as mf
    assert mf.validate_patch(patch) is None


def test_fabric_validator_gap_for_gates_is_pinned():
    # documented sibling defect: the gate atoms were removed from
    # MODULES (2026-08-26, Control library only), so fabric's
    # validate_patch reports them as unknown even though the jsfx
    # oracle runs them fine. The harness IR (ATOMS) is the authority.
    p = Program("gate_app",
                blocks=[Block("ui", "tap"), Block("tg", "toggle"),
                        Block("gb", "gate_buffer")],
                wires=[Wire("ui.tap", "tg.trig"),
                       Wire("tg.state", "gb.in")])
    assert p.validate() == [], p.validate()
    import fabric.microfx as mf
    err = mf.validate_patch(p.compile("microfx"))
    assert err is not None and "unknown primitive" in err, err


def test_short_aliases_normalize_to_canonical_ids():
    p = Program("alias",
                blocks=[Block("c1", "const", {"value": 1}),
                        Block("ta", "toggle", {"initial": 0}),
                        Block("and1", "and")],
                wires=[Wire("ui.tap", "ta.trig"),
                       Wire("ta.state", "and1.a")])
    assert p.validate() == [], p.validate()
    mod = {m["id"]: m for m in p.compile("microfx")["modules"]}
    assert mod["and1"]["primitive"] == "gate_and"
    assert mod["ta"]["primitive"] == "toggle"


def test_eel2_target_concatenates_bodies_and_skips_sinks():
    src = _cv_program().compile("eel2")
    assert isinstance(src, str) and src.startswith("// Autogenerated")
    assert "c1 (const)" in src and "@tick" in src
    assert "viz_series" not in src  # sinks render from views[], no body


def test_mermaid_and_wgsl_targets_emit():
    m = _cv_program().compile("mermaid")
    assert m.startswith("flowchart TD") and "c1" in m
    assert "c1 --> g1" in m
    w = _cv_program().compile("wgsl")
    assert w.startswith("// WGSL") and "cv_app" in w


def test_unknown_target_raises():
    with pytest.raises(ProgramError):
        _cv_program().compile("fortran")


def test_python_target_is_a_live_runner(tmp_path):
    src = _cv_program().compile("python")
    path = tmp_path / "run_cv.py"
    path.write_text(src)
    out = subprocess.run([sys.executable, str(path), "10"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    final = json.loads(out.stdout)
    # const(5) -> gain(x2) with the 1-tick wire latency: settled at 10
    assert final["g1.cv"] == 10
    assert "s1.cv" in final
