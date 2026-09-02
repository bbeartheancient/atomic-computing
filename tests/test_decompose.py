"""Decompose tests (goal 8): Python/EEL2 -> Program."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/M1Multitronic"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic import decompose_python, decompose_eel2, decompose_expr, DecomposeError, Engine

def test_decompose_python_const_and_gain():
    p = decompose_python("x = 2 * 3", name="t")
    assert p.validate()==[], p.validate()
    res = Engine(p.compile("microfx")["modules"], p.compile("microfx")["wires"]).run(5)
    # at least one numeric output reaches sink via gain chain
    assert any(isinstance(v, float) for v in res["final"].values())

def test_decompose_python_threshold():
    p = decompose_python("a = 5\nif a > 3:\n    y = 1", name="thr")
    assert p.validate()==[], p.validate()
    assert any(b.primitive=="threshold" for b in p.blocks)

def test_decompose_python_arith_valid():
    p = decompose_python("out = 4 + 2", name="add")
    assert p.validate()==[]

def test_decompose_python_empty_raises():
    try:
        decompose_python("", name="empty")
        assert False
    except DecomposeError:
        pass

def test_decompose_python_function():
    p = decompose_python("def f(x):\n    return x * 2", name="fn")
    assert p.validate()==[], p.validate()

def test_decompose_eel2():
    src = "v = input('in'); output('cv', v * 2.0);"
    p = decompose_eel2(src, name="eel")
    assert p.validate()==[], p.validate()
    assert any(b.primitive=="gain" for b in p.blocks)

def test_decompose_eel2_empty_raises():
    try:
        decompose_eel2("  ", name="e")
        assert False
    except DecomposeError:
        pass

def test_decompose_expr():
    p = decompose_expr("2 * 5", name="expr")
    assert p.validate()==[]

def test_decompose_roundtrip_engine():
    p = decompose_python("y = 2 * 4", name="rt")
    patch = p.compile("microfx")
    assert patch["modules"] and patch["wires"]
    res = Engine(patch["modules"], patch["wires"]).run(4)
    assert res["final"] is not None
