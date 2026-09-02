"""Iter 13 polish: Sub/Div wire-SUM, teach QBF, evolver swarm, wgsl extended."""
import os, sys, tempfile, shutil
# sys.path sibling no longer needed (fabric/ is vendored)
from atomic import Engine, Program, Block, Wire, decompose_python, decompose_eel2, Evolver
from atomic.teach import TeacherRegistry, REGISTRY, save_registry, load_registry
from atomic.qbfstore import open_trace_store, close_all

def test_decompose_sub_wire_sum():
    prog = decompose_python("x = a - b", name="sub_sum")
    assert prog.validate()==[]
    assert any(b.primitive=="bias" for b in prog.blocks)
    assert any(b.primitive=="gain" and b.params.get("factor")==-1.0 for b in prog.blocks)
    # const fold: 7-2 -> 5
    prog2 = decompose_python("x = 7 - 2", name="sub_const")
    assert prog2.validate()==[]
    patch = prog2.compile("microfx")
    res = Engine(patch["modules"], patch["wires"]).run(3)
    bias_id = next(b.id for b in prog2.blocks if b.primitive=="bias")
    assert abs(float(res["final"][bias_id+".cv"]) - 5.0) < 1e-9

def test_decompose_div_annassign():
    prog = decompose_python("y: float = 8 / 2", name="div_ann")
    assert prog.validate()==[]
    patch = prog.compile("microfx")
    res = Engine(patch["modules"], patch["wires"]).run(3)
    assert any(isinstance(v,(int,float)) for v in res["final"].values())
    # AugAssign + chain
    prog2 = decompose_python("x = 3\nx += 2", name="aug")
    assert prog2.validate()==[]
    prog3 = decompose_python("a = 2\nb = 3\nc = a + b\nc = c * 2", name="chain")
    assert prog3.validate()==[]

def test_teach_qbf_roundtrip():
    tmp = tempfile.mkdtemp(prefix="test_teach13_")
    try:
        reg = TeacherRegistry()
        p = Program("tprog", blocks=[Block("c0","const",{"value":5}),Block("v0","viz_series")], wires=[Wire("c0.cv","v0.in")], description="teach qbf test")
        reg.register("teach qbf test example", p, domain="signal")
        path = reg.save_qbf(store_name="teach13_%d" % os.getpid(), shard_dir=tmp)
        loaded = TeacherRegistry.load_qbf("teach13_%d" % os.getpid(), shard_dir=tmp)
        assert len(loaded.examples)==1
        assert loaded.match("teach qbf test example") is not None
        # file path
        fp = os.path.join(tmp, "teach_file.qbf")
        reg.save_qbf(path=fp)
        loaded2 = TeacherRegistry.load_qbf(fp)
        assert len(loaded2.examples)==1
    finally:
        try: close_all()
        except: pass
        shutil.rmtree(tmp, ignore_errors=True)

def test_evolver_swarm_parallel():
    def fitness(final): return -abs(float(final.get("g1.cv",0))-10.0)
    prog = Program("tun", blocks=[Block("c0","const",{"value":5}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
    ev = Evolver(prog, fitness, seed=7, ticks=5)
    ev.run_swarm(generations=8, population=4, parallel=True)
    assert ev.best_score > -5.0
    ev2 = Evolver(prog, fitness, seed=7, ticks=5)
    ev2.run_swarm(generations=8, population=4, parallel=False)
    assert ev.best.hash==ev2.best.hash

def test_wgsl_extended_primitives():
    p = Program("wgsl_ext", blocks=[Block("c0","const",{"value":2}), Block("th","threshold",{"hi":0.5,"lo":-0.5}), Block("cl","clamp",{"lo":-1,"hi":1}), Block("lf","sine_lfo"), Block("v0","viz_series")],
                wires=[Wire("c0.cv","th.in"), Wire("th.gate","cl.in"), Wire("cl.cv","v0.in")])
    w = p.compile("wgsl")
    assert "@compute" in w and "host-RAM" in w
    assert "clamp" in w and "tick_th" in w
    assert "sin(" in w
    assert "tick_lf" in w
    pc = Program("wgsl_clk", blocks=[Block("clk","clock_bpm",{"bpm":60}), Block("v0","viz_series")], wires=[Wire("clk.trig","v0.in")])
    wc = pc.compile("wgsl")
    assert "tick_clk" in wc

def test_decompose_eel2_complex_valid():
    src = "v = input('sensor'); w = input('aux'); output('cv', v * 2.0 + w);"
    prog = decompose_eel2(src, name="eel_complex")
    assert prog.validate()==[]
    assert any(b.primitive=="sensor" for b in prog.blocks)
