"""Iter 12 hardening: decompose Add wire-SUM, EEL2 multi-input, Evolver QBF history, H4 bridge."""
import os, sys, tempfile, shutil
# sys.path sibling no longer needed (fabric/ is vendored)
from atomic import Engine, Program, Block, Wire, decompose_python, decompose_eel2, Evolver
from atomic.bridge import HostBridge
from atomic.qbfstore import open_trace_store, close_all

def test_decompose_add_wire_sum_structural():
    prog = decompose_python("x = a + b", name="add_sum")
    assert prog.validate()==[], prog.validate()
    # must be wire SUM: at least one sink has two incoming wires
    dst_counts = {}
    for w in prog.wires:
        dst_counts[w.dst] = dst_counts.get(w.dst, 0)+1
    assert max(dst_counts.values()) >= 2, dst_counts
    # functional: explicit Add via bias SUM gives 5
    patch = {"modules":[{"id":"c1","primitive":"const","params":{"value":2}},{"id":"c2","primitive":"const","params":{"value":3}},{"id":"s","primitive":"bias","params":{"add":0}},{"id":"v","primitive":"viz_series","params":{}}],"wires":[{"from":"c1.cv","to":"s.in"},{"from":"c2.cv","to":"s.in"},{"from":"s.cv","to":"v.in"}],"views":[]}
    res = Engine(patch["modules"], patch["wires"]).run(3)
    assert res["final"]["s.cv"]==5.0

def test_decompose_eel2_multi_input():
    src = "v = input('sensor'); w = input('aux'); output('cv', v + w);"
    prog = decompose_eel2(src, name="eel_multi")
    assert prog.validate()==[], prog.validate()
    assert any(b.primitive=="sensor" for b in prog.blocks)
    assert len(prog.blocks) >= 3

def test_evolver_qbf_history_roundtrip():
    def fitness(final): return -abs(float(final.get("g1.cv",0))-10.0)
    prog = Program("tun", blocks=[Block("c0","const",{"value":5}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
    ev = Evolver(prog, fitness, seed=1, ticks=5)
    ev.run(5)
    tmp = tempfile.mkdtemp(prefix="test_evolver12_")
    try:
        store_name = "evolve12_%d" % os.getpid()
        ev.save_history(store_name=store_name, shard_dir=tmp)
        loaded = Evolver.load_history(store_name, shard_dir=tmp)
        assert loaded["history"]==ev.history
        assert loaded["best"]["modules"][1]["params"]["factor"]==ev.best.blocks[1].params["factor"]
    finally:
        try: close_all()
        except: pass
        shutil.rmtree(tmp, ignore_errors=True)

def test_bridge_h4_codec():
    b = HostBridge(latency=1, capacity=8, use_h4=True)
    payload = {"a":1.0,"b":2.0,"c":3.0,"d":4.0}
    b.push(0, payload)
    out = b.pop(1)
    assert out is not None
    for k in payload:
        assert abs(out[k]-payload[k])<1e-6, (k,out[k])
    b2 = HostBridge(latency=1, capacity=8, use_h4=True)
    b2.push(0, {"x":5.0,"y":6.0})
    assert b2.pop(1)=={"x":5.0,"y":6.0}

def test_wgsl_shape_iter12():
    p = Program("wgsl12", blocks=[Block("c0","const",{"value":1}), Block("g1","gain",{"factor":2}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
    w = p.compile("wgsl")
    assert w.startswith("// WGSL")
    assert "@compute" in w and "@group(0)" in w and "host-RAM" in w
    assert "fn tick_c0" in w and "fn tick_g1" in w

def test_decompose_add_const_fold():
    prog = decompose_python("x = 2 + 3", name="add_const")
    assert prog.validate()==[]
    # const fold via bias add param: at least one bias with add=3
    biases = [b for b in prog.blocks if b.primitive=="bias"]
    assert any(abs(b.params.get("add",0)-3)<1e-9 for b in biases) or any(b.params.get("add",0)==3 for b in biases)
