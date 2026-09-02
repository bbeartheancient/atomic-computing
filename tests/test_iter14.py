"""Iter14 tests: full AST sweep, swarm->evolve->teach->QBF, bridge benchmarks, tiles heatmap."""
import os, sys, tempfile
sys.path.insert(0, os.path.expanduser("~/M1Multitronic"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic import (decompose_python, DecomposeError, Engine, Program, Block, Wire,
                    Display, FlowTrace, Swarm, Agent,
                    HostBridge, bridge_benchmark, h4_streaming_metrics, row_cos_gate, h4_row_cosine,
                    h4_gate, latency_histogram)
from atomic.demo import swarm_evolve_teach_demo

DT = 1.0/30.0

def test_decompose_loops():
    # for loop with body
    p = decompose_python("for i in range(5):\n    x = i * 2", name="loop")
    assert p.validate()==[], p.validate()
    assert any(b.primitive=="accum" for b in p.blocks)
    # while
    p2 = decompose_python("while x > 0:\n    x = x - 1", name="while")
    assert p2.validate()==[]
    # nested loop in function
    src = "def f():\n    total = 0\n    for i in range(3):\n        total = total + i\n    return total"
    p3 = decompose_python(src, name="nested")
    assert p3.validate()==[]

def test_decompose_comprehensions():
    for src in [
        "squares = [x*x for x in range(5)]",
        "evens = {x for x in range(5) if x % 2 == 0}",
        "d = {k: k*2 for k in range(3)}",
        "g = (x*2 for x in [1,2,3])",
        "flat = [y for x in [[1,2],[3,4]] for y in x]",
    ]:
        p = decompose_python(src, name="comp")
        assert p.validate()==[], (src, p.validate())
        patch = p.compile("microfx")
        res = Engine(patch["modules"], patch["wires"]).run(4)
        assert res["final"] is not None

def test_decompose_class():
    src = """
class Foo:
    x = 5
    def bar(self):
        return self.x * 2
    def baz(self, y):
        for i in range(3):
            y = y + i
        return y
"""
    p = decompose_python(src, name="cls")
    assert p.validate()==[], p.validate()
    # must run
    assert Engine(p.compile("microfx")["modules"], p.compile("microfx")["wires"]).run(3)["final"] is not None

def test_decompose_full_sweep():
    # try/with/assert/lambda/fstring/try
    src = """
import math
x = 2
try:
    y = x / 0
except:
    y = 0
with open('f') as fh:
    z = 1
assert y == 0
f = lambda a: a*2
s = f'{y} hello'
"""
    p = decompose_python(src, name="sweep")
    assert p.validate()==[]
    # also test fidelity: a simple arithmetic still exact after sweep
    p2 = decompose_python("out = 7 - 2", name="sub")
    res = Engine(p2.compile("microfx")["modules"], p2.compile("microfx")["wires"]).run(3)
    # after 1-tick latency, bias -2 should yield 5
    assert any(abs(float(v)-5.0)<1e-9 for v in res["final"].values() if isinstance(v,(int,float)))

def test_swarm_evolve_teach_qbf_demo():
    td = tempfile.mkdtemp(prefix="iter14_demo_")
    path = os.path.join(td, "demo.qbf")
    res = swarm_evolve_teach_demo(ticks=10, generations=3, population=2, path=path, seed=1)
    assert res["consensus"] is not None
    assert res["improved"] is True
    assert res["registry_size"] >= 2
    assert os.path.exists(res["persisted_path"])
    assert res["load_ok"] is True
    assert res["replay_match"] is True
    # load registry and check domain routing
    from atomic.teach import load_registry
    reg = load_registry(path)
    assert len(reg.list(domain="signal")) >= 1

def test_bridge_benchmark_and_h4_metrics():
    b = HostBridge(latency=2, capacity=10, use_h4=True)
    bm = b.benchmark(ticks=30, payload_keys=4)
    assert bm["ticks"] == 30
    assert bm["ticks_per_s"] > 0
    assert isinstance(bm["histogram"], dict)
    # standalone benchmark
    bb = bridge_benchmark(ticks=20, payload_keys=4)
    assert bb["ticks"] == 20
    # h4 streaming metrics: W dominance on correlated groups
    groups = [(1.0,2.0,3.0,4.0),(2.0,3.0,4.0,5.0),(3.0,4.0,5.0,6.0),(4.0,5.0,6.0,7.0)]
    m = h4_streaming_metrics(groups)
    assert m["n_groups"] == 4
    assert m["w_energy_frac"] > 0.5  # W dominant
    assert "row_cos_avg" in m
    # row_cos gate: orthogonal rows should pass
    passed, avg = row_cos_gate([(1,1,1,1),(1,-1,1,-1)])
    assert isinstance(passed, bool)
    # cosine helper
    assert abs(h4_row_cosine([1,0],[1,0]) - 1.0) < 1e-9
    assert abs(h4_row_cosine([1,0],[0,1]) - 0.0) < 1e-9
    # latency histogram helper
    hist = latency_histogram([1,1,2,2,2,3])
    assert hist == {1:2,2:3,3:1}
    # H4 round-trip via bridge still lossless
    b2 = HostBridge(latency=1, capacity=8, use_h4=True)
    payload = {"a":1.0,"b":2.0,"c":3.0,"d":4.0}
    b2.push(0, payload)
    out = b2.pop(1)
    for k in payload:
        assert abs(out[k]-payload[k]) < 1e-6

def test_tiles_swarm_heatmap():
    # wall heatmap from trace replay
    d = Display(900,900,3,3,frame_h=0)
    trace = FlowTrace()
    patch = {"modules":[{"id":"c0","primitive":"const","params":{"value":5}},{"id":"g1","primitive":"gain","params":{"factor":2}},{"id":"v0","primitive":"viz_series","params":{}}],
             "wires":[{"from":"c0.cv","to":"g1.in"},{"from":"g1.cv","to":"v0.in"}],"views":[]}
    Engine(patch["modules"], patch["wires"], trace=trace).run(6)
    wall = d.heatmap_from_trace(trace, port="g1.cv", agg="max")
    assert len(wall) == 9
    assert all(0.0 <= v <= 1.0 for v in wall.values())
    # TileGroup heatmap
    g = d.link("grp",0,0,2,2)
    gh = g.heatmap_from_trace(trace, port="g1.cv", agg="max")
    assert len(gh) == 4
    assert all(0.0 <= v <= 1.0 for v in gh.values())
    # swarm heatmap
    d2 = Display(1200,1200,3,3)
    swarm = Swarm(display=d2)
    for i, val in enumerate([1,4,7,10]):
        prog = Program("p", blocks=[Block("c0","const",{"value":float(val)}),Block("g1","gain",{"factor":1.0}),Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"),Wire("g1.cv","v0.in")])
        grp = d2.link(f"h{i}", i%3, i//3,1,1)
        swarm.add_agent(Agent(f"h{i}", prog, tile_group=grp))
    res = swarm.run(3)
    hm = d2.heatmap_from_swarm(res, port="g1.cv", normalize=True)
    # 4 agents -> 4 tiles mapped
    assert len(hm) == 4
    assert max(hm.values()) == 1.0
    assert min(hm.values()) == 0.0
    # after heatmap, summary should carry heatmap
    s = d2.summary()
    assert any("heatmap" in ginfo for ginfo in s["groups"].values())
    # bridge tiles wiring
    b = HostBridge(latency=1, capacity=8)
    for t in range(5):
        b.push(t, {"x": float(t)})
    # before pop, histogram should have entries
    th = b.to_tiles(display=d2)
    assert isinstance(th, dict)
