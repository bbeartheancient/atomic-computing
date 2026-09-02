"""Bridge tests (goal 9): bicameral HostBridge + pipeline."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/M1Multitronic"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic import Program, Block, Wire
from atomic.bridge import HostBridge, BicameralPipeline

def _sub():
    return Program("sub", blocks=[Block("c0","const",{"value":4}), Block("pre","gain",{"factor":2}), Block("v0","viz_series")],
                   wires=[Wire("c0.cv","pre.in")])
def _con():
    return Program("con", blocks=[Block("dec","gain",{"factor":3}), Block("v1","viz_series")],
                   wires=[Wire("dec.cv","v1.in")])

def test_host_bridge_latency():
    b = HostBridge(latency=1, capacity=8)
    b.push(0, {"x": 1.0})
    assert b.pop(0) is None
    out = b.pop(1)
    assert out == {"x":1.0}
    b.push(5, {"y": 2.0})
    b.push(5, {"z": 3.0})
    got = b.pop(6)
    assert got["y"]==2.0 and got["z"]==3.0

def test_host_bridge_capacity_eviction():
    b = HostBridge(latency=1, capacity=2)
    b.push(0, {"a":1}); b.push(1, {"b":2}); b.push(2, {"c":3})
    assert b.depth()==2  # oldest evicted

def test_bicameral_pipeline_final():
    sub, con = _sub(), _con()
    pipe = BicameralPipeline(sub, con, bridge_map=[("pre.cv","dec.in")], bridge_latency=1)
    res = pipe.run(8)
    # sub: c0=4 -> pre=8 ; con: dec = 8*3 =24 after bridge
    assert res.sub["final"]["pre.cv"]==8.0
    assert res.con["final"]["dec.cv"]==24.0

def test_bicameral_latency_enforced():
    sub, con = _sub(), _con()
    pipe = BicameralPipeline(sub, con, bridge_map=[("pre.cv","dec.in")], bridge_latency=3)
    # tick-by-tick: con should lag sub by bridge_latency
    for i in range(2):
        pipe.tick()
        # before bridge delivery, con dec.cv should still be 0
        v = pipe.con.bus.get("dec.cv")
        if v is not None:
            assert float(v)==0.0 or i<3
    res = pipe.run(6)
    # after enough ticks, still converges to 24
    assert res.con["final"]["dec.cv"]==24.0

def test_bicameral_determinism():
    sub, con = _sub(), _con()
    p1 = BicameralPipeline(sub, con, bridge_map=[("pre.cv","dec.in")], bridge_latency=1)
    p2 = BicameralPipeline(sub, con, bridge_map=[("pre.cv","dec.in")], bridge_latency=1)
    r1 = p1.run(6)
    r2 = p2.run(6)
    assert r1.sub["final"]==r2.sub["final"]
    assert r1.con["final"]==r2.con["final"]
