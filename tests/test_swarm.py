"""Swarm tests (goal 10): parallel agents, tile exclusivity, determinism."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/M1Multitronic"))
sys.path.insert(0, os.path.expanduser("~/hoa64"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic import Program, Block, Wire, Swarm, Agent, Display, TileError
from atomic import h4_gate

def _prog(val):
    return Program("p", blocks=[Block("c0","const",{"value": float(val)}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")],
                   wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])

def test_swarm_parallel_consensus():
    d = Display(1200,1200,3,3)
    swarm = Swarm(display=d)
    for i, v in enumerate([1,2,3,4]):
        r,c = i%3, i//3
        g = d.link("a%d"%i, r,c,1,1)
        swarm.add_agent(Agent("a%d"%i, _prog(v), tile_group=g))
    res = swarm.run(3, parallel=True)
    assert len(res.results)==4
    assert res.consensus("g1.cv")==10.0  # 1+2+3+4
    # serial must match parallel
    res2 = swarm.run(3, parallel=False)
    for a in ["a0","a1","a2","a3"]:
        assert res.by_id(a)["final"]["g1.cv"]==res2.by_id(a)["final"]["g1.cv"]

def test_swarm_isolation_and_determinism():
    d = Display(900,900,3,3)
    swarm = Swarm()
    swarm.add_agent(Agent("x", _prog(5)))
    swarm.add_agent(Agent("y", _prog(7)))
    a = swarm.run(5)
    b = swarm.run(5)
    assert a.by_id("x")["final"]["g1.cv"]==5.0
    assert a.by_id("y")["final"]["g1.cv"]==7.0
    assert a.by_id("x")["final"]==b.by_id("x")["final"]

def test_swarm_tile_exclusivity():
    d = Display(1200,1200,4,4,frame_h=0)
    swarm = Swarm(display=d)
    g0 = d.link("g0",0,0,2,2)
    g1 = Display(1200,1200,4,4).link("g0",0,0,1,1)  # same name different display ok
    swarm.add_agent(Agent("a0", _prog(1), tile_group=g0))
    try:
        swarm.add_agent(Agent("a1", _prog(2), tile_group=g0))
        assert False, "duplicate tile group must raise"
    except ValueError:
        pass
    # duplicate agent id
    try:
        swarm.add_agent(Agent("a0", _prog(3)))
        assert False
    except ValueError:
        pass

def test_swarm_empty():
    swarm = Swarm()
    res = swarm.run(10)
    assert res.results==[]
