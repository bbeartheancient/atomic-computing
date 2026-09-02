"""Evolve tests (goal 7): self-improvement loop over params."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/M1Multitronic"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic import Program, Block, Wire
from atomic.evolve import Evolver

def _prog(factor=1.0):
    return Program("tun", blocks=[Block("c0","const",{"value":5}), Block("g1","gain",{"factor": factor}), Block("v0","viz_series")],
                   wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])

def test_evolve_improves_toward_target():
    # target g1.cv == 10 (c0=5 * factor 2)
    def fitness(final):
        return -abs(float(final.get("g1.cv",0))-10.0)
    base = _prog(factor=1.0)
    ev = Evolver(base, fitness, seed=0, ticks=5)
    start = ev.best_score
    assert start == -5.0
    ev.run(20)
    assert ev.best_score > start  # strictly better after 20 gens
    # best factor should be closer to 2
    assert abs(float(ev.best.blocks[1].params["factor"])-2.0) < abs(1.0-2.0)

def test_evolve_determinism():
    def fitness(final): return float(final.get("g1.cv",0))
    base = _prog(factor=1.0)
    ev1 = Evolver(base, fitness, seed=42, ticks=5)
    ev1.run(15)
    ev2 = Evolver(base, fitness, seed=42, ticks=5)
    ev2.run(15)
    assert ev1.best.hash == ev2.best.hash
    assert ev1.best_score == ev2.best_score
    assert len(ev1.history)==16  # gen 0..15

def test_evolve_mutate_valid():
    base = _prog()
    ev = Evolver(base, lambda f: 0, seed=1)
    for _ in range(5):
        child = ev.mutate(base)
        assert child.validate()==[], child.validate()

def test_evolve_promote_to_registry():
    from atomic.teach import TeacherRegistry
    base = _prog()
    ev = Evolver(base, lambda f: float(f.get("g1.cv",0)), seed=7, ticks=3)
    ev.run(5)
    h = ev.promote_to_registry("evolved tuner", domain="general")
    assert isinstance(h, str) and len(h)==12
