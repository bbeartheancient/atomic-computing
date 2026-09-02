"""Teach tests (goals 4/5): teacher registry + domain routing + from_description."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/M1Multitronic"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic import Program, Block, Wire, Engine
from atomic.teach import TeacherRegistry, REGISTRY, domain_vocab, DOMAINS
from atomic import from_description

def test_registry_seed_and_match():
    assert len(REGISTRY.examples) >= 3
    hit = REGISTRY.match("gated clock counter chart")
    assert hit is not None
    assert hit["program"].name == "gated_clock_counter"

def test_domain_routing():
    # spatial domain should prefer hadamard/wxyz
    p = REGISTRY.from_description("hadamard spatial wxyz 3d scope", domain="spatial")
    assert any(b.primitive=="h4_slide" for b in p.blocks)
    # medical domain
    p2 = REGISTRY.from_description("medical sensor threshold alarm", domain="medical")
    assert any(b.primitive=="threshold" for b in p2.blocks)

def test_from_description_synthesis():
    p = from_description("a clock that counts with smooth chart", domain="control")
    assert p.validate()==[], p.validate()
    # must run
    patch = p.compile("microfx")
    res = Engine(patch["modules"], patch["wires"]).run(10)
    assert res["final"] is not None

def test_program_from_description_classmethod():
    p = Program.from_description("gain scale audio", domain="signal")
    assert isinstance(p, Program)
    assert p.validate()==[], p.validate()

def test_register_and_retrieve():
    reg = TeacherRegistry()
    prog = Program("demo", blocks=[Block("c0","const",{"value":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","v0.in")], description="demo prog")
    h = reg.register("demo prog for testing", prog, domain="general")
    assert h
    hit = reg.match("demo prog for testing")
    assert hit is not None and hit["hash"]==h

def test_domain_vocab():
    v = domain_vocab("audio")
    assert "hadamard" in v
    assert domain_vocab("unknown")=={}

def test_synthesize_fallback_empty_keywords():
    reg = TeacherRegistry()
    p = reg.synthesize("zzzzz unknown words")
    assert p.validate()==[], p.validate()
