"""Teach tests (goals 4/5/6): teacher registry + domain routing + from_description + QBF round-trip."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic import Program, Block, Wire, Engine
from atomic.teach import TeacherRegistry, REGISTRY, domain_vocab, DOMAINS, save_registry, load_registry
from atomic import from_description


def test_registry_seed_and_match():
    assert len(REGISTRY.examples) >= 14, f"expected >=14 seeded examples, got {len(REGISTRY.examples)}"
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
    # audio domain
    p3 = REGISTRY.from_description("audio spectrum flux detector", domain="audio")
    assert any(b.primitive=="mdct_flux" for b in p3.blocks), f"got {[b.primitive for b in p3.blocks]}"
    # signal domain
    p4 = REGISTRY.from_description("signal moving average filter", domain="signal")
    assert any(b.primitive=="moving_avg" for b in p4.blocks), f"got {[b.primitive for b in p4.blocks]}"
    # control domain
    p5 = REGISTRY.from_description("clock bpm counter toggle", domain="control")
    assert any(b.primitive=="clock_bpm" for b in p5.blocks), f"got {[b.primitive for b in p5.blocks]}"


def test_domain_coverage():
    for dom in DOMAINS:
        examples = REGISTRY.list(domain=dom)
        assert len(examples) >= 2, f"domain {dom} needs >=2 examples, got {len(examples)}"


def test_from_description_synthesis():
    p = from_description("a clock that counts with smooth chart", domain="control")
    assert p.validate()==[], p.validate()
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
    assert "mdct" in v
    assert domain_vocab("unknown")=={}
    v_medical = domain_vocab("medical")
    assert "pulse" in v_medical or "ecg" in v_medical
    v_signal = domain_vocab("signal")
    assert "hysteresis" in v_signal


def test_synthesize_fallback_empty_keywords():
    reg = TeacherRegistry()
    p = reg.synthesize("zzzzz unknown words")
    assert p.validate()==[], p.validate()


def test_qbf_roundtrip_expanded():
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="teach_iter26_")
    try:
        reg = TeacherRegistry()
        p = Program("test_prog", blocks=[Block("c0","const",{"value":5}),Block("v0","viz_series")],
                    wires=[Wire("c0.cv","v0.in")], description="test qbf expanded")
        reg.register("test qbf expanded", p, domain="signal")
        reg.register("spatial h4 wxyz consensus", REGISTRY.match("hadamard spatial wxyz 3d scope")["program"], domain="spatial")
        path = os.path.join(tmp, "expanded.qbf")
        saved = reg.save_qbf(path=path)
        assert os.path.exists(saved), f"QBF not written: {saved}"
        loaded = TeacherRegistry.load_qbf(saved)
        assert len(loaded.examples) == len(reg.examples), f"got {len(loaded.examples)} != {len(reg.examples)}"
        # domain routing on loaded
        hit = loaded.match("test qbf expanded", domain="signal")
        assert hit is not None and hit["program"].name == "test_prog"
        hit2 = loaded.match("spatial h4 wxyz", domain="spatial")
        assert hit2 is not None
        # each domain that was registered has at least one example
        registered_domains = set(e["domain"] for e in reg.examples)
        for dom in registered_domains:
            lst = loaded.list(domain=dom)
            assert len(lst) >= 1, f"domain {dom} missing after round-trip"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_expanded_keyword_routing():
    # hadamard -> h4_slide across domains
    for dom in ["spatial", "audio"]:
        p = REGISTRY.from_description("hadamard gate wxyz scope", domain=dom)
        assert any(b.primitive=="h4_slide" for b in p.blocks), f"{dom}: {[b.primitive for b in p.blocks]}"
    # threshold -> threshold across medical/signal
    for dom in ["medical", "signal"]:
        p = REGISTRY.from_description("sensor threshold gate alarm", domain=dom)
        assert any(b.primitive=="threshold" for b in p.blocks), f"{dom}: {[b.primitive for b in p.blocks]}"
    # smooth -> smooth
    # smooth -> smooth via synthesize path (use fresh registry so seed doesn't match)
    reg = TeacherRegistry()
    p = reg.synthesize("sierra november oscar smooth filter", domain="signal")
    assert any(b.primitive == "smooth" for b in p.blocks), \
        f"smooth synthesize: {[b.primitive for b in p.blocks]}"
    # moving average -> moving_avg
    p2 = reg.synthesize("tango uniform victor moving average", domain="signal")
    assert any(b.primitive == "moving_avg" for b in p2.blocks), \
        f"moving_avg synthesize: {[b.primitive for b in p2.blocks]}"
    # divider -> toggle
    p3 = reg.synthesize("whiskey xray yankee clock divider sequencer", domain="control")
    assert any(b.primitive == "clock_bpm" for b in p3.blocks), \
        f"divider synthesize: {[b.primitive for b in p3.blocks]}"


def test_teach_example_runs():
    # Every seeded example must compile and run without error
    failures = []
    for e in REGISTRY.examples:
        p = e["program"]
        try:
            errs = p.validate()
            if errs:
                failures.append((e["hash"], "validate", errs))
                continue
            patch = p.compile("microfx")
            res = Engine(patch["modules"], patch["wires"]).run(10)
            if res["final"] is None:
                failures.append((e["hash"], "run", "final is None"))
        except Exception as ex:
            failures.append((e["hash"], "exception", str(ex)))
    assert not failures, f"seeded examples failed: {failures[:3]}"
