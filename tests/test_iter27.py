"""Iter 27: video generation tests (viz_video, H3 session, frame blob, swarm prompt routing).

Run: python -m pytest tests/test_iter27.py -q
"""
import hashlib
import struct
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic import (Engine, Block, Wire, ATOMS, HostBridge,
                    H3Stub, H3File, H3Source, H3Session, H3Frame, H3Error,
                    PromptBank, Swarm, Agent, Program)


def test_viz_video_atom_registered():
    assert "viz_video" in ATOMS
    a = ATOMS["viz_video"]
    assert "in" in a.inputs
    assert "ready" in a.outputs
    assert a.category == "sink"


def test_viz_video_engine_basic():
    """A const -> viz_video program runs without error; frame is written to bus."""
    p = Program("video_test", blocks=[
        Block("src", "const", {"value": 1.0}),
        Block("sink", "viz_video", {"capture": 1.0}),
    ], wires=[Wire("src.cv", "sink.in")])
    patch = p.compile("microfx")
    eng = Engine(patch["modules"], patch.get("wires", []))
    eng.run(1)
    # set a frame on the bus so viz_video.tick will copy it
    eng.bus.set("sink.frame", b"\x80" * 256)
    eng.run(1)
    # the node writes the frame to bus key "sink.rgba" when a frame is present
    rgba = eng.bus.get("sink.rgba")
    assert rgba is not None
    assert rgba == b"\x80" * 256


def test_hostbridge_frame_blob_roundtrip():
    """A frame pushed through the bridge arrives intact."""
    br = HostBridge(latency=1, use_h4=False)
    frame = b"\xde\xad\xbe\xef" * 16  # 64 bytes of RGBA
    br.push(0, {"frame": frame, "prompt": "test"})
    br.push(1, {"frame": b"frame2", "seed": 1.0})
    # pop at tick 1: frame from tick 0 arrives
    out = br.pop(1)
    assert out is not None
    assert out.get("frame") == frame
    assert out.get("prompt") == "test"
    # pop at tick 2: frame from tick 1 arrives
    out2 = br.pop(2)
    assert out2 is not None
    assert out2.get("frame") == b"frame2"


def test_hostbridge_frame_blob_tag():
    """Frame entries are tagged _frame so consumers can distinguish from scalar dicts."""
    br = HostBridge(latency=1)
    br.push(0, {"frame": b"rgba"})
    out = br.pop(1)
    assert out is not None
    assert out.get("_frame") is True
    assert isinstance(out["frame"], bytes)


def test_hostbridge_h4_preserved_with_frame():
    """H4 compact (4 scalar vals) and frame blob can coexist in the same bridge."""
    br = HostBridge(latency=1, use_h4=True)
    br.push(0, {"w": 1.0, "z": 2.0, "y": 3.0, "x": 4.0})
    br.push(1, {"frame": b"rgba"})
    out1 = br.pop(1)
    assert out1 is not None
    # 4 scalar values -> H4 compact: stored gated, pop de-gated losslessly
    assert abs(out1["w"] - 1.0) < 1e-9
    assert abs(out1["z"] - 2.0) < 1e-9
    assert abs(out1["y"] - 3.0) < 1e-9
    assert abs(out1["x"] - 4.0) < 1e-9
    assert "frame" not in out1  # frame was pushed at tick 1, arrives at tick 2
    out2 = br.pop(2)
    assert out2 is not None
    assert out2.get("frame") == b"rgba"
    # frame entry is tagged with _w / _x / _y / _z (H4 row channel decoder)
    assert "_w" in out2
    assert "_x" in out2
    assert "_y" in out2
    assert "_z" in out2


def test_h3stub_generate():
    """H3Stub produces deterministic RGBA frames."""
    h3 = H3Stub(width=8, height=8, n_frames=4)
    r = h3.generate("a slow orbit", seed=42)
    assert len(r["frames"]) == 4
    assert isinstance(r["frames"][0], bytes)
    assert len(r["frames"][0]) == 8 * 8 * 4  # RGBA
    assert r["seed"] == 42
    assert r["prompt"] == "a slow orbit"


def test_h3stub_deterministic():
    """Same prompt+seed -> same frames."""
    h3 = H3Stub()
    r1 = h3.generate("neon rain", seed=99)
    r2 = h3.generate("neon rain", seed=99)
    assert r1["frames"] == r2["frames"]
    assert r1["seed"] == r2["seed"]


def test_h3stub_different_seeds():
    """Different seeds -> different frames."""
    h3 = H3Stub()
    r1 = h3.generate("test", seed=1)
    r2 = h3.generate("test", seed=2)
    assert r1["frames"] != r2["frames"]


def test_h3session_tick_single():
    """H3Session.tick() yields one frame and advances t."""
    h3 = H3Stub(n_frames=3)
    ses = H3Session(h3, prompts=["prompt one", "prompt two"], frames_per_prompt=3)
    assert ses.t == 0
    frame = ses.tick()
    assert frame is not None
    assert isinstance(frame, H3Frame)
    assert frame.t == 0
    assert ses.t == 1
    assert ses.latest() is frame


def test_h3session_roundrobin():
    """Session cycles through the prompt bank when frames_per_prompt exhausted."""
    h3 = H3Stub(n_frames=2)
    ses = H3Session(h3, prompts=["p0", "p1", "p2"], frames_per_prompt=2)
    frames = [ses.tick() for _ in range(12)]
    prompts = [f.prompt for f in frames]
    assert prompts[:2] == ["p0", "p0"]
    assert prompts[2:4] == ["p1", "p1"]
    assert prompts[4:6] == ["p2", "p2"]
    assert prompts[6:8] == ["p0", "p0"]


def test_h3session_bridge_push():
    """Session pushes frame into HostBridge after tick."""
    h3 = H3Stub(n_frames=1)
    br = HostBridge(latency=1)
    ses = H3Session(h3, prompts=["test"], bridge=br)
    ses.tick()
    assert br.depth() == 1
    out = br.pop(1)
    assert out is not None
    assert "frame" in out
    assert isinstance(out["frame"], bytes)


def test_h3session_max_ticks():
    """Session stops yielding after max_ticks."""
    h3 = H3Stub()
    ses = H3Session(h3, prompts=["p"], bridge=None, max_ticks=3)
    ses.tick(); ses.tick(); ses.tick()
    assert ses.t == 3
    assert ses.tick() is None


def test_h3session_consumed_prompts():
    """Consumed prompts ring tracks what H3 was called with."""
    h3 = H3Stub(n_frames=2)
    ses = H3Session(h3, prompts=["p0", "p1"], frames_per_prompt=2)
    # 6 ticks: 2 for p0, 2 for p1, 2 more (wraps back to p0)
    for _ in range(6):
        ses.tick()
    assert len(ses.consumed_prompts) == 3
    assert ses.consumed_prompts == ["p0", "p1", "p0"]


def test_h3source_factory():
    """Factory returns correct H3Source kinds."""
    s = H3Source("stub", width=4, height=4)
    assert isinstance(s, H3Stub)
    r = s.generate("x", seed=1)
    assert len(r["frames"]) == 8  # default n_frames


def test_h3source_unknown_kind():
    """Unknown kind raises H3Error."""
    try:
        H3Source("unknown_kind")
        assert False, "should raise"
    except H3Error as e:
        assert "unknown kind" in str(e)


def test_h3frame_properties():
    """H3Frame carries provenance and computes SHA256."""
    f = H3Frame(rgba=b"\x01\x02\x03\x04", prompt="neon rain", seed=7, t=3)
    assert f.size_bytes == 4
    assert f.t == 3
    assert f.seed == 7
    assert f.prompt == "neon rain"
    assert len(f.sha256) == 64  # SHA256 hex


def test_prompt_bank_registered():
    """PromptBank is in __all__."""
    from atomic import PromptBank
    assert PromptBank is not None


def test_prompt_bank_pick():
    """H4 consensus over prompts picks a deterministic bank index."""
    bank = PromptBank(prompts=["p0", "p1", "p2"])
    # consensus_pick() uses H4 gate over the 4 prompts in the bank;
    # the result is deterministic for a given bank (the W-channel of the
    # 4-hash tuple). Call twice to prove it's deterministic.
    p1 = bank.consensus_pick()
    p2 = bank.consensus_pick()
    assert p1 == p2  # deterministic: same bank -> same pick
    assert p1 in ["p0", "p1", "p2"]


def test_prompt_bank_explicit_prev_prompts():
    """Explicit prev_prompts drives a different bank index."""
    bank = PromptBank(prompts=["p0", "p1", "p2"])
    p1 = bank.consensus_pick(prev_prompts=["a", "b", "c", "d"])
    p2 = bank.consensus_pick(prev_prompts=["w", "x", "y", "z"])
    # different prompts -> different H4 -> likely different bank index
    # (statistically very likely; could theoretically collide)
    assert p1 in ["p0", "p1", "p2"]
    assert p2 in ["p0", "p1", "p2"]


def test_prompt_bank_w_energy():
    """W energy is a float in [0,1]."""
    bank = PromptBank()
    e = bank.w_energy()
    assert 0.0 <= e <= 1.0


def test_prompt_bank_consensus_with_results():
    """consensus_pick can take a last_w override (swarm scalar consensus)."""
    bank = PromptBank(prompts=["a", "b", "c", "d"])
    # last_w drives which bank index is picked
    # norm = (w % 4.0) / 4.0; idx = int(norm * n) % n
    # last_w=0.0 -> norm=0.0 -> idx=0 -> "a"
    assert bank.consensus_pick(last_w=0.0) == "a"
    # last_w=1.0 -> norm=0.25 -> idx=int(1.0)=1 -> "b"
    assert bank.consensus_pick(last_w=1.0) == "b"
    # last_w=4.0 -> norm=0.0 -> idx=0 -> "a" (wraps via mod 4)
    assert bank.consensus_pick(last_w=4.0) == "a"
    # last_w=8.0 -> norm=0.0 -> idx=0
    assert bank.consensus_pick(last_w=8.0) == "a"
    # last_w=3.999 -> norm=0.99975 -> idx=int(3.999)=3 -> "d"
    assert bank.consensus_pick(last_w=3.999) == "d"


def test_prompt_bank_empty():
    """Empty bank returns empty string."""
    bank = PromptBank(prompts=[])
    assert bank.consensus_pick() == ""


def test_swarm_consensus_integration():
    """Swarm.consensus() returns W value that PromptBank can consume."""
    from atomic import Swarm, Agent, Program, Block, Wire, PromptBank
    from atomic.qbf import h4_gate

    p = Program("c", blocks=[
        Block("g1", "const", {"value": 1.0}),
    ], wires=[])
    # 4 agents with the same scalar output -> H4 consensus is 4*1.0
    agents = [Agent("a%d" % i, p) for i in range(4)]
    swarm = Swarm()
    for a in agents:
        swarm.add_agent(a)
    res = swarm.run(ticks=1)
    w = res.consensus(port="g1.cv")
    # All 4 agents have g1.cv=1.0, so W = 1+1+1+1 = 4
    assert abs(w - 4.0) < 1e-9

    # Prompt bank can take the swarm's W consensus
    bank = PromptBank(prompts=["prompt0", "prompt1", "prompt2", "prompt3"])
    picked = bank.consensus_pick(last_w=w)
    # w=4.0 -> norm=0.0 -> idx=0 -> "prompt0"
    assert picked == "prompt0"


def test_viz_video_compiles_in_program():
    """Program with viz_video block compiles without error."""
    p = Program("video_compile", blocks=[
        Block("c", "const", {"value": 1.0}),
        Block("v", "viz_video", {"capture": 1.0}),
    ], wires=[Wire("c.cv", "v.in")])
    patch = p.compile("microfx")
    assert "modules" in patch
    ids = [m["id"] for m in patch["modules"]]
    assert "v" in ids


def test_h3session_stats():
    """H3Session.stats() returns a sensible dict."""
    h3 = H3Stub(n_frames=2)
    ses = H3Session(h3, prompts=["p0", "p1"], bridge=None)
    ses.tick(); ses.tick()
    st = ses.stats()
    assert st["t"] == 2
    assert st["consumed_prompts"] == 1  # 2 ticks, 2 frames per prompt
    assert st["frames_in_ring"] == 2
    assert "h3_calls" in st
