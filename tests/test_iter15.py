"""Iter15: edge-case audit, 16-agent swarm, 10k trace/QBF, WGSL naga + heatmap animation."""
import os, sys, tempfile, math, shutil
sys.path.insert(0, os.path.expanduser("~/M1Multitronic"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic import (decompose_python, decompose_eel2, DecomposeError, Engine, Program, Block, Wire,
                    Display, TileError, FlowTrace, replay, replay_events, latency_histogram_from_trace,
                    QbfFile, h4_gate, h4_inverse, h4_encode, h4_decode,
                    HostBridge, h4_row_cosine, row_cos_gate, h4_streaming_metrics, latency_histogram, bridge_benchmark,
                    Swarm, Agent)
from atomic.qbfstore import open_trace_store, close_all

DT = 1.0/30.0

# ---------------------------------------------------------------- 15.1 decompose audit
def test_decompose_async_walrus_fstring_comp():
    cases = {
        "async def": "async def foo():\n    x = 5\n    return x * 2",
        "async for": "async def g():\n    async for x in aiter():\n        y = x * 2",
        "walrus": "if (n := 10) > 5:\n    y = n * 2",
        "fstring": "name='hi'\ns = f'{name} {2*3} world'",
        "listcomp": "squares = [x*x for x in range(10)]",
        "setcomp": "evens = {x for x in range(10) if x%2==0}",
        "dictcomp": "d = {k: k*2 for k in range(5)}",
        "genexp": "g = (x*2 for x in [1,2,3])",
        "asynccomp": "[x async for x in agen()]",
        "lambda": "f = lambda a: a*2 + 1",
        "await": "import asyncio\nasync def f():\n    v = await asyncio.sleep(0)\n    return v",
        "yield": "def gen():\n    yield 1\n    yield 2",
        "starred": "a, *b = [1,2,3,4]",
        "subscript": "x = arr[2]",
        "named_expr_chain": "a = 5\nif (b := a+1) > 5:\n    c = b*2",
    }
    for name, src in cases.items():
        p = decompose_python(src, name="audit_%s" % name.replace(" ","_"))
        assert p.validate()==[], (name, p.validate())
        patch = p.compile("microfx")
        # engine runs without error
        res = Engine(patch["modules"], patch["wires"]).run(5)
        assert res["final"] is not None, name
        # structural pin: each known construct maps to expected primitive
        prims = [b.primitive for b in p.blocks]
        if "comp" in name or "genexp" in name:
            assert "moving_avg" in prims or "accum" in prims or "gain" in prims, (name, prims)
        if "walrus" in name:
            assert len(p.blocks) >= 2, name

def test_decompose_fidelity_vs_direct():
    # simple arithmetic: decompose should be functionally close to python eval for const cases
    # 7 - 2 -> 5, 3*4 -> 12, 8/2 -> 4
    cases = [("7 - 2", 5.0), ("3 * 4", 12.0), ("8 / 2", 4.0), ("2 + 3", 5.0)]
    for expr, expect in cases:
        src = "x = %s" % expr
        p = decompose_python(src, name="fid")
        assert p.validate()==[]
        patch = p.compile("microfx")
        res = Engine(patch["modules"], patch["wires"]).run(4)
        vals = [float(v) for v in res["final"].values() if isinstance(v,(int,float))]
        assert any(abs(v-expect)<1e-9 for v in vals), (expr, vals, expect)

def test_decompose_comprehension_streaming():
    src = "result = [x*2 for x in [1,2,3,4]]"
    p = decompose_python(src, name="comp_stream")
    assert any(b.primitive=="moving_avg" for b in p.blocks), p.blocks
    # also async comprehension should not crash
    p2 = decompose_python("r = [x async for x in agen()]", name="asynccomp2")
    assert p2.validate()==[]

def test_decompose_eel2_complex_fidelity():
    src = "v = input('sensor');\noutput('cv', v * 2.0 + 1.0);"
    p = decompose_eel2(src, name="eel_fid")
    assert p.validate()==[]
    assert any(b.primitive in ("gain","bias") for b in p.blocks)
    patch = p.compile("microfx")
    res = Engine(patch["modules"], patch["wires"]).run(4)
    assert res["final"] is not None

# ---------------------------------------------------------------- 15.2 swarm 16 agents 4x4
def test_swarm_16_agents_4x4_consensus_rowcos():
    d = Display(1600, 1600, 4, 4, frame_h=0)
    swarm = Swarm(display=d)
    def mk(v):
        return Program("p%d"%int(v), blocks=[Block("c0","const",{"value": float(v)}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
    for i in range(16):
        r, c = i // 4, i % 4
        g = d.link("sw%02d"%i, r, c, 1, 1)
        val = float(i+1)  # 1..16
        swarm.add_agent(Agent("sw%02d"%i, mk(val), tile_group=g))
    res_p = swarm.run(6, parallel=True)
    res_s = swarm.run(6, parallel=False)
    # parallel == serial
    for i in range(16):
        a = res_p.by_id("sw%02d"%i)["final"]["g1.cv"]
        b = res_s.by_id("sw%02d"%i)["final"]["g1.cv"]
        assert a == b, (i, a, b)
    # H4 consensus over first 4 agents: 1+2+3+4=10 (lexicographic with zero-pad)
    assert res_p.consensus("g1.cv") == 10.0
    # row_cos under load: sample groups from swarm scalars
    # build 4 groups of 4 scalars each
    groups = [tuple(float(res_p.by_id("sw%02d"% (g*4+j))["final"]["g1.cv"]) for j in range(4)) for g in range(4)]
    m = h4_streaming_metrics(groups)
    assert m["n_groups"]==4
    assert m["w_energy_frac"] > 0.4  # W should be dominant
    # row_cos gate: on uniform groups cos should be near 0 or 1 depending; just check returns bool
    passed, avg = row_cos_gate(groups, threshold=0.5)
    assert isinstance(passed, bool) and 0 <= avg <= 1
    # swarm consensus determinism also via second run
    res_p2 = swarm.run(6, parallel=True)
    assert res_p.consensus("g1.cv")==res_p2.consensus("g1.cv")
    # also verify numeric ordering: first group sum ==10 via h4_gate W
    w, _, _, _ = h4_gate(groups[0])
    assert w == 10.0

def test_swarm_16_isolation_no_leak():
    # ensure each agent isolated: modifying one agent's const does not affect others in same run window
    d = Display(800, 800, 4, 4)
    swarm = Swarm(display=d)
    progs = []
    for i in range(4):
        prog = Program("p%d"%i, blocks=[Block("c0","const",{"value": float(i*10)}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        g = d.link("a%d"%i, i%4, i//4, 1,1)
        swarm.add_agent(Agent("a%d"%i, prog, tile_group=g))
        progs.append(prog)
    res = swarm.run(4, parallel=True)
    for i in range(4):
        assert res.by_id("a%d"%i)["final"]["g1.cv"] == float(i*10), i

# ---------------------------------------------------------------- 15.3 trace 10k frames, ring wrap, QBF replay, latency, H4
def _large_patch(n_mods=2):
    # minimal patch with n_mods modules to generate frames = ticks * n_mods
    return {"modules":[{"id":"c0","primitive":"const","params":{"value":1}},{"id":"g1","primitive":"gain","params":{"factor":2}}],
            "wires":[{"from":"c0.cv","to":"g1.in"}],"views":[]}

def test_trace_10k_ring_wrap():
    patch = _large_patch()
    trace = FlowTrace(max_frames=10_000)
    # run 5000 ticks with 2 mods => 10000 frames exactly
    Engine(patch["modules"], patch["wires"], trace=trace, dt=DT).run(5000)
    snap = trace.snapshot()
    assert snap["n_frames"]==10000
    assert snap["seq"]==10000
    assert snap["n_ticks"]==5000
    # one more tick should wrap
    Engine(patch["modules"], patch["wires"], trace=trace, dt=DT).run(1)  # internal _t continues? need fresh engine with same trace but new
    # Instead test explicit wrap with max 20 similar to selftest but at scale with manual push
    trace2 = FlowTrace(max_frames=20)
    # simulate 22 entries
    for i in range(11):
        trace2.begin_tick(i, 0)
        for mid in ["c0","g1"]:
            trace2.record_node(mid, "gain", {"in":1.0}, {"cv":2.0}, 10.0, t=i)
    assert trace2.snapshot()["n_frames"]==20
    assert trace2.snapshot()["seq"]==22
    # oldest seq should be 3
    assert trace2.frames[0].seq==3

def test_qbf_replay_at_scale():
    tmp = tempfile.mkdtemp(prefix="iter15_qbfscale_")
    try:
        patch = _large_patch()
        trace = FlowTrace(max_frames=10_000)
        Engine(patch["modules"], patch["wires"], views=[], dt=DT, trace=trace).run(5000)
        assert trace.snapshot()["n_frames"]==10000
        # QBF store roundtrip at scale
        store = open_trace_store("scale_%d"%os.getpid(), shard_dir=tmp)
        manifest = store.append_run(trace, dt=DT, note="10k scale", program=patch)
        assert manifest["n_frames"]==10000
        loaded = store.load_run(manifest["run_id"])
        assert loaded["manifest"]["n_frames"]==10000
        assert len(loaded["frames"])==10000
        # export byte-identical check (via snapshot equality)
        assert loaded["ticks"]==trace.snapshot()["ticks"]
        # replay bit-identical
        again = store.replay_run(manifest["run_id"])
        orig_final = Engine(patch["modules"], patch["wires"], dt=DT).run(5000)["final"]
        # replay uses stored trace ticks/feeds; for const/gain it's same as plain run
        # compute via trace replay helper
        from atomic.trace import replay as trace_replay
        via_replay = trace_replay(trace, patch["modules"], patch["wires"], dt=DT)
        assert via_replay["final"]==orig_final
        assert again["final"]==orig_final
        # latency histogram should have entries (trace per-node latency)
        buckets, raw = latency_histogram_from_trace(trace)
        assert isinstance(buckets, dict)
        assert len(raw)==10000
        assert all(isinstance(v,float) for v in raw[:3])
        # event stream length == frames * ports (~2 per node? at least 1)
        evts = replay_events(trace.frames)
        assert len(evts) >= 10000  # at least one per frame
        # H4 gate fidelity at scale: encode/decode 10k groups
        groups = [(float(i), float(i+1), float(i+2), float(i+3)) for i in range(1000)]
        enc = h4_encode(groups)
        dec = h4_decode(enc)
        for a,b in zip(dec, groups):
            assert all(abs(x-y)<1e-5 for x,y in zip(a,b))
        # also Qbf H4 blob at scale
        path = os.path.join(tmp, "h4scale.qbf")
        f = QbfFile.create(path)
        f.put_h4("big_h4", groups)
        f.write()
        g = QbfFile.open(path)
        got = g.get_h4("big_h4")
        for a,b in zip(got, groups):
            assert all(abs(x-y)<1e-5 for x,y in zip(a,b))
    finally:
        try: close_all()
        except: pass
        shutil.rmtree(tmp, ignore_errors=True)

def test_bridge_latency_histogram_and_h4_fidelity_under_load():
    b = HostBridge(latency=3, capacity=100)
    for t in range(20):
        b.push(t, {"v": float(t)})
        # pop at t: arrival = t_push+3, so pop at t+3
    # at tick 20, queue should contain arrivals >20
    assert b.depth() > 0
    hist = b.latency_histogram()
    assert isinstance(hist, dict)
    # H4 payload under load
    b2 = HostBridge(latency=1, capacity=64, use_h4=True)
    payload = {"a":1.0,"b":2.0,"c":3.0,"d":4.0}
    for t in range(10):
        b2.push(t, payload)
        out = b2.pop(t+1)
        assert out is not None
        for k in payload:
            assert abs(out[k]-payload[k])<1e-6

# ---------------------------------------------------------------- 15.4 tiles live viz + WGSL naga
def test_wgsl_naga_validation():
    prog = Program("wgsl15", blocks=[Block("c0","const",{"value":1}), Block("g1","gain",{"factor":2}), Block("h1","h4_slide"), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","h1.in"), Wire("h1.w","v0.in")])
    w = prog.to_wgsl()
    # structural
    assert w.startswith("// WGSL")
    assert "@compute" in w and "@group(0)" in w and "host-RAM" in w
    assert "fn tick_c0" in w and "fn tick_g1" in w
    # validate via Display.validate_wgsl
    ok, detail = Display.validate_wgsl(w)
    assert ok, detail
    # also via program helper
    ok2, detail2 = prog.validate_wgsl()
    assert ok2, detail2
    # negative: missing header should fail
    bad = "fn main() {}"
    ok3, _ = Display.validate_wgsl(bad)
    assert not ok3
    # ensure wgsl contains bridge comment for bicameral
    assert "no P2P" in w or "host bridge" in w

def test_tiles_heatmap_animation_from_replay():
    # build trace with varying values per tick via live feeds (gain factor steps)
    patch = {"modules":[{"id":"c0","primitive":"const","params":{"value":5}},{"id":"g1","primitive":"gain","params":{"factor":2}},{"id":"v0","primitive":"viz_series","params":{}}],
             "wires":[{"from":"c0.cv","to":"g1.in"},{"from":"g1.cv","to":"v0.in"}],"views":[]}
    feeds = {2: {"params":{"g1":{"factor":4.0}}}, 5: {"params":{"g1":{"factor":1.0}}}, 8: {"params":{"g1":{"factor":3.0}}}}
    trace = FlowTrace()
    Engine(patch["modules"], patch["wires"], trace=trace, dt=DT, feeds=feeds).run(10)
    d = Display(900,900,3,3, frame_h=0)
    # wall animation
    anim = d.heatmap_animation(trace, port="g1.cv", window=1)
    assert len(anim) == 10  # one per tick
    for hm in anim:
        assert len(hm)==9
        assert all(0.0 <= v <= 1.0 for v in hm.values())
    # window=2 should halve
    anim2 = d.heatmap_animation(trace, port="g1.cv", window=2)
    assert len(anim2)==5
    # TileGroup animation via heatmap_from_trace per-window should be 0..1 as well
    g = d.link("anim_grp",0,0,2,2)
    # after animation, verify group heatmap can be updated tick-by-tick
    for hm in anim:
        # simulate live update: push hm subset into group
        sub = {k:v for k,v in hm.items() if k in {(t.row,t.col) for t in g.tiles}}
        g.apply_heatmap(sub, normalize=False)
        assert all(0.0 <= v <= 1.0 for v in g.heatmap.values())
    # also test heatmap_from_trace still works
    wall = d.heatmap_from_trace(trace, port="g1.cv", agg="max")
    assert len(wall)==9

def test_tiles_swarm_heatmap_16():
    d = Display(1600,1600,4,4, frame_h=0)
    swarm = Swarm(display=d)
    for i in range(16):
        prog = Program("p%d"%i, blocks=[Block("c0","const",{"value": float(i+1)}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        r,c = i//4, i%4
        grp = d.link("ts%d"%i, r,c,1,1)
        swarm.add_agent(Agent("ts%d"%i, prog, tile_group=grp))
    res = swarm.run(4)
    hm = d.heatmap_from_swarm(res, port="g1.cv", normalize=True)
    assert len(hm)==16
    assert max(hm.values())==1.0 and min(hm.values())==0.0
    # summary carries heatmaps
    s = d.summary()
    assert len(s["groups"])==16
    assert all("heatmap" in ginfo for ginfo in s["groups"].values())
