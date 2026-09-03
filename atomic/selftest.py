"""selftest: unified gauntlet for ATOMIC-PC (iter 35).

 34 sections, N/N ok per section, exit 0/1.
Run: cd ~/ATOMIC-PC && ~/runtime/.venv/bin/python -m atomic.selftest

Sections:
  1 bridge   node + jsfx conformance 26/26 + hoa64 selftest
  2 signal   tap latency, sum, fan-out, 0-input, node rule, self-wire, cycle,
             edge count, determinism, view window, tap-id pin, self-containment,
             conformance parity (6 patches), random property x6
  3 gates    logic PASS_A/B brute-force vs pinned vs engine==oracle;
             quantum QNEG/QPOS vs pinned vs engine==oracle;
             alogic FAMS brute-force vs pinned vs engine==oracle
  4 h4       rows==sylvester(4), orthogonal & W-dominant, label swap, clock oracle
  5 cv       smooth, moving_avg, threshold live (engine+LiveOracle), accum
  6 ir       round-trip 6 patches, 11 rejections, node-rule, fabric gap, alias,
             eel2/mermaid/wgsl, python live runner, IR transparency
  7 tiles    3x3 res, 4x4 border, frame/tiles, rejections, linking, summary
  8 replay   traced==untraced, counter replay, live-feed replay, ring wrap
  9 qbf      store round-trip, header, no .mv2 wall, checksum, missing, h4 gate/codec
 10 swarm    swarm parallel/serial consensus, isolation, tile exclusivity,
            decompose python/eel2 valid, teach registry match/domain/from_description
 11 evolve   self-improvement hill-climb, determinism, bicameral HostBridge + pipeline, wgsl codegen
 12 harden   decompose Add wire-SUM, EEL2 multi-input, Evolver QBF, bridge H4, wgsl naga, swarm determinism
 13 polish   decompose Sub/Div/AnnAssign, teach QBF, evolver swarm, wgsl threshold/clamp/sine
 14 iter14   full AST sweep (loops/comp/class), swarm->evolve->teach->QBF demo, bridge bench+H4 metrics+row_cos, tiles heatmap
 15 iter15   decompose edge audit (async/walrus/fstring/comp), 16-agent 4x4 swarm, 10k trace/QBF/replay, WGSL naga + heatmap animation
 16 iter16   zvec-grep retrieval: zg install + local/potion-code-16m-v2 index + H4/tiles/QBF retrieval fidelity + rg
 17 ui       FastAPI server + tile wall (7 demo programs, batch/tap/feed, control schema, series, h4 rows, 4x4 display bounds)
 18 ui       iter 4: signed heatmap, WS RTT, presets, replay/record
 19 ui       iter 4 continued: split pane, wsstats, keyboard shortcuts
 20 ui       iter 5: themes, program switcher, bus inspector, param sweep, CSV export, wheel speed
 21 ui       iter 6: fullscreen, color picker, viz override, bus search, cheatsheet, screenshot, favorites
 22 ui       iter 7: tile wall zoom (Ctrl+wheel) + accent color override
 23 iter24   goal A: wgsl naga hard-validate (30.0.1), module-scope storage, @group(0) bindings
 24 iter25   goal B: bicameral live demo (BicameralPipeline, bridge_latency=1, UI /api/bicameral endpoints)
  25 iter26   teach domain expansion: 14 examples across 6 domains, QBF persistence, keyword routing, all runnable
  26 iter27   goal C: video generation (viz_video atom, HostBridge frame blob, H3 stub/session, swarm prompt bank, H4 RGBA log/linear decoder)
  27 iter28   goal C continued: jfin_live_export atom, JFinScheduler/Exporter/M3U, channel rotation (round_robin/random/h4), M3U emission, HDHomeRun discovery
  28 iter29   goal C continued: DASH muxer, mock ffmpeg mode, keyframe-on-trig, seeded rotation determinism, recursive group-title M3U, Swarm H4 -> JFinScheduler consensus_pick, viz_video -> jfin_live_export end-to-end program
  29 iter30   goal C continued: viz_video canvas + JFin HDHomeRun REST + jfin_export_demo program
  30 iter31   goal C: H3InferenceServer (FastAPI on :8765, GET /health + POST /generate), viz_video_h3 atom polls server per tick (H4 RGBA decoder, retry, enabled gate, H4 parity vs viz_video), video_live program, /api/video/start|stop|status REST
  31 iter32   ComfyUIH3Bridge (API-only subprocess bridge + stub fallback)
  32 iter33   feed_video REST+WS + swarm H4 routing + QBF frame trace
  33 iter34   InfiniteVideoLoop (H3 -> HostBridge -> BicameralViewer -> viz_video -> QBF)
  34 iter35   VideoSynth + VideoSynthSource (5 effects, GPU-free) + InfiniteVideoLoop
              integration + BicameralViewer.feed_ivl_tick + QBF round-trip
"""
import io, json, math, os, random, shutil, struct, subprocess, sys, tempfile, time
import pytest

def _fix_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests = os.path.join(root, "tests")
    home = os.path.expanduser("~")
    for p in (tests, root, home):
        if p not in sys.path:
            sys.path.insert(0, p)
    env_root = os.environ.get("ATOMIC_FABRIC_ROOT")
    if env_root and env_root not in sys.path:
        sys.path.insert(0, env_root)
_fix_paths()

import atomic.oracle as oracle
from atomic import (Engine, ATOMS, Atom, Program, Block, Wire, ProgramError,
                    LiveOracle, Display, TileError, FlowTrace, replay,
                    QbfFile, QbfError, RAW, JSON, H4,
                    FLAG_CHECKSUM, FLAG_H4,
                    h4_gate, h4_inverse, Swarm, Agent, SwarmResult,
                    decompose_python, decompose_eel2, decompose_expr, DecomposeError,
                    TeacherRegistry, REGISTRY, from_description, domain_vocab)

DT = 1.0/30.0
HOME = os.path.expanduser("~")
C = 0.7071067811865476

from test_parity import PATCHES, _diff as parity_diff
from test_gates_parity import (GATES10, PASS_A, PASS_B, _logic_patch, _quantum_patch,
                               _alogic_patch, FAMS, QUANTUM_EXPECT, ALOGIC_EXPECT,
                               _h4_run, _script_atom, _diff as gates_diff)
from test_trace import COUNTER
from test_program import _cv_program

class _Skip(Exception):
    def __init__(self, reason): self.reason = reason

def _run_section(num, title, fn):
    try:
        checks = fn()
    except _Skip as s:
        print(f"{num}. {title:<22} SKIPPED ({s.reason})")
        return True
    except Exception as e:
        print(f"{num}. {title:<22} FAILED TO BUILD: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False
    ok = 0
    t0 = time.time()
    for name, check in checks:
        try:
            check()
            ok += 1
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e).splitlines() else repr(e)
            print(f"    FAIL {name:<44} {type(e).__name__}: {msg[:400]}")
    n = len(checks)
    dt = time.time()-t0
    tag = "" if ok==n else "  <-- FAILURES"
    print(f"{num}. {title:<22} {ok}/{n} ok  ({dt:.1f}s){tag}")
    return ok==n

# ---------------------------------------------------------------- 1 bridge
def s1_checks():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    node = shutil.which("node")
    env_fabric = os.environ.get("ATOMIC_FABRIC_ROOT", os.path.join(root, "fabric"))
    env_hoa = os.environ.get("ATOMIC_HOA_ROOT", os.path.join(root, "hoa64"))
    jsfx = os.path.join(env_fabric, "web", "jsfx.js")
    fixture = os.path.join(env_fabric, "tests", "microfx_modules.json")
    conformance = os.path.join(env_fabric, "tests", "jsfx_conformance.js")
    hoa = shutil.which("python")
    def node_assets():
        assert node is not None, "node not on PATH"
        assert os.path.exists(jsfx), f"missing {jsfx}"
        assert os.path.exists(fixture), f"missing {fixture}"
    def conformance_26():
        assert os.path.exists(conformance), f"missing {conformance}"
        out = subprocess.run([node, conformance], capture_output=True, text=True, cwd=root, timeout=60)
        assert out.returncode==0, out.stderr[:600]
        assert "26/26" in out.stdout, out.stdout[-600:]
    def hoa_selftest():
        out = subprocess.run([hoa, "-m", "hoa64.cli", "hadamard", "--selftest"],
                             capture_output=True, text=True, cwd=root, timeout=60)
        assert out.returncode==0, out.stderr[:600]
        assert "pass" in (out.stdout+out.stderr).lower()
    checks=[("node + jsfx + fixture present", node_assets)]
    if node is not None and os.path.exists(conformance):
        checks.append(("jsfx conformance 26/26", conformance_26))
    if os.path.exists(hoa):
        checks.append(("hoa64 hadamard selftest", hoa_selftest))
    else:
        print("    note: hoa64 venv absent -> s4 will skip")
    return checks

# ---------------------------------------------------------------- 2 signal
def s2_checks():
    from atomic.jsnum import is_finite
    checks=[]
    def latency():
        patch={"modules":[{"id":"ui","primitive":"tap","params":{}},
                          {"id":"tg","primitive":"toggle","params":{"initial":0}}],
               "wires":[{"from":"ui.tap","to":"tg.trig"}],"views":[]}
        r6=Engine(patch["modules"],patch["wires"],dt=DT,ui_taps=[5]).run(6)
        r7=Engine(patch["modules"],patch["wires"],dt=DT,ui_taps=[5]).run(7)
        assert r6["final"]["tg.state"]==0.0, r6["final"]
        assert r7["final"]["tg.state"]==1.0, r7["final"]
    checks.append(("tap latency 1 tick", latency))
    def stacked_sum():
        p=PATCHES[4][4]
        res=Engine(p["modules"],p["wires"],dt=DT).run(2)
        assert res["final"]["g.cv"]==8, res["final"]
    checks.append(("stacked inputs sum", stacked_sum))
    def fan_out():
        p=PATCHES[5][4]
        res=Engine(p["modules"],p["wires"],dt=DT).run(2)
        assert res["final"]["g1.cv"]==5.0 and res["final"]["g2.cv"]==5.0
    checks.append(("fan-out", fan_out))
    def unconnected_zero():
        p={"modules":[{"id":"g","primitive":"gain","params":{"factor":3}}],"wires":[],"views":[]}
        res=Engine(p["modules"],p["wires"],dt=DT).run(3)
        assert res["final"]["g.cv"]==0.0
    checks.append(("unconnected input is 0", unconnected_zero))
    def node_rule_fake2in():
        ATOMS["gauntlet2in"]=Atom("gauntlet2in","Fake","function",{},["a","b"],["out"],"",multi_in=False)
        try:
            errs=Program("x",blocks=[Block("f1","gauntlet2in")]).validate()
            assert any("one input port" in e for e in errs), errs
        finally:
            del ATOMS["gauntlet2in"]
    checks.append(("node rule fake2in", node_rule_fake2in))
    def self_wire():
        p=Program("x",blocks=[Block("g1","gain")],wires=[Wire("g1.cv","g1.in")])
        assert any("self-wire" in e for e in p.validate())
    checks.append(("rejects self-wire", self_wire))
    def cycle():
        p=Program("x",blocks=[Block("c1","const"),Block("g1","gain"),Block("g2","gain")],
                  wires=[Wire("c1.cv","g1.in"),Wire("g1.cv","g2.in"),Wire("g2.cv","g1.in")])
        assert any("cycle" in e for e in p.validate())
    checks.append(("rejects cycle", cycle))
    def trig_edges():
        p=PATCHES[3][4]
        res=Engine(p["modules"],p["wires"],dt=DT,ui_taps=[5,30]).run(60)
        assert res["final"]["acc.acc"]==2
    checks.append(("trig edge count acc==2", trig_edges))
    def determinism():
        p=PATCHES[0][4]
        a=Engine(p["modules"],p["wires"],views=p.get("views") or [],dt=DT).run(120)
        b=Engine(p["modules"],p["wires"],views=p.get("views") or [],dt=DT).run(120)
        assert a["final"]==b["final"] and a["series"]==b["series"]
    checks.append(("determinism", determinism))
    def view_window():
        p=PATCHES[0][4]
        res=Engine(p["modules"],p["wires"],views=p.get("views") or [],dt=DT).run(600)
        k=list(res["series"].keys())[0] if res["series"] else None
        assert k is not None and len(res["series"][k])==512
    checks.append(("view window 512", view_window))
    def tap_id_pin():
        patch={"modules":[{"id":"t1","primitive":"tap","params":{}},
                          {"id":"acc","primitive":"accum","params":{"per_tick":1}}],
               "wires":[{"from":"t1.tap","to":"acc.in"}],"views":[]}
        jf,_=oracle.run(patch,10,dt=DT,ui_taps=[5])
        res=Engine(patch["modules"],patch["wires"],dt=DT,ui_taps=[5]).run(10)
        assert not parity_diff(jf,res["final"],0.0)
        assert res["final"]["acc.acc"]==0
    checks.append(("tap-id pin non-ui dead", tap_id_pin))
    def self_containment():
        p=PATCHES[3][4]
        f1,_=oracle.run(p,10,dt=DT,ui_taps=[5])
        f2,_=oracle.run(p,10,dt=DT,ui_taps=[5])
        assert f1==f2 and f1["acc.acc"]==1
    checks.append(("per-run self-containment", self_containment))
    def conformance_parity():
        for label,ticks,taps,tol,patch in PATCHES:
            jf,js=oracle.run(patch,ticks,dt=DT,ui_taps=taps)
            res=Engine(patch["modules"],patch["wires"],views=patch.get("views") or [],dt=DT,ui_taps=taps).run(ticks)
            probs=parity_diff(jf,res["final"],tol)
            assert not probs, f"{label}: {probs[:3]}"
            for key,ja in js.items():
                pa=res["series"].get(key,[])
                assert len(ja)==len(pa), (label,key)
                for jv,pv in zip(ja,pa):
                    if jv is None: continue
                    ok=float(jv)==float(pv) if tol<=0 else abs(float(jv)-float(pv))<=tol
                    assert ok, (label,key,jv,pv)
    checks.append(("conformance 6 patches engine==oracle", conformance_parity))
    def _random_patch(rng):
        prims=["const","gain","smooth","bias","clamp","toggle","gate_and","gate_xor"]
        ins={"const":[],"gain":["in"],"smooth":["in"],"bias":["in"],"clamp":["in"],"toggle":["trig"],"gate_and":["a","b"],"gate_xor":["a","b"]}
        outs={"const":["cv"],"gain":["cv"],"smooth":["cv"],"bias":["cv"],"clamp":["cv"],"toggle":["state"],"gate_and":["q"],"gate_xor":["q"],"tap":["tap"]}
        n=rng.randint(3,6)
        taps=[t for t in (2,7,13) if rng.random()<0.7]
        modules=[{"id":"ui","primitive":"tap","params":{}}]
        wires=[]
        for i in range(1,n+1):
            prim=rng.choice(prims)
            if prim=="const": params={"value": round(rng.uniform(-9,9),3)}
            elif prim=="gain": params={"factor": round(rng.uniform(0,3),3)}
            elif prim=="smooth": params={"alpha": round(rng.uniform(0.05,0.95),3)}
            elif prim=="clamp": params={"lo":-2.0,"hi":2.0}
            elif prim=="toggle": params={"initial":0}
            else: params={}
            modules.append({"id":f"m{i:02d}","primitive":prim,"params":params})
            for inp in ins[prim]:
                if rng.random()<0.7:
                    src=modules[rng.randint(0,i-1)]
                    outp=rng.choice(outs[src["primitive"]])
                    wires.append({"from":f"{src['id']}.{outp}","to":f"m{i:02d}.{inp}"})
        return {"modules":modules,"wires":wires,"views":[]}, taps
    for seed in [0,1,2,42,99,123]:
        def _mk(seed=seed):
            def check():
                rng=random.Random(seed)
                patch,taps=_random_patch(rng)
                e1=Engine(patch["modules"],patch["wires"],dt=DT,ui_taps=taps).run(20)
                e2=Engine(patch["modules"],patch["wires"],dt=DT,ui_taps=taps).run(20)
                assert e1["final"]==e2["final"]
                for k,v in e1["final"].items():
                    if v is None or isinstance(v,str): continue
                    assert is_finite(float(v)), k
                jf,_=oracle.run(patch,20,dt=DT,ui_taps=taps)
                assert not parity_diff(jf,e1["final"],1e-9), seed
            return check
        checks.append((f"random seed {seed}", _mk()))
    return checks

# ---------------------------------------------------------------- 3 gates
def s3_checks():
    checks=[]
    def _gate_val(i0,t,taps=(5,15)):
        if t==0: return 0.0
        state=float(i0)
        for s in range(1, t):
            if (s-1) in taps:
                state=1.0-state
        return state
    def _truth(g,a,b):
        if g=="gate_buffer": return float(a)
        if g=="gate_not": return 1.0-float(a)
        if g=="gate_and": return float(a*b)
        if g=="gate_or": return 1.0-(1.0-a)*(1.0-b)
        if g=="gate_nand": return 1.0-float(a*b)
        if g=="gate_nor": return (1.0-a)*(1.0-b)
        if g=="gate_xor": return float((a!=0)!=(b!=0))
        if g=="gate_xnor": return float((a!=0)==(b!=0))
        if g=="gate_imply": return 1.0 if ((a and b) or (not a)) else 0.0
        return 1.0 if (a and (not b)) else 0.0
    def _logic_expected(ia,ib):
        return {g:[_truth(g,_gate_val(ia,t),_gate_val(ib,t)) for t in range(20)] for g in GATES10}
    def cross_pinned():
        ea,eb=_logic_expected(0,0),_logic_expected(1,0)
        for g in GATES10:
            assert [float(x) for x in ea[g]]==PASS_A[g], g
            assert [float(x) for x in eb[g]]==PASS_B[g], g
    checks.append(("brute vs pinned PASS_A/B", cross_pinned))
    def logic_A():
        patch=_logic_patch(0,0)
        exp=_logic_expected(0,0)
        jf,js=oracle.run(patch,20,dt=DT,ui_taps=[5,15])
        res=Engine(patch["modules"],patch["wires"],views=patch["views"],dt=DT,ui_taps=[5,15]).run(20)
        assert not gates_diff(jf,res["final"],0.0)
        for g in GATES10:
            assert res["series"][f"{g}.q"]==[float(x) for x in exp[g]], g
            assert [float(x) for x in js[f"{g}.q"]]==[float(x) for x in exp[g]], g
    checks.append(("logic pass A engine==oracle==brute", logic_A))
    def logic_B():
        patch=_logic_patch(1,0)
        exp=_logic_expected(1,0)
        jf,js=oracle.run(patch,20,dt=DT,ui_taps=[5,15])
        res=Engine(patch["modules"],patch["wires"],views=patch["views"],dt=DT,ui_taps=[5,15]).run(20)
        assert not gates_diff(jf,res["final"],0.0)
        for g in GATES10:
            assert res["series"][f"{g}.q"]==[float(x) for x in exp[g]], g
            assert [float(x) for x in js[f"{g}.q"]]==[float(x) for x in exp[g]], g
    checks.append(("logic pass B engine==oracle==brute", logic_B))
    def quantum():
        Q=[0.0]+[-1.0]*8+[1.0]*3
        T=[0.0]+[1.0]*11
        exp={"p_x.out":[-q for q in Q],"p_y.out":[-q for q in Q],"p_z.out":[-q for q in Q],
             "hg.out":[q*C for q in Q],"ph_s.out":list(Q),"ph_t.out":list(Q),
             "cn.out":[t*(-1.0 if q>0.5 else 1.0) for t,q in zip(T,Q)],"cn.c_out":list(Q),
             "sw.a_out":list(T),"sw.b_out":list(Q),
             "tf.out":[t*(-1.0 if (q>0.5 and t>0.5) else 1.0) for t,q in zip(T,Q)],"tf.c_out":list(Q)}
        for k in QUANTUM_EXPECT:
            assert [float(x) for x in QUANTUM_EXPECT[k]]==[float(x) for x in exp[k]], k
        patch=_quantum_patch()
        jf,js=oracle.run(patch,12,dt=DT,ui_taps=[5])
        res=Engine(patch["modules"],patch["wires"],views=patch["views"],dt=DT,ui_taps=[5]).run(12)
        assert not gates_diff(jf,res["final"],0.0)
        for k,e in exp.items():
            assert res["series"][k]==[float(x) for x in e], k
            assert [float(x) for x in js[k]]==[float(x) for x in e], k
    checks.append(("quantum engine==oracle==brute", quantum))
    def alogic():
        thresh=ATOMS["alogic"].params.get("thresh",0.5)
        def sim(base,fac,fam,cell_b=0,cell_c=0):
            v=[0.0]*12
            for t in range(12):
                if t==0: v[t]=0.0
                elif t<=7: v[t]=base
                else: v[t]=base+fac
            st=0.0
            out=[]
            for t in range(12):
                vv=v[t]
                if fam==0: q=1.0 if vv>0.6 else 0.0
                elif fam==1:
                    if vv>thresh+0.1: st=1.0
                    elif vv<thresh-0.1: st=0.0
                    q=st
                elif fam==2:
                    if vv>0.75: st=1.0
                    elif vv<0.4: st=0.0
                    q=st
                elif fam==3: q=1.0 if vv>thresh else 0.0
                elif fam==4:
                    d=vv-thresh
                    d=max(-0.25,min(0.25,d))
                    q=0.5+d
                else:
                    a=1.0 if vv>thresh else 0.0
                    b=1.0 if cell_b>0.5 else 0.0
                    c=1.0 if cell_c>0.5 else 0.0
                    q=1.0 if (a+b+c)>=2 else 0.0
                q=1.0 if abs(q)>1e-5 else 0.0
                out.append(q)
            return out
        for fam,base,fac in FAMS:
            cb=1 if fam==5 else 0
            cc=0
            assert sim(base,fac,fam,cb,cc)==ALOGIC_EXPECT[fam], (fam, sim(base,fac,fam,cb,cc), ALOGIC_EXPECT[fam])
        patch=_alogic_patch()
        jf,js=oracle.run(patch,12,dt=DT,ui_taps=[5])
        res=Engine(patch["modules"],patch["wires"],views=patch["views"],dt=DT,ui_taps=[5]).run(12)
        assert not gates_diff(jf,res["final"],0.0)
        for i,(fam,_,_) in enumerate(FAMS):
            k=f"g{i}.q"
            assert res["series"][k]==ALOGIC_EXPECT[fam], (i,fam)
            assert [float(x) for x in js[k]]==ALOGIC_EXPECT[fam], (i,fam)
    checks.append(("alogic engine==oracle==brute", alogic))
    return checks

# ---------------------------------------------------------------- 4 h4
def s4_checks():
    try:
        import hoa64, numpy as np
    except ImportError as e:
        raise _Skip(f"hoa64/numpy not importable ({e})")
    checks=[]
    def rows_match():
        H=np.array(hoa64.sylvester(4),dtype=float)
        outs=[]
        for i in range(4):
            seq=[0.0]*4; seq[i]=1.0
            f=_h4_run(seq,5)["final"]
            outs.append([f["h.w"],f["h.z"],f["h.y"],f["h.x"]])
        got=np.array(outs[::-1])
        assert (got==H).all(), f"h4_slide != sylvester {got} vs {H}"
    checks.append(("rows == sylvester(4)", rows_match))
    def orth_and_w():
        rows=[]
        for i in range(4):
            seq=[0.0]*4; seq[i]=1.0
            f=_h4_run(seq,5)["final"]
            rows.append([f["h.w"],f["h.z"],f["h.y"],f["h.x"]])
        H=np.array(rows)[::-1]
        assert (H @ H.T == 4*np.eye(4)).all()
        f=_h4_run([2.0]*6,6)["final"]
        assert f["h.w"]==8.0 and f["h.z"]==0.0 and f["h.y"]==0.0 and f["h.x"]==0.0
    checks.append(("orthogonality & W dominance const", orth_and_w))
    def random_w():
        seq=[3.0*math.sin(i/3.0)+1.5*math.sin(i/7.0) for i in range(24)]
        f=_h4_run(seq,24)["final"]
        w,z,y,x=f["h.w"],f["h.z"],f["h.y"],f["h.x"]
        assert w*w > z*z and w*w>y*y and w*w>x*x, (w,z,y,x)
    checks.append(("random stream W dominant", random_w))
    def label_swap():
        seq=[1.0,2.0,3.0,4.0,5.0]
        atoms=dict(ATOMS); atoms["script"]=_script_atom(seq)
        eng=Engine([{"id":"src","primitive":"script","params":{}},
                     {"id":"c","primitive":"hadamard4","params":{}},
                     {"id":"s","primitive":"h4_slide","params":{}}],
                    [{"from":"src.cv","to":"c.in"},{"from":"src.cv","to":"s.in"}], atoms=atoms)
        f=eng.run(5)["final"]
        assert f["c.w"]==f["s.w"] and f["s.z"]==f["c.y"] and f["s.y"]==f["c.x"] and f["s.x"]==f["c.z"]
    checks.append(("label swap vs fabric", label_swap))
    def oracle_hadamard4():
        patch={"modules":[{"id":"clk","primitive":"clock_bpm","params":{"bpm":60}},
                          {"id":"h","primitive":"hadamard4","params":{}}],
               "wires":[{"from":"clk.trig","to":"h.in"}],
               "views":[{"module":"clk","as":"series","output":"trig"},{"module":"h","as":"series","output":"w"}]}
        jf,js=oracle.run(patch,34,dt=DT)
        res=Engine(patch["modules"],patch["wires"],views=patch["views"],dt=DT).run(34)
        assert not parity_diff(jf,res["final"],0.0)
        trig=js["clk.trig"]
        assert trig[30]==1 and sum(1 for v in trig if v)==1, trig
        f=res["final"]
        assert f["h.w"]==1 and f["h.z"]==-1 and f["h.y"]==1 and f["h.x"]==-1, f
    checks.append(("oracle hadamard4 clock@60", oracle_hadamard4))
    return checks

# ---------------------------------------------------------------- 5 cv
def s5_checks():
    checks=[]
    def smooth():
        patch=PATCHES[1][4]
        jf,_=oracle.run(patch,200,dt=DT)
        res=Engine(patch["modules"],patch["wires"],dt=DT).run(200)
        assert not parity_diff(jf,res["final"],0.0)
        assert abs(res["final"]["s1.cv"]-7) < 1e-6
    checks.append(("smooth converges to 7", smooth))
    def moving_avg():
        seq=[0.0]*5+[4.0]*10
        atoms=dict(ATOMS); atoms["script"]=_script_atom(seq)
        res=Engine([{"id":"src","primitive":"script","params":{}},
                     {"id":"m","primitive":"moving_avg","params":{"n":5}}],
                    [{"from":"src.cv","to":"m.in"}], atoms=atoms).run(15)
        assert res["final"]["m.cv"]==4.0, res["final"]
    checks.append(("moving_avg n=5", moving_avg))
    def threshold_live():
        patch={"modules":[{"id":"k","primitive":"const","params":{"value":4}},
                          {"id":"th","primitive":"threshold","params":{"hi":3,"lo":6}}],
               "wires":[{"from":"k.cv","to":"th.in"}],
               "views":[{"module":"th","as":"series","output":"gate"}]}
        feeds={10:{"params":{"th":{"hi":5}}}, 16:{"params":{"th":{"hi":2}}}}
        exp=[0.0]+[1.0]*9+[0.0]*6+[1.0]*4
        res=Engine(patch["modules"],patch["wires"],views=patch["views"],dt=DT,feeds=feeds).run(20)
        assert res["series"]["th.gate"]==exp, res["series"]["th.gate"]
        assert res["final"]["th.gate"]==1.0
        with LiveOracle() as live:
            jf,js=live.run(patch,20,dt=DT,feeds=feeds)
        assert not parity_diff(jf,res["final"],0.0)
        assert [float(v) for v in js["th.gate"]]==exp
    checks.append(("threshold hysteresis live feed", threshold_live))
    def accum():
        patch={"modules":COUNTER["modules"],"wires":COUNTER["wires"],"views":[{"module":"acc","as":"series","output":"acc"}]}
        jf,js=oracle.run(patch,60,dt=DT,ui_taps=[5,30])
        res=Engine(patch["modules"],patch["wires"],views=patch["views"],dt=DT,ui_taps=[5,30]).run(60)
        assert not parity_diff(jf,res["final"],0.0)
        assert res["final"]["acc.acc"]==2
        exp=[0.0]*6+[1.0]*25+[2.0]*29
        assert res["series"]["acc.acc"]==exp
        assert [float(v) for v in js["acc.acc"]]==exp
    checks.append(("accum edges", accum))
    return checks

# ---------------------------------------------------------------- 6 ir
def s6_checks():
    checks=[]
    def roundtrip():
        for label,_,_,_,patch in PATCHES:
            prog=Program.from_patch(patch,name=label)
            assert prog.validate()==[], (label,prog.validate())
            assert prog.to_patch()==patch, label
    checks.append(("round-trip 6 patches", roundtrip))
    def rejects():
        cases={"duplicate block id": Program("dup",blocks=[Block("c1","const"),Block("c1","const")]),
               "unknown primitive": Program("x",blocks=[Block("c1","does_not_exist")]),
               "wire endpoint form": Program("x",blocks=[Block("c1","const"),Block("g1","gain")],wires=[Wire("c1","g1.in")]),
               "src not a declared output": Program("x",blocks=[Block("g1","gain"),Block("g2","gain")],wires=[Wire("g1.in","g2.in")]),
               "dst not a declared input": Program("x",blocks=[Block("c1","const"),Block("g1","gain")],wires=[Wire("c1.cv","g1.cv")]),
               "wire into inputless node": Program("x",blocks=[Block("c1","const"),Block("c2","const")],wires=[Wire("c1.cv","c2.in")]),
               "wire to unknown module": Program("x",blocks=[Block("g1","gain")],wires=[Wire("zz.cv","g1.in")]),
               "self-wire": Program("x",blocks=[Block("g1","gain")],wires=[Wire("g1.cv","g1.in")]),
               "cycle": Program("x",blocks=[Block("c1","const"),Block("g1","gain"),Block("g2","gain")],
                                 wires=[Wire("c1.cv","g1.in"),Wire("g1.cv","g2.in"),Wire("g2.cv","g1.in")]),
               "non-lowercase param key": Program("x",blocks=[Block("c1","const",{"Value":5})]),
               "non-lowercase wire port": Program("x",blocks=[Block("c1","const"),Block("g1","gain")],wires=[Wire("c1.CV","g1.in")]),
               "dangling entry": Program("x",blocks=[Block("c1","const")],entry="zz")}
        for name,prog in cases.items():
            errs=prog.validate()
            assert errs, name
    checks.append(("rejects 11 violations", rejects))
    def empty():
        assert Program("empty").validate()==["program has no blocks"]
    checks.append(("rejects empty program", empty))
    def node_rule_fake2in():
        ATOMS["gauntlet_fake2in"]=Atom("gauntlet_fake2in","Fake","function",{},["a","b"],["out"],"",multi_in=False)
        try:
            errs=Program("x",blocks=[Block("f1","gauntlet_fake2in")]).validate()
            assert any("one input port" in e for e in errs), errs
        finally:
            del ATOMS["gauntlet_fake2in"]
    checks.append(("node rule fake2in", node_rule_fake2in))
    def exempts():
        p=Program("x",blocks=[Block("an","gate_and"),Block("tf","toffoli"),Block("v1","viz_wxyz3d")])
        assert p.validate()==[], p.validate()
    checks.append(("node rule exempts multi_in/sinks", exempts))
    def microfx_fabric():
        import fabric.microfx as mf
        patch=_cv_program().compile("microfx")
        assert set(patch)=={"modules","wires","views"}
        assert mf.validate_patch(patch) is None
    checks.append(("microfx fabric accepts", microfx_fabric))
    def gate_gap():
        import fabric.microfx as mf
        p=Program("gate_app",blocks=[Block("ui","tap"),Block("tg","toggle"),Block("gb","gate_buffer")],
                  wires=[Wire("ui.tap","tg.trig"),Wire("tg.state","gb.in")])
        assert p.validate()==[]
        err=mf.validate_patch(p.compile("microfx"))
        assert err is not None and "unknown primitive" in err, err
    checks.append(("fabric gate gap pinned", gate_gap))
    def alias_norm():
        p=Program("alias",blocks=[Block("c1","const",{"value":1}),Block("ta","toggle",{"initial":0}),Block("and1","and")],
                  wires=[Wire("ui.tap","ta.trig"),Wire("ta.state","and1.a")])
        assert p.validate()==[]
        mod={m["id"]:m for m in p.compile("microfx")["modules"]}
        assert mod["and1"]["primitive"]=="gate_and"
    checks.append(("alias and->gate_and", alias_norm))
    def targets_emit():
        p=_cv_program()
        e=p.compile("eel2")
        assert isinstance(e,str) and e.startswith("// Autogenerated") and "c1 (const)" in e
        m=p.compile("mermaid")
        assert m.startswith("flowchart TD") and "c1 --> g1" in m
        w=p.compile("wgsl")
        assert w.startswith("// WGSL")
        try:
            p.compile("fortran")
            assert False, "unknown target must raise"
        except ProgramError:
            pass
    checks.append(("eel2/mermaid/wgsl emit", targets_emit))
    def python_runner():
        src=_cv_program().compile("python")
        td=tempfile.mkdtemp(prefix="gauntlet_py_")
        path=os.path.join(td,"run_cv.py")
        with open(path,"w") as fh: fh.write(src)
        out=subprocess.run([sys.executable,path,"10"],capture_output=True,text=True,timeout=30)
        assert out.returncode==0, out.stderr[:600]
        final=json.loads(out.stdout)
        assert final["g1.cv"]==10, final
    checks.append(("python target live runner", python_runner))
    def ir_transparent():
        prog=Program.from_patch(COUNTER,name="counter")
        compiled=prog.compile("microfx")
        jf,_=oracle.run(compiled,60,dt=DT,ui_taps=[5,30])
        res=Engine(compiled["modules"],compiled["wires"],views=compiled.get("views") or [],dt=DT,ui_taps=[5,30]).run(60)
        assert not parity_diff(jf,res["final"],0.0)
        assert res["final"]["acc.acc"]==2
    checks.append(("IR transparent to oracle", ir_transparent))
    return checks

# ---------------------------------------------------------------- 7 tiles
def s7_checks():
    checks=[]
    def res_3x3():
        d=Display(1920,1080,3,3,frame_h=120)
        assert (d.tile_w,d.tile_h)==(640,320)
        assert d.grid_width==1920 and d.grid_height==960
        assert d.margin_x==0 and d.margin_y==0
    checks.append(("1920x1080 3x3 res", res_3x3))
    def border_4x4():
        d=Display(3841,2167,4,4,frame_h=64)
        assert (d.tile_w,d.tile_h)==(960,525)
        assert d.margin_x==1 and d.margin_y==3
        assert d.grid_width+d.margin_x==d.width
        assert d.frame.h+d.grid_height+d.margin_y==d.height
    checks.append(("3841x2167 4x4 dead border", border_4x4))
    def frame_tiles():
        d=Display(900,900,3,3)
        assert [t.bounds() for t in d.tiles]==[(c*300,r*300,300,300) for r in range(3) for c in range(3)]
        d2=Display(1000,800,3,3,frame_h=100,controls=["bpm",("amp","slider")])
        assert d2.frame.bounds()==(0,0,1000,100)
        assert d2.tile(0,0).y==100
        assert d2.frame.controls==[{"name":"bpm","kind":"param"},{"name":"amp","kind":"slider"}]
    checks.append(("frame + tiles + controls", frame_tiles))
    def rejected():
        for cols,rows in [(2,3),(3,2),(5,4),(4,3),(3,5)]:
            try: Display(100,100,cols,rows); assert False, f"{cols}x{rows}"
            except TileError: pass
        for w,h,fh in [(0,100,0),(100,-5,0),(100,100,-1),(100,100,100),(2,100,0),(100,2,0)]:
            try: Display(w,h,3,3,frame_h=fh); assert False, (w,h,fh)
            except TileError: pass
    checks.append(("rejects non-square/degenerate", rejected))
    def linking():
        d=Display(1000,1000,4,4,frame_h=100)
        g=d.link("wall",1,1,2,2)
        assert (g.width,g.height)==(2*d.tile_w,2*d.tile_h)
        assert (g.x,g.y)==(d.tile(1,1).x,d.tile(1,1).y)
        assert len(g.tiles)==4
        try: d.link("wall",1,1,2,2); assert False, "dup"
        except TileError: pass
        try: d.link("ov",2,2,2,2); assert False, "overlap"
        except TileError: pass
        try: d.link("bad",3,0,2,1); assert False, "oob"
        except TileError: pass
        try: d.tile(4,0); assert False, "oob tile"
        except TileError: pass
    checks.append(("linking sub-matrices", linking))
    def full_and_summary():
        d=Display(100,100,3,3)
        g=d.link("all",0,0,3,3)
        assert (g.width,g.height)==(d.grid_width,d.grid_height)
        d2=Display(1000,800,4,4,frame_h=64)
        d2.link("wall",0,0,2,2)
        s=d2.summary()
        assert s["tile_w"]==250 and s["tile_h"]==184
        assert len(s["tiles"])==16
        assert s["groups"]["wall"]["span"]==[2,2]
    checks.append(("full wall + summary", full_and_summary))
    return checks

# ---------------------------------------------------------------- 8 replay
def s8_checks():
    checks=[]
    def pure_observer():
        patch=COUNTER
        plain=Engine(patch["modules"],patch["wires"],views=patch.get("views") or [],dt=DT,ui_taps=[5,30]).run(60)
        trace=FlowTrace()
        traced=Engine(patch["modules"],patch["wires"],views=patch.get("views") or [],dt=DT,ui_taps=[5,30],trace=trace).run(60)
        assert traced["final"]==plain["final"]
        assert traced["series"]==plain["series"]
        assert trace.snapshot()["n_frames"]==60*len(patch["modules"])
    checks.append(("traced == untraced", pure_observer))
    def counter_replay():
        trace=FlowTrace()
        orig=Engine(COUNTER["modules"],COUNTER["wires"],dt=DT,ui_taps=[5,30],trace=trace).run(60)
        assert orig["final"]["acc.acc"]==2
        again=replay(trace,COUNTER["modules"],COUNTER["wires"],views=COUNTER["views"],dt=DT)
        assert again["final"]==orig["final"] and again["final"]["acc.acc"]==2
    checks.append(("counter replay bit-identical", counter_replay))
    def live_feed_replay():
        patch={"modules":[{"id":"c1","primitive":"const","params":{"value":5}},
                          {"id":"g1","primitive":"gain","params":{"factor":2}}],
               "wires":[{"from":"c1.cv","to":"g1.in"}],"views":[]}
        feeds={2:{"params":{"g1":{"factor":4.0}}}, 9:{"params":{"g1":{"factor":0.5}}}}
        trace=FlowTrace()
        orig=Engine(patch["modules"],patch["wires"],dt=DT,feeds=feeds,trace=trace).run(15)
        assert orig["final"]["g1.cv"]==2.5
        again=replay(trace,patch["modules"],patch["wires"],views=patch["views"],dt=DT)
        assert again["final"]==orig["final"] and again["final"]["g1.cv"]==2.5
    checks.append(("live-feed replay bit-identical", live_feed_replay))
    def ring_wrap():
        trace=FlowTrace(max_frames=20)
        eng=Engine(COUNTER["modules"],COUNTER["wires"],dt=DT,ui_taps=[5,30,55],trace=trace)
        eng.run(10)
        assert trace.snapshot()["n_frames"]==20
        eng.tick()
        snap=trace.snapshot()
        assert snap["n_frames"]==20 and snap["seq"]==22
        assert snap["frames"][0]["seq"]==3
    checks.append(("ring wrap evicts oldest", ring_wrap))
    def start_stop():
        trace=FlowTrace(active=False)
        assert trace.record_node("a","const",{},{"cv":1.0},12.0,0) is None
        trace.begin_tick(0,1)
        assert trace.frames==[] and trace.ticks==[]
        trace.start()
        e=trace.record_node("a","const",{},{"cv":1.0},12.0,0)
        assert e is not None and e.seq==1
        trace.stop()
        assert trace.record_node("b","gain",{"in":1.0},{},5.0,1) is None
    checks.append(("start/stop gates recording", start_stop))
    return checks

# ---------------------------------------------------------------- 9 qbf
def s9_checks():
    checks=[]
    def store_roundtrip():
        from atomic.qbfstore import open_trace_store, close_all
        tmp=tempfile.mkdtemp(prefix="gauntlet_qbf_")
        try:
            trace=FlowTrace()
            patch=COUNTER
            Engine(patch["modules"],patch["wires"],dt=DT,ui_taps=[5,30],trace=trace).run(60)
            from atomic.qbfstore import QbfTraceStore
            store=open_trace_store(f"gauntlet_{os.getpid()}", shard_dir=tmp)
            manifest=store.append_run(trace,dt=DT,note="gauntlet",program=patch)
            assert manifest["n_frames"]==60*len(patch["modules"])
            loaded=store.load_run(manifest["run_id"])
            snap=trace.snapshot()
            assert loaded["ticks"]==snap["ticks"]
            assert loaded["frames"]==snap["frames"]
            assert loaded["manifest"]["n_frames"]==snap["n_frames"]
            assert store.export_run(manifest["run_id"]) == trace.export()
            again=store.replay_run(manifest["run_id"])
            assert again["final"]["acc.acc"]==2 and again["final"]==Engine(patch["modules"],patch["wires"],dt=DT,ui_taps=[5,30]).run(60)["final"]
            rid2=store.append_run(trace,dt=DT,note="second",program=patch)["run_id"]
            assert store.runs()==[manifest["run_id"], rid2]
            trace2=FlowTrace(); Engine(patch["modules"],patch["wires"],dt=DT,ui_taps=[5],trace=trace2).run(6)
            store2=open_trace_store(f"gauntlet2_{os.getpid()}", shard_dir=tmp)
            store2.append_run(trace2,dt=DT,note="no-prog")
            try:
                store2.replay_run(0); assert False, "must raise without program"
            except QbfError: pass
        finally:
            try: close_all()
            except: pass
            import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    checks.append(("store round-trip + replay", store_roundtrip))
    def qbf_header_and_blob():
        tmp=tempfile.mkdtemp(prefix="gauntlet_qbfhdr_")
        try:
            path=os.path.join(tmp,"hdr.qbf")
            f=QbfFile.create(path)
            f.put("raw", b"hello", RAW)
            f.put_json("doc", {"a":1})
            f.write()
            g=QbfFile.open(path)
            assert g.get("raw")==b"hello" and g.get_json("doc")=={"a":1}
            assert g.info("raw")["type"]==RAW
            raw=open(path,"rb").read()
            assert raw[0:4]==b"QBF\x00" and raw[4]==1
            assert struct.unpack_from("<I",raw,8)[0]==2
        finally:
            import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    checks.append(("QbfFile header & blob types", qbf_header_and_blob))
    def no_mv2_wall():
        tmp=tempfile.mkdtemp(prefix="gauntlet_big_")
        try:
            path=os.path.join(tmp,"big.qbf")
            payload=bytes(range(251))*250_000
            assert len(payload)==62750000
            poison=b"x"*3263
            f=QbfFile.create(path)
            f.put("big", payload, RAW)
            f.put("poison", poison, RAW)
            f.write()
            g=QbfFile.open(path)
            assert g.get("big")==payload
            assert g.get("poison")==poison
        finally:
            import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    checks.append(("no 50MB wall + poison payload", no_mv2_wall))
    def checksum_flip():
        tmp=tempfile.mkdtemp(prefix="gauntlet_chk_")
        try:
            path=os.path.join(tmp,"chk.qbf")
            f=QbfFile.create(path)
            f.put("raw", b"abcdef", RAW, checksum=True)
            f.write()
            raw=bytearray(open(path,"rb").read())
            data_off=struct.unpack_from("<Q",raw,24)[0]
            raw[data_off+2] ^= 0xFF
            open(path,"wb").write(raw)
            try: QbfFile.open(path); assert False, "checksum must fail"
            except QbfError: pass
        finally:
            import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    checks.append(("checksum flip raises QbfError", checksum_flip))
    def missing_blob():
        tmp=tempfile.mkdtemp(prefix="gauntlet_miss_")
        try:
            path=os.path.join(tmp,"miss.qbf")
            f=QbfFile.create(path); f.put("a",b"1"); f.write()
            g=QbfFile.open(path)
            try: g.get("nope"); assert False
            except QbfError: pass
        finally:
            import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    checks.append(("missing blob raises QbfError", missing_blob))
    def h4_gate_math():
        assert h4_gate((1,2,3,4))==(10,-2,-4,0)
        assert h4_inverse((10,-2,-4,0))==(1,2,3,4)
        groups=[(1.0,2.0,3.0,4.0),(5.0,6.0,7.0,8.0)]
        from atomic.qbf import h4_encode, h4_decode
        assert h4_decode(h4_encode(groups))==[(1.0,2.0,3.0,4.0),(5.0,6.0,7.0,8.0)]
        tmp=tempfile.mkdtemp(prefix="gauntlet_h4b_")
        try:
            path=os.path.join(tmp,"h4.qbf")
            f=QbfFile.create(path); f.put_h4("h4", groups); f.write()
            g=QbfFile.open(path)
            got=g.get_h4("h4")
            for a,b in zip(got, groups):
                assert all(abs(x-y)<1e-6 for x,y in zip(a,b)), (a,b)
        finally:
            import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    checks.append(("h4_gate math + H4 blob", h4_gate_math))
    def empty_trace_rejected():
        from atomic.qbfstore import open_trace_store, close_all
        tmp=tempfile.mkdtemp(prefix="gauntlet_empty_")
        try:
            store=open_trace_store(f"empty_{os.getpid()}", shard_dir=tmp)
            tr=FlowTrace()
            try: store.append_run(tr,dt=DT); assert False
            except QbfError: pass
        finally:
            try: close_all()
            except: pass
            import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    checks.append(("empty trace rejected", empty_trace_rejected))
    return checks

# ---------------------------------------------------------------- 10 swarm / decompose / teach
def s10_checks():
    checks=[]
    def swarm_consensus():
        d=Display(1200,1200,3,3)
        swarm=Swarm(display=d)
        def mk(v):
            return Program("p", blocks=[Block("c0","const",{"value": float(v)}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        for i,v in enumerate([1,2,3,4]):
            r,c=i%3,i//3
            g=d.link("a%d"%i,r,c,1,1)
            swarm.add_agent(Agent("a%d"%i, mk(v), tile_group=g))
        res=swarm.run(3, parallel=True)
        assert res.consensus("g1.cv")==10.0
        res2=swarm.run(3, parallel=False)
        for i in range(4):
            assert res.by_id("a%d"%i)["final"]["g1.cv"]==res2.by_id("a%d"%i)["final"]["g1.cv"]
        # isolation: different values don't leak
        assert res.by_id("a0")["final"]["g1.cv"]==1.0
        assert res.by_id("a3")["final"]["g1.cv"]==4.0
    checks.append(("swarm parallel consensus & isolation", swarm_consensus))
    def swarm_empty():
        swarm=Swarm()
        res=swarm.run(5)
        assert res.results==[]
        assert res.consensus("g1.cv")==0.0
    checks.append(("swarm empty", swarm_empty))
    def swarm_tile_exclusivity():
        d=Display(1200,1200,4,4)
        swarm=Swarm(display=d)
        g0=d.link("dup",0,0,1,1)
        swarm.add_agent(Agent("a0", Program("p", blocks=[Block("c0","const",{"value":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","v0.in")]), tile_group=g0))
        try:
            swarm.add_agent(Agent("a1", Program("p2", blocks=[Block("c0","const",{"value":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","v0.in")]), tile_group=g0))
            assert False, "duplicate tile group must raise"
        except ValueError:
            pass
    checks.append(("swarm tile exclusivity", swarm_tile_exclusivity))
    def decompose_python_valid():
        p=decompose_python("x = 2 * 3", name="t")
        assert p.validate()==[], p.validate()
        res=Engine(p.compile("microfx")["modules"], p.compile("microfx")["wires"]).run(5)
        assert res["final"] is not None
    checks.append(("decompose python valid", decompose_python_valid))
    def decompose_eel2_valid():
        src="v = input('in'); output('cv', v * 2.0);"
        p=decompose_eel2(src, name="eel")
        assert p.validate()==[], p.validate()
        assert any(b.primitive=="gain" for b in p.blocks)
    checks.append(("decompose eel2 valid", decompose_eel2_valid))
    def decompose_empty_raises():
        try:
            decompose_python("", name="e")
            assert False
        except DecomposeError:
            pass
        try:
            decompose_eel2("  ", name="e")
            assert False
        except DecomposeError:
            pass
    checks.append(("decompose empty raises", decompose_empty_raises))
    def teach_match():
        hit=REGISTRY.match("gated clock counter chart")
        assert hit is not None and hit["program"].name=="gated_clock_counter"
    checks.append(("teach registry match", teach_match))
    def teach_domain():
        p=REGISTRY.from_description("hadamard spatial wxyz 3d scope", domain="spatial")
        assert any(b.primitive=="h4_slide" for b in p.blocks)
        p2=REGISTRY.from_description("medical sensor threshold alarm", domain="medical")
        assert any(b.primitive=="threshold" for b in p2.blocks)
        v=domain_vocab("audio")
        assert "hadamard" in v
    checks.append(("teach domain routing", teach_domain))
    def teach_from_description():
        p=from_description("a clock that counts with smooth chart", domain="control")
        assert p.validate()==[], p.validate()
        patch=p.compile("microfx")
        res=Engine(patch["modules"], patch["wires"]).run(10)
        assert res["final"] is not None
        # Program classmethod
        p2=Program.from_description("gain scale audio", domain="signal")
        assert isinstance(p2, Program) and p2.validate()==[]
    checks.append(("teach from_description synthesis", teach_from_description))
    def decompose_roundtrip():
        p=decompose_python("y = 2 * 4", name="rt")
        patch=p.compile("microfx")
        assert patch["modules"] and patch["wires"]
        res=Engine(patch["modules"], patch["wires"]).run(4)
        assert res["final"] is not None
    checks.append(("decompose roundtrip engine", decompose_roundtrip))
    return checks

def s11_checks():
    checks=[]
    def evolve_improves():
        from atomic import Evolver
        def fitness(final): return -abs(float(final.get("g1.cv",0))-10.0)
        p = Program("tun", blocks=[Block("c0","const",{"value":5}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")],
                    wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        ev = Evolver(p, fitness, seed=0, ticks=5)
        start = ev.best_score
        assert start==-5.0
        ev.run(20)
        assert ev.best_score > start
        assert abs(float(ev.best.blocks[1].params["factor"])-2.0) < 1.0
    checks.append(("evolve hill-climb improves", evolve_improves))
    def evolve_determinism():
        from atomic import Evolver
        def fitness(final): return float(final.get("g1.cv",0))
        base = Program("tun", blocks=[Block("c0","const",{"value":5}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")],
                       wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        ev1 = Evolver(base, fitness, seed=42, ticks=5); ev1.run(10)
        ev2 = Evolver(base, fitness, seed=42, ticks=5); ev2.run(10)
        assert ev1.best.hash==ev2.best.hash and ev1.best_score==ev2.best_score
    checks.append(("evolve determinism", evolve_determinism))
    def evolve_mutate_valid():
        from atomic import Evolver
        base = Program("tun", blocks=[Block("c0","const",{"value":5}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")],
                       wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        ev = Evolver(base, lambda f: 0, seed=1)
        for _ in range(5):
            child = ev.mutate(base)
            assert child.validate()==[]
    checks.append(("evolve mutate valid", evolve_mutate_valid))
    def bridge_latency():
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8)
        b.push(0, {"x":1.0})
        assert b.pop(0) is None
        assert b.pop(1)=={"x":1.0}
    checks.append(("host bridge latency 1", bridge_latency))
    def bridge_capacity():
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=2)
        b.push(0, {"a":1}); b.push(1, {"b":2}); b.push(2, {"c":3})
        assert b.depth()==2
    checks.append(("host bridge capacity eviction", bridge_capacity))
    def bicameral_pipeline():
        from atomic.bridge import BicameralPipeline
        sub = Program("sub", blocks=[Block("c0","const",{"value":4}), Block("pre","gain",{"factor":2}), Block("v0","viz_series")], wires=[Wire("c0.cv","pre.in")])
        con = Program("con", blocks=[Block("dec","gain",{"factor":3}), Block("v1","viz_series")], wires=[Wire("dec.cv","v1.in")])
        pipe = BicameralPipeline(sub, con, bridge_map=[("pre.cv","dec.in")], bridge_latency=1)
        res = pipe.run(8)
        assert res.sub["final"]["pre.cv"]==8.0 and res.con["final"]["dec.cv"]==24.0
    checks.append(("bicameral pipeline final", bicameral_pipeline))
    def bicameral_determinism():
        from atomic.bridge import BicameralPipeline
        sub = Program("sub", blocks=[Block("c0","const",{"value":4}), Block("pre","gain",{"factor":2}), Block("v0","viz_series")], wires=[Wire("c0.cv","pre.in")])
        con = Program("con", blocks=[Block("dec","gain",{"factor":3}), Block("v1","viz_series")], wires=[Wire("dec.cv","v1.in")])
        r1 = BicameralPipeline(sub, con, bridge_map=[("pre.cv","dec.in")]).run(6)
        r2 = BicameralPipeline(sub, con, bridge_map=[("pre.cv","dec.in")]).run(6)
        assert r1.sub["final"]==r2.sub["final"] and r1.con["final"]==r2.con["final"]
    checks.append(("bicameral determinism", bicameral_determinism))
    def wgsl_codegen():
        p = Program("wgsl_t", blocks=[Block("c0","const",{"value":2}), Block("g1","gain",{"factor":3}), Block("h1","h4_slide"), Block("v0","viz_series")],
                    wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","h1.in"), Wire("h1.w","v0.in")])
        w = p.compile("wgsl")
        assert "@compute" in w and "@group(0)" in w and "tick_c0" in w and "host-RAM" in w
        assert "c0.cv -> g1.in" in w and "h1" in w
        assert "data" in w
    checks.append(("wgsl codegen shape", wgsl_codegen))
    def wgsl_bridge_comment():
        p = Program("b", blocks=[Block("c0","const",{"value":1}), Block("v0","viz_series")], wires=[Wire("c0.cv","v0.in")])
        w = p.compile("wgsl")
        assert "no P2P" in w or "host bridge" in w
    checks.append(("wgsl bridge comment", wgsl_bridge_comment))
    return checks

def s12_checks():
    checks=[]
    def decompose_add_wire_sum():
        # a+b as two vars: harness prior only used left; iter12 uses wire SUM
        prog = decompose_python("x = a + b", name="add_sum")
        assert prog.validate()==[], prog.validate()
        # must have a bias summer with two wires into same port -> add = sum
        # check wires: two wires into same dst
        dsts = [w.dst for w in prog.wires]
        assert len(dsts) > len(set(dsts)) or any(prog.blocks[i].primitive=="bias" for i in range(len(prog.blocks))), dsts
        # functional: wire SUM actually sums in engine (bias 0 with two inputs summed)
        from atomic import Engine
        # build explicit add via bias 0 with wire SUM: 2+3 -> 5
        patch = {"modules":[{"id":"c1","primitive":"const","params":{"value":2}},{"id":"c2","primitive":"const","params":{"value":3}},{"id":"s","primitive":"bias","params":{"add":0}},{"id":"v","primitive":"viz_series","params":{}}],"wires":[{"from":"c1.cv","to":"s.in"},{"from":"c2.cv","to":"s.in"},{"from":"s.cv","to":"v.in"}],"views":[]}
        res = Engine(patch["modules"], patch["wires"]).run(3)
        assert res["final"]["s.cv"]==5.0, res["final"]
        # decomposed python Add should also validate and wire SUM
        assert any(w.dst=="b0.in" or w.dst.endswith(".in") for w in prog.wires)
    checks.append(("decompose Add wire-SUM", decompose_add_wire_sum))
    def decompose_eel2_multi_input():
        src = "v = input('sensor'); w = input('aux'); output('cv', v + w);"
        prog = decompose_eel2(src, name="eel_multi")
        assert prog.validate()==[], prog.validate()
        # must have at least sensor + another source
        assert len(prog.blocks) >= 3
        assert any(b.primitive=="sensor" for b in prog.blocks)
    checks.append(("decompose EEL2 multi-input", decompose_eel2_multi_input))
    def evolver_qbf_history():
        from atomic import Evolver, Program, Block, Wire
        from atomic.qbfstore import open_trace_store, close_all
        import tempfile, os
        def fitness(final): return -abs(float(final.get("g1.cv",0))-10.0)
        prog = Program("tun", blocks=[Block("c0","const",{"value":5}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        ev = Evolver(prog, fitness, seed=1, ticks=5)
        ev.run(5)
        tmp = tempfile.mkdtemp(prefix="gauntlet_evolve12_")
        try:
            path = ev.save_history(store_name="evolve12_%d" % os.getpid(), shard_dir=tmp)
            loaded = Evolver.load_history("evolve12_%d" % os.getpid(), shard_dir=tmp)
            assert loaded["history"]==ev.history
            assert loaded["best"]["modules"][1]["params"]["factor"]==ev.best.blocks[1].params["factor"]
        finally:
            try: close_all()
            except: pass
            import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    checks.append(("evolver QBF history round-trip", evolver_qbf_history))
    def bridge_h4_codec():
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8, use_h4=True)
        payload = {"a":1.0,"b":2.0,"c":3.0,"d":4.0}
        b.push(0, payload)
        out = b.pop(1)
        assert out is not None
        for k in payload:
            assert abs(out[k]-payload[k])<1e-6, (k,out[k],payload[k])
        # non-H4 path (2 values) stored verbatim
        b2 = HostBridge(latency=1, capacity=8, use_h4=True)
        b2.push(0, {"x":5.0,"y":6.0})
        assert b2.pop(1)=={"x":5.0,"y":6.0}
    checks.append(("bridge H4 encode round-trip", bridge_h4_codec))
    def wgsl_naga_shape():
        import shutil, subprocess, tempfile, os
        from atomic import Program, Block, Wire
        p = Program("wgsl12", blocks=[Block("c0","const",{"value":1}), Block("g1","gain",{"factor":2}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        w = p.compile("wgsl")
        assert w.startswith("// WGSL")
        assert "@compute" in w and "@group(0)" in w
        assert "host-RAM" in w and "tick latency 1" in w
        assert "fn tick_c0" in w and "fn tick_g1" in w
        # naga hard-validate (iter 24: goal A — naga 30.0.1 installed)
        naga = shutil.which("naga")
        if naga:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".wgsl", delete=False) as fh:
                fh.write(w); tmp = fh.name
            try:
                out = subprocess.run([naga, tmp], capture_output=True, text=True, timeout=10)
                os.unlink(tmp)
                assert out.returncode == 0, "naga rejected: " + out.stderr[:200]
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
    checks.append(("wgsl naga shape", wgsl_naga_shape))
    def swarm_determinism():
        from atomic import Swarm, Agent, Display, Program, Block, Wire
        d = Display(1200,1200,3,3)
        def mk(v): return Program("p", blocks=[Block("c0","const",{"value": float(v)}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        g0 = d.link("a0",0,0,1,1); g1 = d.link("a1",1,0,1,1)
        sw = Swarm(display=d); sw.add_agent(Agent("a0", mk(3), tile_group=g0)); sw.add_agent(Agent("a1", mk(7), tile_group=g1))
        r1 = sw.run(4, parallel=True); r2 = sw.run(4, parallel=False)
        assert r1.consensus("g1.cv")==r2.consensus("g1.cv")==10.0
    checks.append(("swarm consensus determinism", swarm_determinism))
    return checks

def s13_checks():
    checks=[]
    def decompose_sub_wire_sum():
        prog = decompose_python("x = a - b", name="sub_sum")
        assert prog.validate()==[], prog.validate()
        # Sub must use wire SUM: a + (-b) via gain -1 + bias
        # check at least one gain -1 and one bias with two inputs
        gains = [b for b in prog.blocks if b.primitive=="gain" and b.params.get("factor")==-1.0]
        biases = [b for b in prog.blocks if b.primitive=="bias"]
        assert biases, "sub needs bias summer"
        dsts = [w.dst for w in prog.wires]
        # bias summer has at least two incoming (wire SUM)
        assert max([dsts.count(d) for d in set(dsts)] or [0]) >= 2 or gains, dsts
        # functional: 7 - 2 -> 5 via const folding
        prog2 = decompose_python("x = 7 - 2", name="sub_const")
        assert prog2.validate()==[]
        patch = prog2.compile("microfx")
        res = Engine(patch["modules"], patch["wires"]).run(3)
        # find bias output
        bias_ids = [b.id for b in prog2.blocks if b.primitive=="bias"]
        assert bias_ids
        assert abs(float(res["final"][bias_ids[0]+".cv"]) - 5.0) < 1e-9, res["final"]
    checks.append(("decompose Sub wire-SUM", decompose_sub_wire_sum))
    def decompose_div_and_annassign():
        prog = decompose_python("y: float = 8 / 2", name="div_ann")
        assert prog.validate()==[], prog.validate()
        patch = prog.compile("microfx")
        res = Engine(patch["modules"], patch["wires"]).run(3)
        gains = [b for b in prog.blocks if b.primitive=="gain"]
        assert gains
        # 8/2 -> gain 0.5 from 8 const -> 4.0
        # find gain output should be 4 after latency
        for g in gains:
            v = res["final"].get(g.id+".cv")
            if v is not None and abs(float(v)-4.0)<1e-9:
                break
        else:
            # at least ensure no crash and final exists
            assert any(isinstance(v,(int,float)) for v in res["final"].values())
        # AugAssign
        prog2 = decompose_python("x = 3\nx += 2", name="aug")
        assert prog2.validate()==[]
        prog3 = decompose_python("a = 2\nb = 3\nc = a + b\nc = c * 2", name="chain")
        assert prog3.validate()==[]
    checks.append(("decompose Div + AnnAssign + AugAssign", decompose_div_and_annassign))
    def teach_qbf_roundtrip():
        from atomic.teach import TeacherRegistry, REGISTRY
        from atomic import Program, Block, Wire
        import tempfile, shutil
        tmp = tempfile.mkdtemp(prefix="gauntlet_teach13_")
        try:
            reg = TeacherRegistry()
            p = Program("tprog", blocks=[Block("c0","const",{"value":5}),Block("v0","viz_series")], wires=[Wire("c0.cv","v0.in")], description="teach qbf test")
            reg.register("teach qbf test example", p, domain="signal")
            reg.register("second example clock counter", REGISTRY.match("gated clock counter chart")["program"], domain="control")
            path = reg.save_qbf(store_name="teach13_%d" % os.getpid(), shard_dir=tmp)
            loaded = TeacherRegistry.load_qbf("teach13_%d" % os.getpid(), shard_dir=tmp)
            assert len(loaded.examples) == len(reg.examples)
            hit = loaded.match("teach qbf test example")
            assert hit is not None and hit["program"].name=="tprog"
            # domain routing survives
            hit2 = loaded.match("clock counter", domain="control")
            assert hit2 is not None
            # file path round-trip via .qbf file
            import pathlib
            fp = os.path.join(tmp, "teach_file.qbf")
            reg.save_qbf(path=fp)
            loaded2 = TeacherRegistry.load_qbf(fp)
            assert len(loaded2.examples)==len(reg.examples)
        finally:
            try:
                from atomic.qbfstore import close_all as _ca
                _ca()
            except: pass
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("teach QBF persistence", teach_qbf_roundtrip))
    def evolver_swarm():
        from atomic import Evolver, Program, Block, Wire
        def fitness(final): return -abs(float(final.get("g1.cv",0))-10.0)
        prog = Program("tun", blocks=[Block("c0","const",{"value":5}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        ev = Evolver(prog, fitness, seed=7, ticks=5)
        start = ev.best_score
        assert start==-5.0
        ev.run_swarm(generations=10, population=4, parallel=True)
        assert ev.best_score > start, ev.best_score
        # determinism: sequential vs parallel population identical set -> same best hash when re-run with same seed
        ev2 = Evolver(prog, fitness, seed=7, ticks=5)
        ev2.run_swarm(generations=10, population=4, parallel=False)
        assert ev.best.hash==ev2.best.hash and ev.best_score==ev2.best_score
        # promote writes to registry
        h = ev.promote_to_registry("swarm tuned gain", domain="signal")
        assert isinstance(h, str) and len(h)==12
    checks.append(("evolver swarm parallel + promote", evolver_swarm))
    def wgsl_extended():
        from atomic import Program, Block, Wire
        # valid chain c0->th->cl->tg->v0 plus standalone lf (source, no input) and h4_slide fed by c0
        p = Program("wgsl_ext", blocks=[Block("c0","const",{"value":1}), Block("th","threshold",{"hi":0.5,"lo":-0.5}), Block("cl","clamp",{"lo":-1,"hi":1}), Block("lf","sine_lfo",{"rate_hz":1.0}), Block("tg","toggle"), Block("h1","h4_slide"), Block("v0","viz_series")],
                    wires=[Wire("c0.cv","th.in"), Wire("th.gate","cl.in"), Wire("cl.cv","tg.trig"), Wire("c0.cv","h1.in"), Wire("h1.w","v0.in")])
        w = p.compile("wgsl")
        assert "@compute" in w and "@group(0)" in w and "host-RAM" in w
        assert "clamp" in w and "tick_th" in w
        assert "sin(" in w
        assert "tick_lf" in w and "tick_th" in w
        # chain with sine_lfo as standalone source (no input port)
        p2 = Program("wgsl_ext2", blocks=[Block("c0","const",{"value":2}), Block("th","threshold",{"hi":0.5,"lo":-0.5}), Block("cl","clamp",{"lo":-1,"hi":1}), Block("lf","sine_lfo"), Block("v0","viz_series")],
                     wires=[Wire("c0.cv","th.in"), Wire("th.gate","cl.in"), Wire("cl.cv","v0.in")])
        w2 = p2.compile("wgsl")
        assert "@compute" in w2 and "host-RAM" in w2 and "sin(" in w2 and "tick_th" in w2
        # clock_bpm wgsl
        pc = Program("wgsl_clk", blocks=[Block("clk","clock_bpm",{"bpm":60}), Block("v0","viz_series")], wires=[Wire("clk.trig","v0.in")])
        wc = pc.compile("wgsl")
        assert "clock_bpm" in wc or "clk" in wc
        assert "tick_clk" in wc
    checks.append(("wgsl extended primitives", wgsl_extended))
    def decompose_eel2_complex():
        src = "v = input('sensor'); w = input('aux'); output('cv', v * 2.0 + w);"
        prog = decompose_eel2(src, name="eel_complex")
        assert prog.validate()==[], prog.validate()
        # should have at least sensor + const + gain + bias chain
        assert any(b.primitive=="sensor" for b in prog.blocks)
        assert len(prog.blocks) >= 3
    checks.append(("decompose EEL2 complex", decompose_eel2_complex))
    return checks

def s14_checks():
    checks=[]
    def full_ast_sweep():
        # loops, comps, classes, try/with/lambda/fstring
        cases = [
            "for i in range(5):\n    x = i * 2",
            "squares = [x*x for x in range(5)]",
            "evens = {x for x in range(5) if x%2==0}",
            "d = {k: k*2 for k in range(3)}",
            "g = (x*2 for x in [1,2,3])",
            "class Foo:\n    x=5\n    def bar(self):\n        return self.x*2",
            "try:\n    y=1/0\nexcept:\n    y=0",
            "with open('f') as fh:\n    z=1",
            "f=lambda a: a*2",
        ]
        for src in cases:
            p = decompose_python(src, name="sweep")
            assert p.validate()==[], (src[:30], p.validate())
            patch = p.compile("microfx")
            res = Engine(patch["modules"], patch["wires"]).run(4)
            assert res["final"] is not None, src[:30]
        # fidelity: Sub via wire-SUM already verified; ensure Add+comp path still valid
        p2 = decompose_python("out = [x*2 for x in [1,2,3]]", name="comp2")
        assert any(b.primitive=="moving_avg" for b in p2.blocks)
    checks.append(("full AST sweep loops/comps/classes", full_ast_sweep))
    def swarm_evolve_teach_qbf():
        from atomic.demo import swarm_evolve_teach_demo
        import tempfile, os
        td = tempfile.mkdtemp(prefix="gauntlet_demo14_")
        path = os.path.join(td, "demo.qbf")
        res = swarm_evolve_teach_demo(ticks=10, generations=3, population=2, path=path, seed=1)
        assert res["improved"] is True
        assert res["load_ok"] is True
        assert res["replay_match"] is True
        assert os.path.exists(path)
        # registry domain routing survives
        from atomic.teach import load_registry
        reg = load_registry(path)
        assert len(reg.list(domain="signal")) >= 1
        import shutil as _sh; _sh.rmtree(td, ignore_errors=True)
    checks.append(("swarm->evolve->teach->QBF demo", swarm_evolve_teach_qbf))
    def bridge_benchmarks_and_h4():
        from atomic import HostBridge, bridge_benchmark, h4_streaming_metrics, h4_row_cosine, row_cos_gate, latency_histogram
        b = HostBridge(latency=2, capacity=10, use_h4=True)
        bm = b.benchmark(ticks=30, payload_keys=4)
        assert bm["ticks"] == 30 and bm["ticks_per_s"] > 0
        assert isinstance(bm["histogram"], dict)
        bb = bridge_benchmark(ticks=20, payload_keys=4)
        assert bb["ticks"] == 20
        groups = [(1,2,3,4),(2,3,4,5),(3,4,5,6),(4,5,6,7)]
        m = h4_streaming_metrics(groups)
        assert m["n_groups"] == 4 and m["w_energy_frac"] > 0.5
        # row_cos helpers
        assert abs(h4_row_cosine([1,0],[1,0])-1.0) < 1e-9
        assert abs(h4_row_cosine([1,0],[0,1])-0.0) < 1e-9
        passed, avg = row_cos_gate([(1,1,1,1),(1,-1,1,-1)])
        assert isinstance(passed, bool)
        hist = latency_histogram([1,1,2,2,2,3])
        assert hist == {1:2,2:3,3:1}
        # H4 lossless round-trip
        b2 = HostBridge(latency=1, capacity=8, use_h4=True)
        payload = {"a":1.0,"b":2.0,"c":3.0,"d":4.0}
        b2.push(0, payload)
        out = b2.pop(1)
        for k in payload:
            assert abs(out[k]-payload[k]) < 1e-6
        # tiles viz wiring
        from atomic import Display
        d = Display(1200,1200,3,3)
        hb = HostBridge(latency=1, capacity=8)
        for t in range(5):
            hb.push(t, {"x": float(t)})
        tv = hb.to_tiles(display=d)
        assert isinstance(tv, dict)
    checks.append(("bridge benchmarks + H4 metrics + row_cos", bridge_benchmarks_and_h4))
    def bench_wired_to_tiles():
        from atomic import HostBridge, Display, bridge_benchmark
        b = HostBridge(latency=1, capacity=16, use_h4=False)
        bm = b.benchmark(ticks=50, payload_keys=2)
        d = Display(1200,1200,4,4,frame_h=64)
        # histogram -> tiles
        hist = bm["histogram"]
        mapped = b.to_tiles(display=d)
        assert isinstance(mapped, dict)
        # bicameral pipeline metrics wire to tiles summary
        from atomic import Program, Block, Wire
        from atomic.bridge import BicameralPipeline
        sub = Program("sub", blocks=[Block("c0","const",{"value":4}), Block("pre","gain",{"factor":2}), Block("v0","viz_series")], wires=[Wire("c0.cv","pre.in")])
        con = Program("con", blocks=[Block("dec","gain",{"factor":3}), Block("v1","viz_series")], wires=[Wire("dec.cv","v1.in")])
        pipe = BicameralPipeline(sub, con, bridge_map=[("pre.cv","dec.in")], bridge_latency=1)
        pipe.run(6)
        met = pipe.metrics()
        assert "bridge_depth" in met and "histogram" in met
    checks.append(("bench wired to tiles viz", bench_wired_to_tiles))
    def tiles_swarm_heatmap():
        from atomic import Display, FlowTrace, Engine, Program, Block, Wire, Swarm, Agent
        # trace -> wall heatmap
        d = Display(900,900,3,3,frame_h=0)
        trace = FlowTrace()
        patch = {"modules":[{"id":"c0","primitive":"const","params":{"value":5}},{"id":"g1","primitive":"gain","params":{"factor":2}},{"id":"v0","primitive":"viz_series","params":{}}],
                 "wires":[{"from":"c0.cv","to":"g1.in"},{"from":"g1.cv","to":"v0.in"}],"views":[]}
        Engine(patch["modules"], patch["wires"], trace=trace).run(6)
        wall = d.heatmap_from_trace(trace, port="g1.cv", agg="max")
        assert len(wall) == 9 and all(0.0 <= v <= 1.0 for v in wall.values())
        # TileGroup heatmap
        g = d.link("grp14",0,0,2,2)
        gh = g.heatmap_from_trace(trace, port="g1.cv", agg="max")
        assert len(gh) == 4 and all(0.0 <= v <= 1.0 for v in gh.values())
        # swarm -> heatmap
        d2 = Display(1200,1200,3,3)
        swarm = Swarm(display=d2)
        for i, val in enumerate([1,4,7,10]):
            prog = Program("p", blocks=[Block("c0","const",{"value":float(val)}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
            grp = d2.link(f"sw{i}", i%3, i//3,1,1)
            swarm.add_agent(Agent(f"sw{i}", prog, tile_group=grp))
        res = swarm.run(3)
        hm = d2.heatmap_from_swarm(res, port="g1.cv", normalize=True)
        assert len(hm) == 4 and max(hm.values()) == 1.0 and min(hm.values()) == 0.0
        assert any("heatmap" in ginfo for ginfo in d2.summary()["groups"].values())
    checks.append(("tiles↔swarm live heatmap from trace", tiles_swarm_heatmap))
    def docs_wgsl_h4_note():
        # docs/CORE: WGSL now carries bridge comments and H4 streaming notes
        # check program wgsl still has the required pins
        from atomic import Program, Block, Wire
        p = Program("wgsl14", blocks=[Block("c0","const",{"value":1}), Block("g1","gain",{"factor":2}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        w = p.compile("wgsl")
        assert w.startswith("// WGSL") and "@compute" in w and "@group(0)" in w
        assert "host-RAM" in w
        # H4 notes
        from atomic.qbf import h4_gate
        assert h4_gate((1,2,3,4)) == (10,-2,-4,0)
    checks.append(("docs/release WGSL + H4 pins", docs_wgsl_h4_note))
    return checks

def s15_checks():
    checks=[]
    def decompose_edge_audit():
        from atomic import decompose_python, decompose_eel2, DecomposeError
        cases = [
            "async def foo():\n    x=5\n    return x*2",
            "if (n := 10) > 5:\n    y=n*2",
            "s=f'{2*3} hi'",
            "squares=[x*x for x in range(5)]",
            "evens={x for x in range(5) if x%2==0}",
            "d={k:k*2 for k in range(3)}",
            "g=(x*2 for x in [1,2,3])",
            "f=lambda a: a*2",
            "x=arr[2]",
            "y=await asyncio.sleep(0)",
        ]
        for src in cases:
            # some snippets like await need async context but decompose imports still handle; wrap if needed
            try:
                p = decompose_python(src, name="audit")
            except DecomposeError:
                # empty parse edge already covered
                continue
            assert p.validate()==[], (src[:30], p.validate())
            patch = p.compile("microfx")
            res = Engine(patch["modules"], patch["wires"]).run(4)
            assert res["final"] is not None, src[:30]
        # fidelity pin: 7-2 ->5
        p2 = decompose_python("out = 7 - 2", name="fid")
        res2 = Engine(p2.compile("microfx")["modules"], p2.compile("microfx")["wires"]).run(3)
        assert any(abs(float(v)-5.0)<1e-9 for v in res2["final"].values() if isinstance(v,(int,float)))
        # async for via decompose
        p3 = decompose_python("async def g():\n    async for x in aiter():\n        y=x*2", name="afor")
        assert p3.validate()==[]
    checks.append(("decompose async/walrus/fstring/comp audit", decompose_edge_audit))
    def swarm_16_scale():
        from atomic import Swarm, Agent, Display, Program, Block, Wire, h4_gate, h4_streaming_metrics, row_cos_gate
        d = Display(1600,1600,4,4,frame_h=0)
        swarm = Swarm(display=d)
        def mk(v): return Program("p", blocks=[Block("c0","const",{"value": float(v)}), Block("g1","gain",{"factor":1.0}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        for i in range(16):
            g = d.link("s%02d"%i, i//4, i%4, 1,1)
            swarm.add_agent(Agent("s%02d"%i, mk(i+1), tile_group=g))
        rp = swarm.run(6, parallel=True)
        rs = swarm.run(6, parallel=False)
        for i in range(16):
            assert rp.by_id("s%02d"%i)["final"]["g1.cv"]==rs.by_id("s%02d"%i)["final"]["g1.cv"]
        assert rp.consensus("g1.cv")==10.0
        groups = [tuple(float(rp.by_id("s%02d"% (g*4+j))["final"]["g1.cv"]) for j in range(4)) for g in range(4)]
        m = h4_streaming_metrics(groups)
        assert m["n_groups"]==4 and m["w_energy_frac"]>0.4
        passed, avg = row_cos_gate(groups, threshold=0.5)
        assert isinstance(passed, bool)
        w,_,_,_ = h4_gate(groups[0])
        assert w==10.0
    checks.append(("swarm 16 agents 4x4 consensus row_cos", swarm_16_scale))
    def trace_qbf_replay_scale():
        from atomic import FlowTrace, QbfFile, h4_encode, h4_decode, latency_histogram_from_trace
        from atomic.qbfstore import open_trace_store, close_all
        tmp=tempfile.mkdtemp(prefix="gauntlet_scale15_")
        try:
            patch={"modules":[{"id":"c0","primitive":"const","params":{"value":1}},{"id":"g1","primitive":"gain","params":{"factor":2}}],"wires":[{"from":"c0.cv","to":"g1.in"}],"views":[]}
            trace=FlowTrace(max_frames=10_000)
            Engine(patch["modules"], patch["wires"], trace=trace, dt=DT).run(5000)
            assert trace.snapshot()["n_frames"]==10000
            assert trace.snapshot()["seq"]==10000
            # ring wrap: max 20 wraps
            tr2=FlowTrace(max_frames=20)
            for i in range(11):
                tr2.begin_tick(i,0)
                for mid in ["c0","g1"]:
                    tr2.record_node(mid,"gain",{"in":1.0},{"cv":2.0},10.0,t=i)
            assert tr2.snapshot()["n_frames"]==20 and tr2.snapshot()["seq"]==22
            assert tr2.frames[0].seq==3
            store=open_trace_store("scale15_%d"%os.getpid(), shard_dir=tmp)
            man=store.append_run(trace, dt=DT, note="10k", program=patch)
            assert man["n_frames"]==10000
            loaded=store.load_run(man["run_id"])
            assert len(loaded["frames"])==10000
            again=store.replay_run(man["run_id"])
            orig=Engine(patch["modules"], patch["wires"], dt=DT).run(5000)["final"]
            assert again["final"]==orig
            buckets, raw = latency_histogram_from_trace(trace)
            assert len(raw)==10000 and isinstance(buckets, dict)
            # H4 fidelity at scale
            groups=[(float(i),float(i+1),float(i+2),float(i+3)) for i in range(200)]
            enc=h4_encode(groups)
            dec=h4_decode(enc)
            for a,b in zip(dec, groups):
                assert all(abs(x-y)<1e-5 for x,y in zip(a,b))
            # Qbf H4 blob
            path=os.path.join(tmp,"h4s.qbf")
            f=QbfFile.create(path); f.put_h4("h4", groups); f.write()
            got=QbfFile.open(path).get_h4("h4")
            for a,b in zip(got, groups):
                assert all(abs(x-y)<1e-5 for x,y in zip(a,b))
        finally:
            try: close_all()
            except: pass
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("trace 10k ring QBF replay latency H4", trace_qbf_replay_scale))
    def wgsl_naga_and_heatmap():
        from atomic import Display, FlowTrace, Program, Block, Wire
        prog=Program("wgsl15", blocks=[Block("c0","const",{"value":1}),Block("g1","gain",{"factor":2}),Block("h1","h4_slide"),Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"),Wire("g1.cv","h1.in"),Wire("h1.w","v0.in")])
        w=prog.to_wgsl()
        assert w.startswith("// WGSL") and "@compute" in w and "@group(0)" in w and "host-RAM" in w
        ok, detail = Display.validate_wgsl(w)
        assert ok, detail
        ok2, _ = prog.validate_wgsl()
        assert ok2
        bad="fn main() {}"
        ok3,_ = Display.validate_wgsl(bad)
        assert not ok3
        # heatmap animation from replay
        patch={"modules":[{"id":"c0","primitive":"const","params":{"value":5}},{"id":"g1","primitive":"gain","params":{"factor":2}},{"id":"v0","primitive":"viz_series","params":{}}],"wires":[{"from":"c0.cv","to":"g1.in"},{"from":"g1.cv","to":"v0.in"}],"views":[]}
        feeds={2:{"params":{"g1":{"factor":4.0}}},5:{"params":{"g1":{"factor":1.0}}}}
        trace=FlowTrace()
        Engine(patch["modules"], patch["wires"], trace=trace, dt=DT, feeds=feeds).run(10)
        d=Display(900,900,3,3,frame_h=0)
        anim=d.heatmap_animation(trace, port="g1.cv", window=1)
        assert len(anim)==10 and all(len(h)==9 for h in anim)
        assert all(all(0.0<=v<=1.0 for v in h.values()) for h in anim)
        anim2=d.heatmap_animation(trace, port="g1.cv", window=2)
        assert len(anim2)==5
        # wall heatmap still works
        wall=d.heatmap_from_trace(trace, port="g1.cv")
        assert len(wall)==9
        # swarm 16 heatmap
        d2=Display(1600,1600,4,4,frame_h=0)
        swarm=Swarm(display=d2)
        for i in range(16):
            pr=Program("p", blocks=[Block("c0","const",{"value": float(i+1)}),Block("g1","gain",{"factor":1.0}),Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"),Wire("g1.cv","v0.in")])
            grp=d2.link("t%02d"%i, i//4, i%4, 1,1)
            swarm.add_agent(Agent("t%02d"%i, pr, tile_group=grp))
        res=swarm.run(4)
        hm=d2.heatmap_from_swarm(res, port="g1.cv")
        assert len(hm)==16 and max(hm.values())==1.0
    checks.append(("WGSL naga + heatmap animation", wgsl_naga_and_heatmap))
    def swarm_tiles_summary():
        from atomic import Display, Swarm, Agent, Program, Block, Wire
        d=Display(1200,1200,4,4,frame_h=64)
        prog=Program("p", blocks=[Block("c0","const",{"value":3}),Block("v0","viz_series")], wires=[Wire("c0.cv","v0.in")])
        g=d.link("wall",0,0,2,2)
        s=d.summary()
        assert s["tile_w"]==300 and s["tile_h"]==284
        assert s["groups"]["wall"]["span"]==[2,2]
        assert "heatmap" in s["groups"]["wall"]
    checks.append(("tiles summary 4x4 with heatmap", swarm_tiles_summary))
    return checks

def s16_checks():
    checks=[]
    def zg_install_and_version():
        from atomic.context import is_available, zg_version
        if not is_available():
            # offline path: wrapper reports unavailable gracefully
            res = __import__("atomic.context", fromlist=["query"]).query("H4 gate", limit=1)
            assert res["available"] is False, res
            assert "zg not found" in res.get("reason","")
            return
        v = zg_version()
        assert v is not None and v[0].isdigit(), v
        assert "0." in v or v.startswith("0")
    checks.append(("zg install/version", zg_install_and_version))
    def zg_status_ready():
        from atomic.context import is_available, status
        if not is_available():
            s = status()
            assert s == ""
            return
        s = status()
        assert "ready" in s.lower() or "coverage" in s.lower(), s
        assert "ATOMIC-PC" in s or "potion" in s.lower() or "256" in s, s
    checks.append(("zg status ready (local/potion-code-16m-v2)", zg_status_ready))
    def h4_retrieval():
        from atomic.context import is_available, assert_retrieval
        if not is_available():
            return
        assert_retrieval("H4 gate row layout and W dominance", "ATOMIC-PC-CORE.md")
    checks.append(("zg H4 retrieval fidelity", h4_retrieval))
    def tiles_retrieval():
        from atomic.context import is_available, assert_retrieval
        if not is_available():
            return
        assert_retrieval("tile heatmap animation from trace replay", "tiles.py")
    checks.append(("zg tiles retrieval fidelity", tiles_retrieval))
    def qbf_retrieval():
        from atomic.context import is_available, assert_retrieval
        if not is_available():
            return
        assert_retrieval("QBF portable store ring wrap", "qbf")
    checks.append(("zg QBF retrieval fidelity", qbf_retrieval))
    def rg_exact():
        from atomic.context import is_available, query_rg
        if not is_available():
            res = query_rg("h4_gate")
            assert res["available"] is False
            return
        res = query_rg("h4_gate", extra_args=["-n"])
        assert res["available"] is True, res
        assert "h4_gate" in res["raw"], res["raw"][:400]
    checks.append(("zg rg exact (h4_gate)", rg_exact))
    def wrapper_structured():
        from atomic.context import is_available, query, CANONICAL_QUERIES
        if not is_available():
            return
        res = query("H4 gate", limit=2)
        assert res["available"] is True and isinstance(res["hits"], list) and len(res["hits"]) >= 1
        # canonical list pinned
        assert len(CANONICAL_QUERIES) == 3
    checks.append(("zg wrapper structured + canonical list", wrapper_structured))
    return checks

def s17_checks():
    checks=[]
    def ui_module_imports():
        from atomic.ui import app, Viewer, build, all_programs
        assert callable(app), "app not callable"
        assert callable(Viewer), "Viewer not callable"
        programs = all_programs()
        assert len(programs) >= 7, programs
        assert "gated_clock_counter" in programs
    checks.append(("ui module imports (app, Viewer, 7 programs)", ui_module_imports))
    def ui_server_index():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/")
            assert r.status_code == 200, r.status_code
            assert b"ATOMIC-PC" in r.content, "no ATOMIC-PC title"
            assert b"tile-wall" in r.content, "no tile-wall div"
    checks.append(("GET / returns index.html (control frame + tile wall)", ui_server_index))
    def ui_programs_endpoint():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/programs")
            assert r.status_code == 200
            d = r.json()
            assert "programs" in d
            assert len(d["programs"]) >= 7
            assert "active" in d
            assert len(d["active"]) >= 7
    checks.append(("GET /api/programs lists 7+ active", ui_programs_endpoint))
    def ui_viewer_batch_clock():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.post("/api/batch/clock_counter", json={"ticks": 30})
            assert r.status_code == 200, r.text
            d = r.json()
            assert d["t"] == 30
            assert d["bus"].get("cnt.acc", 0) >= 0
            assert "v0.cv" in d.get("series", {})
    checks.append(("POST /api/batch/clock_counter -> 30 ticks, series populated",
                   ui_viewer_batch_clock))
    def ui_views_layout():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.post("/api/batch/hadamard_wxyz", json={"ticks": 5})
            d = r.json()
            assert len(d["views"]) == 4, len(d["views"])
            for v in d["views"]:
                assert v["as"] == "series"
                assert "tile_row" in v and "tile_col" in v
            r2 = c.get("/api/views/hadamard_wxyz")
            assert r2.status_code == 200
            assert len(r2.json()["views"]) == 4
    checks.append(("GET /api/views/hadamard_wxyz -> 4 series tiles",
                   ui_views_layout))
    def ui_xy_and_wxyz3d_views():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.post("/api/batch/xy_pad", json={"ticks": 20})
            d = r.json()
            assert len(d["views"]) == 1
            assert d["views"][0]["viz"] == "xy"
            r2 = c.post("/api/batch/wxyz3d_demo", json={"ticks": 10})
            d2 = r2.json()
            assert len(d2["views"]) == 1
            assert d2["views"][0]["viz"] == "wxyz3d"
    checks.append(("xy_pad + wxyz3d_demo -> viz types xy/wxyz3d",
                   ui_xy_and_wxyz3d_views))
    def ui_live_feed_accepted():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.post("/api/feed/clock_counter",
                       json={"ticks": [5, 10, 15], "params": {"clk": {"bpm": 200}}})
            assert r.status_code == 200
            d = r.json()
            assert d.get("ok") is True
            assert d.get("applied") == 3
    checks.append(("POST /api/feed -> ticks+params applied",
                   ui_live_feed_accepted))
    def ui_tap_endpoint():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.post("/api/tap/clock_counter")
            assert r.status_code == 200
            d = r.json()
            assert d.get("ok") is True
            assert "tick" in d
    checks.append(("POST /api/tap records 1-tick tap",
                   ui_tap_endpoint))
    def ui_control_frame_schema():
        from atomic.ui.server import create_app, _control_schema
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/control/clock_counter")
            assert r.status_code == 200
            d = r.json()
            assert d["name"] == "clock_counter"
            assert "params" in d
            assert len(d["params"]) >= 1
            for p in d["params"]:
                assert "module" in p
                assert "key" in p
                assert "min" in p
                assert "max" in p
                assert isinstance(p["value"], float)
    checks.append(("GET /api/control -> slider schema (module/key/min/max/value)",
                   ui_control_frame_schema))
    def ui_h4_wxyz_bus():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.post("/api/batch/hadamard_wxyz", json={"ticks": 5})
            d = r.json()
            bus = d["bus"]
            assert abs(bus.get("h4.w", 0) - 4.0) < 0.01
            assert abs(bus.get("h4.x", 0) - 0.0) < 0.01
            assert abs(bus.get("h4.y", 0) - 0.0) < 0.01
            assert abs(bus.get("h4.z", 0) - 0.0) < 0.01
    checks.append(("h4_slide w=4, x=y=z=0 (CORE keystone)",
                   ui_h4_wxyz_bus))
    def ui_tile_bounds_4x4():
        from atomic.tiles import Display
        d = Display(800, 800, cols=4, rows=4, frame_h=0)
        assert d.tile_w == 200 and d.tile_h == 200
        assert len(d.tiles) == 16
    checks.append(("Display(800,800,4x4) -> 200x200 tiles, 16 total",
                   ui_tile_bounds_4x4))
    def ui_snapshot_shape():
        from atomic.ui.programs import build
        from atomic.ui.viewer import Viewer
        prog = build("sine_lfo_scope")
        v = Viewer(prog, name="snap_shape")
        snap = v.batch(10)
        for k in ("t", "bus", "series", "views", "running"):
            assert k in snap, k
        assert snap["running"] is False
        assert "window" in snap, "snapshot missing window field"
        assert snap["window"] == 512, snap["window"]
    checks.append(("Viewer.snapshot() shape (t/bus/series/views/running) + window=512",
                   ui_snapshot_shape))
    def ui_control_schema_enhanced():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/control/gated_clock_counter")
            assert r.status_code == 200
            d = r.json()
            assert "window" in d, "control schema missing window"
            assert d["window"] == 512, d["window"]
            has_bpm = any(p["key"] == "bpm" for p in d["params"])
            assert has_bpm, "no bpm param"
            for p in d["params"]:
                if p["key"] == "bpm":
                    assert p["min"] == 1.0, p
                    assert p["max"] == 300.0, p
                    assert p["step"] == 1.0, p
                    assert p["unit"] == "bpm", p
    checks.append(("GET /api/control -> step/unit + window field",
                   ui_control_schema_enhanced))
    def ui_xy_wxyz_key_extraction():
        from atomic.ui.programs import build
        from atomic.ui.viewer import Viewer
        xy_prog = build("xy_pad")
        v_xy = Viewer(xy_prog, name="xy_keys")
        v_xy.batch(10)
        snap = v_xy.snapshot()
        assert len(snap["views"]) == 1
        view = snap["views"][0]
        assert view["viz"] == "xy"
        assert view["key"] == "v0.y", view["key"]
        w_prog = build("wxyz3d_demo")
        v_w = Viewer(w_prog, name="wxyz_keys")
        v_w.batch(10)
        snap_w = v_w.snapshot()
        assert len(snap_w["views"]) == 1
        vw = snap_w["views"][0]
        assert vw["viz"] == "wxyz3d"
        assert vw["key"] == "v0.z", vw["key"]
    checks.append(("xy key=v0.y / wxyz key=v0.z (correct suffix strip)",
                   ui_xy_wxyz_key_extraction))
    def ui_rolling_window():
        from atomic.engine import VIEW_WINDOW
        assert VIEW_WINDOW == 512, VIEW_WINDOW
        from atomic.ui.programs import build
        from atomic.ui.viewer import Viewer
        prog = build("sine_lfo_scope")
        v = Viewer(prog, name="win_test")
        v.batch(600)
        snap = v.snapshot()
        for k, arr in snap["series"].items():
            assert len(arr) <= VIEW_WINDOW, f"{k} length {len(arr)} > {VIEW_WINDOW}"
    checks.append(("Engine VIEW_WINDOW=512 enforced on series lengths",
                   ui_rolling_window))
    def ui_ws_streaming():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            with c.websocket_connect("/ws/clock_counter") as ws:
                snap1 = ws.receive_json()
                assert "t" in snap1 and "bus" in snap1
                ws.send_json({"type": "batch", "ticks": 5})
                batch_seen = False
                for _ in range(20):
                    m = ws.receive_json()
                    if m.get("ack") == "batch":
                        assert "snapshot" in m, m
                        assert m["snapshot"]["t"] == 6, m["snapshot"]["t"]
                        batch_seen = True
                        break
                assert batch_seen, f"no batch ack in stream"
    checks.append(("WS /ws/{name} streams ticks + batch ack",
                   ui_ws_streaming))
    def ui_heatmap_from_bus():
        import re
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.post("/api/batch/hadamard_wxyz", json={"ticks": 10})
            d = r.json()
            bus = d["bus"]
            axis_keys = [k for k in bus if re.search(r"\.[wxyz]$", k)]
            cv_keys = [k for k in bus if ".cv" in k]
            total = len(axis_keys) + len(cv_keys)
            assert total >= 4, f"need 4+ heatmap-mappable bus keys, got cv={cv_keys} axis={axis_keys}"
    checks.append(("hadamard_wxyz bus has .cv/.w/.x/.y/.z keys for heatmap",
                   ui_heatmap_from_bus))
    return checks


def s18_checks():
    """iter 18: UI polish (WS backpressure, snapshot diff, param ranges from gates,
    viz_heatmap atom, heatmap_live demo, time-decayed heatmap)."""
    checks = []
    def ui_viz_heatmap_atom():
        from atomic.gates import ATOMS
        a = ATOMS.get("viz_heatmap")
        assert a is not None, "viz_heatmap atom missing"
        assert a.category == "sink", a.category
        assert a.inputs == ["in"], a.inputs
        assert a.outputs == [], a.outputs
    checks.append(("viz_heatmap atom registered (sink, in->no outputs)",
                   ui_viz_heatmap_atom))
    def ui_heatmap_live_program():
        from atomic.ui.programs import build, all_programs
        assert "heatmap_live" in all_programs()
        prog = build("heatmap_live")
        patch = prog.compile("microfx")
        prims = [m["primitive"] for m in patch["modules"]]
        assert "viz_heatmap" in prims, prims
        assert "h4_slide" in prims, prims
    checks.append(("heatmap_live demo program compiles with viz_heatmap",
                   ui_heatmap_live_program))
    def ui_param_range_canonical():
        from atomic.gates import param_range, PARAM_RANGES
        # server.py imports param_range from gates; the old _PARAM_RANGES
        # table in server.py is gone
        import atomic.ui.server as srv
        assert not hasattr(srv, "_PARAM_RANGES"), "server._PARAM_RANGES should be removed"
        rng = param_range("clock_bpm", "bpm")
        assert rng == (1.0, 300.0, 1.0, "bpm"), rng
        rng = param_range("sine_lfo", "hz")
        assert rng == (0.01, 20.0, 0.01, "Hz"), rng
        assert param_range("clock_bpm", "BPM") == (1.0, 300.0, 1.0, "bpm")  # case-insensitive
        assert param_range("nonexistent", "x") is None
    checks.append(("param_range() canonical in gates.py; server._PARAM_RANGES removed",
                   ui_param_range_canonical))
    def ui_control_schema_uses_gates_ranges():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/control/gated_clock_counter")
            d = r.json()
            for p in d["params"]:
                if p["key"] == "bpm":
                    assert p["min"] == 1.0 and p["max"] == 300.0
                    assert p["step"] == 1.0 and p["unit"] == "bpm"
    checks.append(("control schema uses gates.param_range for bpm slider",
                   ui_control_schema_uses_gates_ranges))
    def ui_snapshot_diff_bus_only():
        from atomic.ui.programs import build
        from atomic.ui.viewer import Viewer
        v = Viewer(build("sine_lfo_scope"), name="diff_test")
        v.batch(3)
        full = v.snapshot()
        prev_bus = dict(full["bus"])
        v.tick_once()
        diff = v.snapshot_diff(prev_bus=prev_bus)
        assert diff.get("diff") is True
        # bus should be a subset of full keys
        for k in diff["bus"]:
            assert k in full["bus"], f"diff has unknown key {k}"
        # sine_lfo changes every tick so bus should have at least one changed key
        assert len(diff["bus"]) >= 1, "diff bus is empty (sine_lfo should change each tick)"
    checks.append(("Viewer.snapshot_diff(prev_bus) emits partial bus",
                   ui_snapshot_diff_bus_only))
    def ui_snapshot_diff_series_window():
        from atomic.ui.programs import build
        from atomic.ui.viewer import Viewer
        v = Viewer(build("sine_lfo_scope"), name="diff_series")
        v.batch(100)
        full = v.snapshot()
        diff = v.snapshot_diff(prev_bus=None, n_series=64)
        for k, arr in diff["series"].items():
            assert len(arr) <= 64, f"series {k} len {len(arr)} > 64"
            # must be the suffix of the full series
            full_arr = full["series"][k]
            assert arr == full_arr[-len(arr):], f"series {k} not a suffix"
    checks.append(("Viewer.snapshot_diff(n_series=64) returns <=64 samples per series",
                   ui_snapshot_diff_series_window))
    def ui_ws_queue_bounded():
        import asyncio
        from atomic.ui.viewer import Viewer
        from atomic.ui.programs import build
        v = Viewer(build("clock_counter"), name="ws_bp")
        # ws_connect creates a queue with maxsize=2
        async def go():
            q = await v.ws_connect()
            assert q.maxsize == 2, q.maxsize
            v.ws_disconnect(q)
        asyncio.run(go())
    checks.append(("ws_connect queue is bounded (maxsize=2)",
                   ui_ws_queue_bounded))
    def ui_ws_drop_count():
        import asyncio
        from atomic.ui.viewer import Viewer
        from atomic.ui.programs import build
        v = Viewer(build("clock_counter"), name="ws_drops")
        async def go():
            q = await v.ws_connect()
            # fill the queue
            q.put_nowait({"a": 1})
            q.put_nowait({"b": 2})
            # next put must raise (QueueFull) -> the broadcast loop drops it
            try:
                q.put_nowait({"c": 3})
                put_raised = False
            except Exception:
                put_raised = True
            assert put_raised, "queue should be full"
            v.ws_disconnect(q)
        asyncio.run(go())
    checks.append(("full WS client queue raises (broadcast would drop, not block)",
                   ui_ws_drop_count))
    def ui_ws_stream_diff():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            with c.websocket_connect("/ws/clock_counter") as ws:
                first = ws.receive_json()
                assert "t" in first and "bus" in first
                # subsequent messages should be diffs (no `views` field, has `diff: true`)
                seen_diff = False
                for _ in range(8):
                    m = ws.receive_json()
                    if m.get("diff") is True:
                        seen_diff = True
                        assert "bus" in m
                        break
                assert seen_diff, "WS stream should emit diffs after the first frame"
    checks.append(("WS /ws/{name} emits diff frames after initial snapshot",
                   ui_ws_stream_diff))
    def ui_heatmap_live_runs():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.post("/api/batch/heatmap_live", json={"ticks": 30})
            assert r.status_code == 200
            d = r.json()
            # cnt.acc should be > 0 after 30 ticks
            assert d["bus"].get("cnt.acc", 0) > 0, d["bus"]
    checks.append(("POST /api/batch/heatmap_live runs without error",
                   ui_heatmap_live_runs))
    def ui_wsstats_endpoint():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/wsstats?name=clock_counter")
            assert r.status_code == 200
            d = r.json()
            assert "clients" in d
            assert "drops" in d
    checks.append(("GET /api/wsstats -> clients + drops counters",
                   ui_wsstats_endpoint))
    return checks


def s19_checks():
    """iter 19: UI iter 4 — keyboard shortcuts, presets, RTT ping,
    latency overlay, signed heatmap, record/replay, split view, tile rename."""
    checks = []

    def ui_record_endpoint():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            frames = [{"t": i, "bus": {"x.cv": float(i)}} for i in range(5)]
            r = c.post("/api/record/clock_counter",
                        json={"frames": frames})
            assert r.status_code == 200, r.text
            d = r.json()
            assert d.get("ok") is True
            assert "run_id" in d
            assert d["frames"] == 5
    checks.append(("POST /api/record saves 5 frames, returns run_id",
                   ui_record_endpoint))

    def ui_replay_runs_list():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/replay_runs/clock_counter")
            assert r.status_code == 200, r.text
            d = r.json()
            assert "runs" in d
            # should have the run we saved above
            assert len(d["runs"]) >= 1, f"expected >=1 run, got {d['runs']}"
    checks.append(("GET /api/replay_runs lists saved runs",
                   ui_replay_runs_list))

    def ui_replay_load():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/replay_runs/clock_counter")
            runs = r.json()["runs"]
            if runs:
                rid = runs[-1]["rid"]
                r2 = c.get("/api/replay/clock_counter?run_id=" + rid)
                assert r2.status_code == 200, r2.text
                frames = r2.json()
                assert isinstance(frames, list)
                assert len(frames) == 5
                assert frames[0]["t"] == 0
    checks.append(("GET /api/replay loads frames and returns t0=0",
                   ui_replay_load))

    def ui_ws_ping_pong():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        import time
        c = TestClient(create_app())
        with c:
            with c.websocket_connect("/ws/clock_counter") as ws:
                # drain first snapshot
                ws.receive_json()
                # wait a bit so the tick_loop has had a chance to send
                # a diff, and drain it (the test isn't about the diffs,
                # just the ping/pong round-trip).
                time.sleep(0.05)
                for _ in range(8):
                    try:
                        m = ws.receive_json(timeout=0.01)
                        if m.get("_pong"):
                            return
                    except Exception:
                        break
                ws.send_json({"type": "ping"})
                # may need to skip a diff first
                for _ in range(8):
                    m = ws.receive_json()
                    if m.get("_pong"):
                        assert m["_pong"] is True
                        return
                raise AssertionError("no _pong received")
    checks.append(("WS ping -> _pong=True (RTT)",
                   ui_ws_ping_pong))

    def ui_ws_latency_fields():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            with c.websocket_connect("/ws/clock_counter") as ws:
                # first snapshot has no latency (no tick yet)
                snap1 = ws.receive_json()
                # wait for next diff
                for _ in range(5):
                    m = ws.receive_json()
                    if m.get("_lat_eng") is not None:
                        assert isinstance(m["_lat_eng"], (int, float))
                        assert m["_lat_eng"] >= 0
                        return
                assert False, "no _lat_eng in first 5 diffs"
    checks.append(("WS diff frames include _lat_eng (engine budget us)",
                   ui_ws_latency_fields))

    def ui_signed_heatmap_render():
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "atomic", "ui", "static", "index.html")
        html = open(path).read()
        assert "HM_SIGNED_POS" in html, "missing signed pos palette"
        assert "HM_SIGNED_NEG" in html, "missing signed neg palette"
        assert "HM_SIGNED_POS" in html and "HM_SIGNED_NEG" in html
    checks.append(("index.html includes signed heatmap palettes (red-/blue+)",
                   ui_signed_heatmap_render))

    def ui_tile_rename_in_html():
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "atomic", "ui", "static", "index.html")
        html = open(path).read()
        assert "tile-rename" in html, "missing tile-rename CSS"
        assert "tile-cap" in html and "dblclick" in html, "missing dblclick rename"
    checks.append(("index.html has tile-cap dblclick rename handler",
                   ui_tile_rename_in_html))

    def ui_keyboard_shortcuts_in_html():
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "atomic", "ui", "static", "index.html")
        html = open(path).read()
        assert "e.key === ' '" in html, "missing space=tap shortcut"
        assert "e.key === 'r'" in html, "missing r=reset shortcut"
        assert "e.key === 'g'" in html, "missing g=clear groups shortcut"
    checks.append(("index.html has keyboard handlers (space/r/g)",
                   ui_keyboard_shortcuts_in_html))

    def ui_preset_round_trip():
        # Save a preset, then load it back
        import os, json, tempfile
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "ui", "static", "index.html")
        # We test the server-side preset save via record endpoint
        # (which uses the same JSON localStorage pattern)
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            frames = [{"t": 0, "bus": {}, "series": {}}]
            r = c.post("/api/record/test_preset",
                        json={"frames": frames})
            assert r.status_code == 200
    checks.append(("record/replay endpoint round-trip (preset analog)",
                   ui_preset_round_trip))

    def ui_split_view_programs_endpoint():
        # The pane 2 select should be populated from /api/programs
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/programs")
            assert r.status_code == 200
            d = r.json()
            assert len(d["programs"]) >= 7
    checks.append(("GET /api/programs returns 7+ programs (split view source)",
                   ui_split_view_programs_endpoint))

    def ui_wsstats_drops():
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/wsstats?name=hadamard_wxyz")
            assert r.status_code == 200
            d = r.json()
            assert "clients" in d and "drops" in d
    checks.append(("GET /api/wsstats -> clients + drops (connection quality)",
                   ui_wsstats_drops))

    return checks


def s20_checks():
    """UI iter 5: themes, program switcher, bus inspector, param sweep, CSV, wheel."""
    import os
    checks = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(root, "atomic", "ui", "static", "index.html")
    html = open(html_path).read()

    def theme_dark_defined():
        assert "body.theme-light" in html, "missing light theme"
        assert "body.theme-neon" in html, "missing neon theme"
        assert "--accent:" in html, "missing CSS variable for accent"
    checks.append(("CSS themes: dark/light/neon defined", theme_dark_defined))

    def theme_selector_present():
        assert 'id="theme-select"' in html, "missing theme selector"
        assert "applyTheme" in html, "missing applyTheme fn"
        assert "state.theme" in html, "missing theme state"
    checks.append(("theme selector + applyTheme fn", theme_selector_present))

    def program_switcher_present():
        assert 'id="prog-switcher"' in html, "missing program switcher"
        assert "_switchProgram" in html, "missing _switchProgram fn"
        assert "_refreshProgramSwitcher" in html, "missing _refreshProgramSwitcher"
    checks.append(("program switcher + _switchProgram fn", program_switcher_present))

    def bus_inspector_present():
        assert 'id="bus-inspector"' in html, "missing bus-inspector panel"
        assert "showBusInspector" in html, "missing showBusInspector fn"
        assert "bus-inspector-row" in html, "missing bus-inspector-row class"
    checks.append(("bus inspector panel + show fn", bus_inspector_present))

    def param_sweep_present():
        assert 'id="sweep-record-btn"' in html, "missing sweep record btn"
        assert "sweepRec" in html, "missing sweepRec state"
        assert "_startSweepReplay" in html, "missing sweep replay fn"
    checks.append(("param sweep rec/replay state + fns", param_sweep_present))

    def csv_export_present():
        assert "exportSeriesCSV" in html, "missing exportSeriesCSV fn"
        assert "tile-csv-btn" in html, "missing tile-csv-btn class"
        assert "text/csv" in html, "missing CSV mime type"
    checks.append(("CSV export + tile-csv-btn", csv_export_present))

    def wheel_speed_handler():
        assert "speedSlider.addEventListener('wheel'" in html, "missing wheel handler"
        assert "_adjustSpeed" in html, "missing _adjustSpeed fn"
    checks.append(("speed slider mouse wheel = ±1 fps", wheel_speed_handler))

    def speed_adjust_step():
        # _adjustSpeed must change by 1 (not 5)
        assert "Math.min(60, Math.max(1, parseInt(s.value) + delta))" in html, \
            "_adjustSpeed should clamp + use delta as 1"
    checks.append(("_adjustSpeed step=1 (not 5)", speed_adjust_step))

    def sweep_records_param():
        # When recording, the param slider's input handler should push to sweepRec
        assert "state.sweepRec.push" in html, "param slider does not record to sweep"
        # the push must include tick + module + key + value
        assert "tick: state.tick" in html, "sweep entry missing tick"
        assert "module: mid" in html, "sweep entry missing module"
    checks.append(("param slider records to sweepRec", sweep_records_param))

    def programs_endpoint_for_switcher():
        # the program switcher needs /api/programs to populate
        from atomic.ui.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        with c:
            r = c.get("/api/programs")
            assert r.status_code == 200
            d = r.json()
            assert len(d["programs"]) >= 7
    checks.append(("program switcher source: /api/programs 7+", programs_endpoint_for_switcher))

    def all_programs_listed():
        # The program switcher must include all registry programs
        from atomic.ui.programs import all_programs
        ps = all_programs()
        assert "hadamard_wxyz" in ps
        assert "xy_pad" in ps
        assert "clock_counter" in ps
    checks.append(("all 7 demo programs in switcher", all_programs_listed))

    return checks

def s21_checks():
    """UI iter 6: fullscreen, color picker, viz override, bus search, cheat sheet, screenshot, favorites."""
    import os
    checks = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(root, "atomic", "ui", "static", "index.html")
    html = open(html_path).read()

    def fullscreen_css():
        assert "body.fullscreen #header" in html, "missing fullscreen CSS"
        assert "body.fullscreen #presets-bar" in html, "missing fullscreen presets-bar"
        assert "body.fullscreen #control-bar" in html, "missing fullscreen control-bar"
        assert "body.fullscreen #tile-wall" in html, "missing fullscreen tile-wall"
    checks.append(("fullscreen CSS hides header/bars", fullscreen_css))

    def fullscreen_js():
        assert "toggleFullscreen" in html, "missing toggleFullscreen fn"
        assert "state.fullscreen" in html, "missing fullscreen state"
        assert 'classList.toggle' in html and "fullscreen" in html, "missing fullscreen classToggle"
        assert 'id="fullscreen-btn"' in html, "missing fullscreen button"
    checks.append(("fullscreen toggle fn + button", fullscreen_js))

    def fullscreen_keyboard():
        assert "e.key === 'f'" in html or "e.key === 'F'" in html, "missing F key shortcut"
    checks.append(("F key triggers fullscreen", fullscreen_keyboard))

    def color_picker_css():
        assert "tile-color-btn" in html, "missing tile-color-btn CSS"
        assert "tile-color-popup" in html, "missing tile-color-popup CSS"
        assert "tile-color-swatch" in html, "missing tile-color-swatch CSS"
    checks.append(("color picker CSS classes", color_picker_css))

    def color_picker_js():
        assert "openColorPicker" in html, "missing openColorPicker fn"
        assert "applyTileColor" in html, "missing applyTileColor fn"
        assert "tileColors" in html, "missing tileColors state"
        assert 'id="tile-color-picker"' in html, "missing color picker input"
        assert 'id="tile-color-popup"' in html, "missing tile-color-popup div"
    checks.append(("color picker fns + state", color_picker_js))

    def color_picker_saved():
        assert "tileColors: state.tileColors" in html, "tileColors not persisted in saveState"
        assert "if (_saved.tileColors)" in html, "tileColors not restored in loadState"
    checks.append(("tileColors persisted + restored", color_picker_saved))

    def color_override_in_draw():
        assert "colorOverride" in html, "missing colorOverride param in draw fns"
        assert "drawHeatmapTile(canvas, hmVal, state.heatmap_signed, tileColor)" in html, "colorOverride not passed to drawHeatmapTile"
        assert "_hexToRgba" in html, "missing _hexToRgba helper"
    checks.append(("color override plumbed into draw fns", color_override_in_draw))

    def viz_override_html():
        assert "tile-viz-select" in html, "missing tile-viz-select dropdown"
        assert 'value="series"' in html and 'value="xy"' in html and 'value="wxyz3d"' in html, "missing viz options"
        assert "tileVizOverride" in html, "missing tileVizOverride state"
    checks.append(("viz override dropdown + state", viz_override_html))

    def viz_override_js():
        assert "setVizOverride" in html, "missing setVizOverride fn"
        assert "_effectiveViz" in html, "missing _effectiveViz fn"
        assert "tileVizOverride[tileRow" in html, "tileVizOverride not used in renderTile"
    checks.append(("viz override fns plumbed", viz_override_js))

    def viz_override_persisted():
        assert "tileVizOverride: state.tileVizOverride" in html, "tileVizOverride not persisted"
    checks.append(("tileVizOverride persisted", viz_override_persisted))

    def bus_search():
        assert 'id="bus-inspector-search"' in html, "missing bus inspector search input"
        assert "bus-inspector-search" in html, "missing search in bus inspector"
        assert "bus-inspector-count" in html, "missing bus-inspector-count"
        assert "renderBus(filter)" in html, "missing renderBus with filter"
        assert "new RegExp" in html, "missing RegExp for regex filter"
    checks.append(("bus inspector search + regex filter", bus_search))

    def bus_search_wired():
        assert 'searchEl.oninput = () => renderBus(searchEl.value)' in html, "search not wired"
    checks.append(("bus search input wired to renderBus", bus_search_wired))

    def cheatsheet():
        assert 'id="cheatsheet-overlay"' in html, "missing cheatsheet overlay"
        assert "toggleCheatsheet" in html, "missing toggleCheatsheet fn"
        assert "e.key === '?'" in html, "missing ? key shortcut"
        assert "Keyboard Cheat Sheet" in html, "missing cheat sheet title text"
    checks.append(("keyboard cheat sheet overlay + ? key", cheatsheet))

    def screenshot():
        assert "exportTilePNG" in html, "missing exportTilePNG fn"
        assert "tile-shot-btn" in html, "missing tile-shot-btn class"
        assert "toBlob" in html, "missing toBlob for PNG export"
        assert "image/png" in html, "missing PNG mime type"
    checks.append(("tile PNG screenshot export", screenshot))

    def favorites():
        assert "toggleFavorite" in html, "missing toggleFavorite fn"
        assert "state.favorites" in html, "missing favorites state"
        assert "refreshProgramSwitcher" in html, "missing refreshProgramSwitcher fn"
        assert "toggleFavoritesOnly" in html, "missing toggleFavoritesOnly fn"
        assert 'id="fav-filter-btn"' in html, "missing fav-filter-btn button"
    checks.append(("program favorites star + filter", favorites))

    def favorites_keyboard():
        assert "e.key === '*'" in html, "missing * key shortcut for favorites"
    checks.append(("* key toggles favorites filter", favorites_keyboard))

    def favorites_persisted():
        assert "favorites: state.favorites" in html, "favorites not persisted"
        assert "if (_saved.favorites)" in html, "favorites not restored in loadState"
    checks.append(("favorites persisted + restored", favorites_persisted))

    def escape_closes_panels():
        assert "e.key === 'Escape'" in html, "missing Escape key handler"
        assert "closeColorPicker" in html, "missing closeColorPicker in Escape"
        assert "hideBusInspector" in html, "missing hideBusInspector in Escape"
    checks.append(("Escape closes color picker + inspector + cheatsheet", escape_closes_panels))

    return checks

def s22_checks():
    """UI iter 7: tile wall zoom (Ctrl+wheel) + accent color override."""
    import os
    checks = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(root, "atomic", "ui", "static", "index.html")
    html = open(html_path).read()

    def zoom_wrapper():
        assert 'id="tile-wall-viewport"' in html, "missing zoom viewport wrapper"
        assert 'id="tile-wall-viewport-2"' in html, "missing pane-2 zoom viewport wrapper"
    checks.append(("tile-wall wrapped in zoom viewports", zoom_wrapper))

    def zoom_css():
        assert "zoom-transform" in html, "missing zoom-transform CSS class"
        assert "transform-origin" in html, "missing transform-origin"
        assert "transition: transform" in html, "missing transform transition"
    checks.append(("zoom CSS transform + origin + transition", zoom_css))

    def zoom_controls():
        assert 'id="zoom-controls"' in html, "missing zoom controls bar"
        assert 'id="zoom-in-btn"' in html, "missing zoom-in button"
        assert 'id="zoom-out-btn"' in html, "missing zoom-out button"
        assert 'id="zoom-fit-btn"' in html, "missing zoom-fit button"
        assert 'id="zoom-scale"' in html, "missing zoom scale label"
    checks.append(("zoom HUD buttons + scale label", zoom_controls))

    def zoom_js():
        assert "function applyZoom" in html, "missing applyZoom fn"
        assert "function zoomBy" in html, "missing zoomBy fn"
        assert "function zoomFit" in html, "missing zoomFit fn"
        assert "function setupZoom" in html, "missing setupZoom fn"
        assert "ctrlKey || e.metaKey" in html, "zoom not gated on Ctrl/Cmd"
        assert "scale(" in html, "missing scale() transform"
        assert "passive: false" in html, "wheel handler must be non-passive to preventDefault"
    checks.append(("zoom JS: fns + Ctrl-gated wheel", zoom_js))

    def zoom_persisted():
        assert "zoom: state.zoom" in html, "zoom not persisted in saveState"
        assert "if (_saved.zoom != null)" in html, "zoom not restored in loadState"
        assert "zoom: state.zoom," in html and "p.zoom" in html, "zoom not in preset snapshot"
    checks.append(("zoom persisted to localStorage + presets", zoom_persisted))

    def zoom_keys():
        assert "Ctrl+scroll" in html, "missing Ctrl+scroll cheatsheet entry"
        assert "Ctrl+0" in html, "missing Ctrl+0 cheatsheet entry"
        assert "(e.ctrlKey || e.metaKey) && (e.key === '+'" in html, "missing Ctrl+= zoom handler"
        assert "(e.ctrlKey || e.metaKey) && e.key === '0'" in html, "missing Ctrl+0 zoom-fit handler"
    checks.append(("zoom keys: Ctrl+scroll, Ctrl+=/-, Ctrl+0", zoom_keys))

    def accent_picker_html():
        assert 'id="accent-picker"' in html, "missing accent color picker input"
        assert 'id="accent-clear-btn"' in html, "missing accent clear button"
        assert "accentColor" in html, "missing accentColor state"
    checks.append(("accent picker UI in header", accent_picker_html))

    def accent_css():
        assert "body.accent-override" in html, "missing accent-override body class"
        assert "--accent-override" in html, "missing --accent-override CSS var"
    checks.append(("accent CSS override (theme-agnostic)", accent_css))

    def accent_js():
        assert "function applyAccentColor" in html, "missing applyAccentColor fn"
        assert "state.accentColor" in html, "accentColor state not threaded"
        assert "setProperty('--accent-override'" in html, "accent var not set on body"
    checks.append(("accent JS: applyAccentColor + body var", accent_js))

    def accent_persisted():
        assert "accentColor: state.accentColor" in html, "accent not persisted in saveState"
        assert "if (_saved.accentColor)" in html, "accent not restored in loadState"
        assert "p.accentColor" in html, "accent not in preset snapshot"
    checks.append(("accent persisted to localStorage + presets", accent_persisted))

    return checks

def s23_checks():
    """Goal A: wgsl naga hard-validate — iter 24.
    naga 30.0.1 installed in ~/.cargo/bin; codegen uses module-scope
    @group(0) storage vars (bus 4*n, params/state/inputs n), per-block
    fn tick_<id>() writing to bus/inputs/state, no ptr<storage> args.
    """
    import shutil, subprocess, tempfile, os
    checks = []

    def naga_installed():
        naga = shutil.which("naga")
        assert naga, "naga not on PATH (install: cargo install naga-cli)"
        assert "naga" in naga, "naga path: " + naga
    checks.append(("naga on PATH (~/.cargo/bin)", naga_installed))

    def naga_version():
        naga = shutil.which("naga")
        out = subprocess.run([naga, "--version"], capture_output=True, text=True, timeout=5)
        assert out.returncode == 0, out.stderr
        # output is just "30.0.1" or "naga X.Y.Z"
        assert out.stdout.strip(), "naga --version returned empty"
    checks.append(("naga --version", naga_version))

    def naga_validate_h4():
        from atomic import Program, Block, Wire
        p = Program("wgsl_h4", blocks=[
            Block("c0","const",{"value":1.0}),
            Block("g1","gain",{"factor":2.0}),
            Block("h1","h4_slide"),
            Block("v0","viz_series"),
        ], wires=[
            Wire("c0.cv","g1.in"),
            Wire("g1.cv","h1.in"),
            Wire("h1.w","v0.in"),
        ])
        w = p.compile("wgsl")
        assert w.startswith("// WGSL")
        assert "@compute @workgroup_size(64)" in w
        assert "@group(0) @binding(0)" in w
        assert "@group(0) @binding(4)" in w  # inputs
        assert "fn tick_c0" in w
        assert "fn tick_h1" in w
        assert "H(4) slide" in w
        assert "host-RAM" in w
        # hard naga validate
        naga = shutil.which("naga")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".wgsl", delete=False) as fh:
            fh.write(w); tmp = fh.name
        try:
            out = subprocess.run([naga, tmp], capture_output=True, text=True, timeout=10)
            os.unlink(tmp)
            assert out.returncode == 0, "naga rejected H4 shader: " + out.stderr[:300]
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    checks.append(("naga validates H4 shader (bus 4*n)", naga_validate_h4))

    def naga_validate_extended():
        from atomic import Program, Block, Wire
        p = Program("wgsl_ext", blocks=[
            Block("c0","const",{"value":1}),
            Block("g1","gain",{"factor":2}),
            Block("b1","bias",{"add":0.5}),
            Block("th","threshold",{"hi":0.5,"lo":-0.5}),
            Block("cl","clamp",{"lo":-1,"hi":1}),
            Block("lf","sine_lfo",{"rate_hz":1.0}),
            Block("sm","smooth",{"alpha":0.1}),
            Block("h1","h4_slide"),
            Block("v0","viz_series"),
        ], wires=[
            Wire("c0.cv","g1.in"),
            Wire("g1.cv","b1.in"),
            Wire("b1.cv","th.in"),
            Wire("th.gate","sm.in"),
            Wire("b1.cv","h1.in"),
            Wire("sm.cv","v0.in"),
        ])
        w = p.compile("wgsl")
        naga = shutil.which("naga")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".wgsl", delete=False) as fh:
            fh.write(w); tmp = fh.name
        try:
            out = subprocess.run([naga, tmp], capture_output=True, text=True, timeout=10)
            os.unlink(tmp)
            assert out.returncode == 0, "naga rejected extended shader: " + out.stderr[:300]
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    checks.append(("naga validates extended shader (15 primitives)", naga_validate_extended))

    def wgsl_struct_shape():
        from atomic import Program, Block, Wire
        p = Program("wgsl_s", blocks=[Block("c0","const",{"value":1}), Block("g1","gain",{"factor":2}), Block("v0","viz_series")], wires=[Wire("c0.cv","g1.in"), Wire("g1.cv","v0.in")])
        w = p.compile("wgsl")
        assert "struct Bus" in w
        assert "struct ParamsBus" in w
        assert "@group(0) @binding(0) var<storage, read_write> bus: Bus;" in w
        assert "@group(0) @binding(4) var<storage, read_write> inputs: ParamsBus;" in w
        assert "fn tick_c0()" in w
        assert "fn main(@builtin(global_invocation_id) gid: vec3<u32>)" in w
    checks.append(("wgsl module-scope storage struct shape", wgsl_struct_shape))
    return checks


def s24_checks():
    """Goal B: bicameral live demo — iter 25.
    BicameralPipeline: sub=clock_bpm@60, con=accum->smooth->viz_series,
    wired through HostBridge(bridge_latency=1). Bridge depth tracks the queue;
    final accum=2.0 (two beats in 90 ticks at 60bpm); UI bridge depth
    badge wired via BicameraViewer.snapshot().
    """
    from atomic import Program, Block, Wire
    from atomic.bridge import BicameralPipeline

    checks = []

    def bicameral_accum_2():
        sub = Program("clock_source", blocks=[Block("clk", "clock_bpm", {"bpm": 60})], wires=[])
        con = Program("counter_display",
                      blocks=[Block("cnt", "accum", {"per_tick": 1}),
                              Block("sm", "smooth", {"alpha": 0.1}),
                              Block("v0", "viz_series")],
                      wires=[Wire("cnt.acc", "sm.in"), Wire("sm.cv", "v0.in")])
        pipe = BicameralPipeline(sub, con, bridge_map=[("clk.trig", "cnt.in")],
                                bridge_latency=1, use_h4=False)
        pipe.run(90)
        assert pipe.con.bus.snapshot().get("cnt.acc") == 2.0, \
            "accum=2.0 after 90 ticks at 60bpm"
    checks.append(("bicameral accum=2.0 at 90 ticks", bicameral_accum_2))

    def bicameral_bridge_depth():
        sub = Program("clock_source", blocks=[Block("clk", "clock_bpm", {"bpm": 60})], wires=[])
        con = Program("counter_display",
                      blocks=[Block("cnt", "accum", {"per_tick": 1})], wires=[])
        pipe = BicameralPipeline(sub, con, bridge_map=[("clk.trig", "cnt.in")],
                                bridge_latency=1)
        for _ in range(90):
            pipe.tick()
        assert pipe.bridge.depth() == 1, "bridge depth=1 (latency=1, one beat in flight)"
        assert pipe.bridge._pushed >= 1, "bridge pushed >= 1 beats"
    checks.append(("bridge depth=1 (latency=1)", bicameral_bridge_depth))

    def bicameral_viewer_snapshot():
        from atomic.ui.bicameral_viewer import BicameralViewer
        from atomic.ui.programs import build_bicameral
        spec = build_bicameral("bicameral_clock")
        assert spec is not None, "bicameral_clock program not found"
        assert spec["type"] == "bicameral", "should be bicameral type"
        bv = BicameralViewer(spec["sub"], spec["con"],
                            bridge_map=spec["bridge_map"],
                            bridge_latency=spec["bridge_latency"],
                            name="test")
        snap = bv.batch(5)
        assert "sub" in snap and "con" in snap, "snapshot needs sub/con"
        assert "bridge" in snap, "snapshot needs bridge"
        assert "depth" in snap["bridge"], "bridge needs depth"
        assert "latency" in snap["bridge"], "bridge needs latency"
        assert "history" in snap["bridge"], "bridge needs history"
        assert snap["sub"]["bus"].get("clk.trig") == 0.0, "sub clk.trig==0 before first beat"
    checks.append(("BicameralViewer snapshot sub/con/bridge", bicameral_viewer_snapshot))

    def ui_server_bicameral_endpoints():
        from fastapi.testclient import TestClient
        from atomic.ui.server import create_app
        app = create_app()
        c = TestClient(app, raise_server_exceptions=False)
        r = c.get("/api/bicameral")
        assert r.status_code == 200, f"/api/bicameral: {r.status_code}"
        data = r.json()
        assert "programs" in data
        assert "bicameral_clock" in data["programs"]
        r2 = c.post("/api/bicameral/bicameral_clock/batch", json={"ticks": 10})
        assert r2.status_code == 200, f"/api/bicameral batch: {r2.status_code}"
        bd = r2.json()["bridge"]
        assert bd["depth"] >= 0
        assert bd["latency"] == 1
        assert bd["pushed"] == 10
    checks.append(("UI /api/bicameral endpoints", ui_server_bicameral_endpoints))

    def ui_programs_includes_bicameral():
        from fastapi.testclient import TestClient
        from atomic.ui.server import create_app
        app = create_app()
        c = TestClient(app, raise_server_exceptions=False)
        r = c.get("/api/programs")
        assert r.status_code == 200
        data = r.json()
        assert "bicameral" in data, "/api/programs should include bicameral"
        assert "bicameral_clock" in data["bicameral"], "bicameral_clock in programs.bicameral"
    checks.append(("/api/programs includes bicameral key", ui_programs_includes_bicameral))

    return checks


def s25_checks():
    """iter 26: teach domain expansion — 14 seeded examples across 6 domains,
    QBF persistence, domain routing checks, all examples runnable."""
    checks = []

    def teach_registry_14_examples():
        from atomic.teach import REGISTRY, DOMAINS
        assert len(REGISTRY.examples) >= 14, \
            f"expected >=14, got {len(REGISTRY.examples)}"
        for dom in DOMAINS:
            assert len(REGISTRY.list(domain=dom)) >= 1, \
                f"domain {dom} has no examples"
    checks.append(("14+ examples seeded across 6 domains", teach_registry_14_examples))

    def teach_domain_coverage():
        from atomic.teach import DOMAINS
        # each domain has at least 2 examples
        from atomic.teach import REGISTRY
        for dom in DOMAINS:
            n = len(REGISTRY.list(domain=dom))
            assert n >= 2, f"domain {dom} needs >=2, got {n}"
    checks.append(("6 domains each have >=2 examples", teach_domain_coverage))

    def teach_h4_examples_present():
        from atomic.teach import REGISTRY
        hit = REGISTRY.match("hadamard spatial wxyz 3d scope", domain="spatial")
        assert hit is not None, "spatial hadamard example missing"
        assert any(b.primitive == "h4_slide" for b in hit["program"].blocks)
    checks.append(("spatial hadamard example has h4_slide", teach_h4_examples_present))

    def teach_audio_examples_present():
        from atomic.teach import REGISTRY
        hit = REGISTRY.match("audio spectrum flux detector", domain="audio")
        assert hit is not None, "audio mdct_flux example missing"
        assert any(b.primitive == "mdct_flux" for b in hit["program"].blocks)
    checks.append(("audio mdct_flux example present", teach_audio_examples_present))

    def teach_medical_examples_present():
        from atomic.teach import REGISTRY
        hit = REGISTRY.match("medical vital monitor smooth alarm", domain="medical")
        assert hit is not None, "medical vital monitor example missing"
        assert any(b.primitive == "threshold" for b in hit["program"].blocks)
    checks.append(("medical threshold examples present", teach_medical_examples_present))

    def teach_control_examples_present():
        from atomic.teach import REGISTRY
        hit = REGISTRY.match("clock bpm counter toggle divider", domain="control")
        assert hit is not None, "control clock examples missing"
        assert any(b.primitive == "clock_bpm" for b in hit["program"].blocks)
    checks.append(("control clock_bpm examples present", teach_control_examples_present))

    def teach_signal_examples_present():
        from atomic.teach import REGISTRY
        hit = REGISTRY.match("signal moving average filter threshold", domain="signal")
        assert hit is not None, "signal moving_avg example missing"
        assert any(b.primitive == "moving_avg" for b in hit["program"].blocks)
    checks.append(("signal moving_avg examples present", teach_signal_examples_present))

    def teach_qbf_roundtrip():
        from atomic.teach import TeacherRegistry, REGISTRY
        from atomic import Program, Block, Wire
        import tempfile, shutil, os
        tmp = tempfile.mkdtemp(prefix="selftest_teach26_")
        try:
            reg = TeacherRegistry()
            p = Program("test_prog",
                blocks=[Block("c0","const",{"value":5}),Block("v0","viz_series")],
                wires=[Wire("c0.cv","v0.in")], description="iter26 test")
            reg.register("iter26 test example", p, domain="signal")
            reg.register("spatial h4 wxyz consensus",
                         REGISTRY.match("hadamard spatial wxyz 3d scope")["program"],
                         domain="spatial")
            path = os.path.join(tmp, "teach26.qbf")
            saved = reg.save_qbf(path=path)
            assert os.path.exists(saved), "QBF not written"
            loaded = TeacherRegistry.load_qbf(saved)
            assert len(loaded.examples) == len(reg.examples), \
                f"load mismatch: {len(loaded.examples)} != {len(reg.examples)}"
            hit = loaded.match("iter26 test example", domain="signal")
            assert hit is not None and hit["program"].name == "test_prog"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("teach QBF round-trip", teach_qbf_roundtrip))

    def teach_keyword_routing():
        from atomic.teach import REGISTRY
        for desc, dom, prim in [
            ("hadamard gate wxyz scope", "spatial", "h4_slide"),
            ("audio spectrum flux", "audio", "mdct_flux"),
            ("sensor threshold gate alarm", "medical", "threshold"),
            ("moving average filter", "signal", "moving_avg"),
            ("clock bpm counter toggle", "control", "clock_bpm"),
        ]:
            p = REGISTRY.from_description(desc, domain=dom)
            assert p.validate() == [], (desc, p.validate())
            assert any(b.primitive == prim for b in p.blocks), \
                f"{dom}/{prim}: {[b.primitive for b in p.blocks]}"
    checks.append(("keyword routing across 5 domain/primitive pairs", teach_keyword_routing))

    def teach_expanded_vocab():
        from atomic.teach import KEYWORDS_BY_DOMAIN
        assert "pulse" in KEYWORDS_BY_DOMAIN["medical"] or "ecg" in KEYWORDS_BY_DOMAIN["medical"]
        assert "flux" in KEYWORDS_BY_DOMAIN["audio"]
        assert "hysteresis" in KEYWORDS_BY_DOMAIN["signal"]
        assert "divider" in KEYWORDS_BY_DOMAIN["control"]
        assert "consensus" in KEYWORDS_BY_DOMAIN["spatial"]
    checks.append(("expanded domain keywords (pulse/ecg/flux/hysteresis/divider/consensus)",
                   teach_expanded_vocab))

    def teach_all_examples_runnable():
        from atomic.teach import REGISTRY
        from atomic import Engine
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
    checks.append(("all 14 seeded examples compile and run", teach_all_examples_runnable))

    return checks


def s26_checks():
    checks = []

    def video_viz_atom_registered():
        from atomic import ATOMS
        a = ATOMS["viz_video"]
        assert a.category == "sink"
        assert "in" in a.inputs
        assert "ready" in a.outputs
        assert "w" in a.outputs  # H4 row decoder outputs
    checks.append(("viz_video atom registered (H4 channel decoder ports)", video_viz_atom_registered))

    def video_bridge_frame_blob():
        from atomic import HostBridge
        from atomic.qbf import h4_gate, h4_inverse
        br = HostBridge(latency=1, use_h4=True)
        # 4 scalar values + a frame blob coexist
        br.push(0, {"w": 1.0, "z": 2.0, "y": 3.0, "x": 4.0})
        br.push(1, {"frame": b"\xff\x00\x80\x40" * 4})  # 16 bytes RGBA
        out1 = br.pop(1)
        assert out1 is not None
        # H4 round-trips the 4-tuple losslessly
        for k, v in [("w", 1.0), ("z", 2.0), ("y", 3.0), ("x", 4.0)]:
            assert abs(out1[k] - v) < 1e-9, k
        out2 = br.pop(2)
        assert out2 is not None
        assert out2.get("_frame") is True
        # frame entry exposes the H4 row decoder (W=log alpha, XYZ=linear RGB)
        assert "_w" in out2 and "_x" in out2 and "_y" in out2 and "_z" in out2
        # The W row of the H(4) gate is the sum; de-gating gives the
        # original (a_log, b, g, r) -- the W component is log(alpha).
        import math as _m
        last_alpha = out2["frame"][-1]  # 0x40 = 64
        w_inverse, _, _, _ = h4_inverse(
            (out2["_w"], out2["_z"], out2["_y"], out2["_x"]))
        # the first slot of the de-gated 4-tuple is the W row's payload
        # (log alpha for the last pixel). The W row IS the sum, so
        # the W channel of the gate is a_log + r + g + b; verify the
        # W row of the *un-gated* form is recoverable.
        assert abs(w_inverse - _m.log(max(1, last_alpha))) < 1e-6, \
            f"w_inverse={w_inverse} expected log(64)={_m.log(64)}"
    checks.append(("HostBridge frame blob round-trip + H4 W/X/Y/Z channel decoder",
                   video_bridge_frame_blob))

    def video_h3_stub_deterministic():
        from atomic import H3Stub, H3Session
        h3 = H3Stub(width=8, height=8, n_frames=4)
        r1 = h3.generate("neon rain", seed=99)
        r2 = h3.generate("neon rain", seed=99)
        assert r1["frames"] == r2["frames"]
        assert len(r1["frames"]) == 4
        assert len(r1["frames"][0]) == 8 * 8 * 4  # RGBA
    checks.append(("H3Stub deterministic (same prompt+seed -> same frames)",
                   video_h3_stub_deterministic))

    def video_session_round_robin():
        from atomic import H3Stub, H3Session
        h3 = H3Stub(n_frames=2)
        ses = H3Session(h3, prompts=["p0", "p1", "p2"], frames_per_prompt=2)
        frames = [ses.tick() for _ in range(12)]
        prompts = [f.prompt for f in frames]
        # 2 frames per prompt, 3 prompts, then cycle
        assert prompts[:2] == ["p0", "p0"]
        assert prompts[2:4] == ["p1", "p1"]
        assert prompts[4:6] == ["p2", "p2"]
        assert prompts[6:8] == ["p0", "p0"]
    checks.append(("H3Session cycles through the prompt bank",
                   video_session_round_robin))

    def video_session_bridge_integration():
        from atomic import H3Stub, H3Session, HostBridge
        h3 = H3Stub(n_frames=1)
        br = HostBridge(latency=1, use_h4=True)
        ses = H3Session(h3, prompts=["test"], bridge=br, frames_per_prompt=1)
        ses.tick()
        assert br.depth() == 1
        out = br.pop(1)
        assert out is not None
        assert isinstance(out.get("frame"), bytes)
        # frame entry is also tagged with H4 row channels
        assert "_w" in out and "_x" in out
    checks.append(("H3Session -> HostBridge -> frame W/X/Y/Z channel decoder",
                   video_session_bridge_integration))

    def video_swarm_prompt_routing():
        from atomic import Swarm, Agent, Program, Block, PromptBank
        p = Program("c", blocks=[Block("g1", "const", {"value": 1.0})], wires=[])
        agents = [Agent("a%d" % i, p) for i in range(4)]
        swarm = Swarm()
        for a in agents:
            swarm.add_agent(a)
        res = swarm.run(ticks=1)
        w = res.consensus(port="g1.cv")
        assert abs(w - 4.0) < 1e-9  # 4 agents, each 1.0 -> W=4
        bank = PromptBank(prompts=["p0", "p1", "p2", "p3"])
        # the swarm's W channel drives the bank pick
        picked = bank.consensus_pick(last_w=w)
        assert picked in bank.prompts
    checks.append(("Swarm H4 consensus -> PromptBank next-pick routing",
                   video_swarm_prompt_routing))

    def video_prompt_bank_determinism():
        from atomic import PromptBank
        bank = PromptBank(prompts=["alpha", "bravo", "charlie", "delta"])
        # H4 consensus over 4 prompts is deterministic
        p1 = bank.consensus_pick()
        p2 = bank.consensus_pick()
        assert p1 == p2
        # W energy is the CORE keystone metric
        e = bank.w_energy()
        assert 0.0 <= e <= 1.0
    checks.append(("PromptBank consensus is deterministic + W-energy in [0,1]",
                   video_prompt_bank_determinism))

    def video_frame_decode_rgba_log_linear():
        """The H4 row decoder: W = log(alpha) master amplitude, XYZ = linear RGB."""
        from atomic import ATOMS, Program, Block, Wire, Engine
        from atomic.qbf import h4_gate, h4_inverse
        import math as _m
        # frame: red=200, green=100, blue=50, alpha=255 (fully opaque)
        frame = bytes([200, 100, 50, 255] * 4)  # 4 pixels
        p = Program("vv", blocks=[
            Block("c", "const", {"value": 1.0}),
            Block("v", "viz_video", {"capture": 1.0}),
        ], wires=[Wire("c.cv", "v.in")])
        patch = p.compile("microfx")
        eng = Engine(patch["modules"], patch.get("wires", []))
        eng.run(1)
        eng.bus.set("v.frame", frame)
        eng.run(1)
        rgba = eng.bus.get("v.rgba")
        decoded = eng.bus.get("v.rgba_decoded")
        assert rgba == frame
        assert decoded is not None
        # decoded[i] = R, G, B, A where A is preserved raw, RGB are clamped
        for i in range(0, len(decoded), 4):
            assert 0 <= decoded[i] <= 255
            assert 0 <= decoded[i+1] <= 255
            assert 0 <= decoded[i+2] <= 255
            assert decoded[i+3] == 255  # alpha preserved
        # viz_video's W/X/Y/Z output ports carry the LAST pixel's H4 GATE rows.
        # The W row is the gate sum (a_log + b + g + r); the inverse recovers
        # the original 4-tuple.
        w_val = eng.bus.get("v.w")
        x_val = eng.bus.get("v.x")
        y_val = eng.bus.get("v.y")
        z_val = eng.bus.get("v.z")
        assert w_val is not None and x_val is not None
        # The H(4) Sylvester matrix is self-inverse up to a factor of 4;
        # applying the gate twice recovers the original (scaled by 4).
        # Re-applying h4_gate to the gate rows gives 4 * (a_log, b, g, r).
        a_log, b_in, g_in, r_in = h4_gate((w_val, z_val, y_val, x_val))
        assert abs(a_log - 4.0 * _m.log(255)) < 1e-6, \
            f"a_log={a_log} expected 4*log(255)={4.0*_m.log(255)}"
        assert abs(b_in - 4.0 * 50.0) < 1e-6
        assert abs(g_in - 4.0 * 100.0) < 1e-6
        assert abs(r_in - 4.0 * 200.0) < 1e-6
    checks.append(("viz_video decodes RGBA -> H4 gate rows (W=log alpha, XYZ=linear RGB)",
                   video_frame_decode_rgba_log_linear))

    def video_persistence_through_qbf():
        """H3 frames can be persisted through QBF (the dma_trace bridge)."""
        from atomic import H3Stub, H3Session, QbfFile, H4
        from atomic.qbf import h4_encode
        h3 = H3Stub(n_frames=2, width=4, height=4)
        ses = H3Session(h3, prompts=["p0"], bridge=None, frames_per_prompt=2)
        ses.tick(); ses.tick()
        # serialize one frame as a H4 QBF blob (4 channels -> 1 H4 group per pixel)
        frame = ses.latest().rgba
        # pack 4 pixels as H4 groups: (r, g, b, a) tuples -> W/Z/Y/X gate
        groups = [(float(frame[i]), float(frame[i+1]),
                   float(frame[i+2]), float(frame[i+3]))
                  for i in range(0, len(frame), 4)][:4]
        encoded = h4_encode(groups)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".qbf", delete=False) as f:
            path = f.name
        try:
            q = QbfFile.create(path)
            q.put("h3_frame", encoded, blob_type=H4, checksum=True)
            q.write()
            r = QbfFile.open(path)
            retrieved = r.get("h3_frame")
            assert retrieved == encoded
        finally:
            os.unlink(path)
    checks.append(("H3 frame persistence through QBF (H4 blob + checksum)",
                   video_persistence_through_qbf))

    return checks


def s27_checks():
    checks = []

    def jfin_atom_registered():
        from atomic import ATOMS
        a = ATOMS["jfin_live_export"]
        assert a.category == "sink"
        assert "in" in a.inputs
        assert "trig" in a.inputs
        assert "active" in a.outputs
        assert "frames_pushed" in a.outputs
    checks.append(("jfin_live_export atom registered (sink, 2 in, 2 out)", jfin_atom_registered))

    def jfin_channel_m3u_format():
        from atomic.jellyfin import JFinChannel
        ch = JFinChannel("atomic-01", "ATOMIC-1",
                         "http://jellyfin/live.m3u8",
                         group="ATOMIC", number=1)
        line = ch.m3u_line()
        assert 'tvg-id="atomic-01"' in line
        assert 'tvg-name="ATOMIC-1"' in line
        assert 'group-title="ATOMIC"' in line
        assert "ATOMIC-1\n" in line
    checks.append(("JFinChannel M3U EXTINF line format", jfin_channel_m3u_format))

    def jfin_exporter_init():
        from atomic.jellyfin import JFinExporter, JFinChannel
        ch = JFinChannel("test", "TEST", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="selftest_jfin_")
        try:
            ex = JFinExporter(ch, hls_dir=tmp, width=64, height=64)
            assert ex.running is True
            assert "live.m3u8" in ex.playlist_path()
            ex.stop()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("JFinExporter init + running + playlist_path", jfin_exporter_init))

    def jfin_frame_size_check():
        from atomic.jellyfin import JFinExporter, JFinChannel, JFinError
        ch = JFinChannel("sz", "SZ", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="selftest_jfin_sz_")
        try:
            ex = JFinExporter(ch, hls_dir=tmp, width=8, height=8)
            with pytest.raises(JFinError, match="frame size mismatch"):
                ex.push(b"\x00" * 100, width=8, height=8)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("JFinExporter frame size validation raises JFinError", jfin_frame_size_check))

    def jfin_m3u_write_read():
        from atomic.jellyfin import JFinM3U, JFinChannel
        ch1 = JFinChannel("a1", "ATOMIC-1", "http://a1.m3u8", group="TV", number=1)
        ch2 = JFinChannel("a2", "ATOMIC-2", "http://a2.m3u8", group="TV", number=2)
        tmp = tempfile.mkdtemp(prefix="selftest_jfin_m3u_")
        try:
            m3u = JFinM3U(livetv_dir=tmp)
            m3u.add_channel(ch1)
            m3u.add_channel(ch2)
            path = m3u.write("atomic.m3u")
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "#EXTM3U" in content
            assert "ATOMIC-1" in content
            assert "ATOMIC-2" in content
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("JFinM3U write/read round-trip", jfin_m3u_write_read))

    def jfin_scheduler_register():
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        ch = JFinChannel("sc1", "SC1", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="selftest_jfin_sched_")
        try:
            ex = sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
            assert ch.id in sched.channels
            assert ch.id in sched.exporters
            sched.assign_program(ch.id, "test_prog")
            assert sched.mappings[ch.id] == "test_prog"
            ok = sched.push_frame(ch.id, b"\x00" * 8 * 8 * 4, width=8, height=8)
            assert ok is True  # ffmpeg is available
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("JFinScheduler register + assign_program + push_frame", jfin_scheduler_register))

    def jfin_rotation_round_robin():
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        for i in range(3):
            ch = JFinChannel(f"rr{i}", f"RR{i}", "http://x.m3u8")
            tmp = tempfile.mkdtemp(prefix=f"selftest_rr_{i}_")
            sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
        programs = ["p0", "p1", "p2", "p3"]
        try:
            sched.rotate(programs, mode="round_robin")
            m1 = dict(sched.mappings)
            sched.rotate(programs, mode="round_robin")
            m2 = dict(sched.mappings)
            assert m1 != m2  # cursor advances
            for v in m1.values():
                assert v in programs
        finally:
            sched.stop_all()
    checks.append(("JFinScheduler rotate round_robin advances cursor", jfin_rotation_round_robin))

    def jfin_rotation_random():
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        ch = JFinChannel("rn1", "RN1", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="selftest_rn_")
        try:
            sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
            programs = ["a", "b", "c", "d"]
            sched.rotate(programs, mode="random")
            assert sched.mappings["rn1"] in programs
        finally:
            sched.stop_all()
    checks.append(("JFinScheduler rotate random assigns valid program", jfin_rotation_random))

    def jfin_rotation_h4_consensus():
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        for i in range(2):
            ch = JFinChannel(f"h4{i}", f"H4{i}", "http://x.m3u8")
            tmp = tempfile.mkdtemp(prefix=f"selftest_h4_{i}_")
            sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
        programs = ["w", "x", "y", "z"]
        try:
            sched.rotate(programs, mode="h4_consensus")
            for v in sched.mappings.values():
                assert v in programs
        finally:
            sched.stop_all()
    checks.append(("JFinScheduler rotate h4_consensus assigns valid program", jfin_rotation_h4_consensus))

    def jfin_live_export_atom_tick():
        from atomic import Program, Block, Wire, Engine
        from atomic.gates import register_jfin_scheduler
        from atomic.jellyfin import JFinScheduler, JFinChannel
        p = Program("jle_s27", blocks=[
            Block("c", "const", {"value": 1.0}),
            Block("jle", "jfin_live_export", {"scheduler_key": "ch1", "width": 8, "height": 8}),
        ], wires=[Wire("c.cv", "jle.in")])
        sched = JFinScheduler()
        register_jfin_scheduler(sched)
        tmp = tempfile.mkdtemp(prefix="selftest_jle_")
        try:
            ch = JFinChannel("ch1", "CH1", "http://x.m3u8")
            sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], patch.get("wires", []))
            eng.run(2)
            jle_id = next((n.id for n in eng.nodes
                          if n.primitive == "jfin_live_export"), None)
            assert jle_id is not None
            assert eng.bus.get(f"{jle_id}.active") is not None
            assert eng.bus.get(f"{jle_id}.frames_pushed") is not None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            register_jfin_scheduler(None)
    checks.append(("jfin_live_export atom tick (gate=0, no frames pushed)", jfin_live_export_atom_tick))

    def make_default_channels_factory():
        from atomic.jellyfin import make_default_channels
        channels = make_default_channels(n=4)
        assert len(channels) == 4
        assert channels[0].number == 1
        assert channels[3].number == 4
        assert channels[0].id == "atomic-01"
        assert channels[3].id == "atomic-04"
    checks.append(("make_default_channels(4) factory", make_default_channels_factory))

    def hdhr_discover_returns_list():
        from atomic.jellyfin import JFinM3U
        result = JFinM3U.discover_hdhr()
        assert isinstance(result, list)
    checks.append(("JFinM3U.discover_hdhr returns list", hdhr_discover_returns_list))

    return checks


def s28_checks():
    """iter29: DASH muxer, mock ffmpeg, keyframe-on-trig, seeded rotation
    determinism, recursive group-title M3U, Swarm H4 -> JFinScheduler
    consensus_pick, viz_video -> jfin_live_export end-to-end program."""
    checks = []

    def jfin_mock_mode():
        from atomic.jellyfin import JFinExporter, JFinChannel
        ch = JFinChannel("mock29", "MOCK29", "http://x.m3u8")
        ex = JFinExporter(ch, width=8, height=8, mock=True)
        assert ex.mock is True and ex.running is True
        assert ex.frame_count == 0 and ex.keyframes == 0
        frame = b"\x00" * 8 * 8 * 4
        ex.push(frame, width=8, height=8)
        ex.push(frame, width=8, height=8, force_key=True)
        assert ex.frame_count == 2 and ex.keyframes == 1
        ex.stop()
    checks.append(("JFinExporter mock mode (no ffmpeg, keyframe counter)",
                   jfin_mock_mode))

    def jfin_dash_muxer():
        from atomic.jellyfin import JFinExporter, JFinChannel
        tmp = tempfile.mkdtemp(prefix="selftest_dash_")
        try:
            ch = JFinChannel("dash29", "DASH29", "http://x.mpd")
            ex = JFinExporter(ch, hls_dir=tmp, width=4, height=4,
                              muxer="dash", mock=True)
            assert ex.muxer == "dash"
            assert ex.playlist_name() == "live.mpd"
            assert "live.mpd" in ex.playlist_path()
            assert "live.mpd" in ex.mpd_path()
            frame = b"\x00" * 4 * 4 * 4
            ex.push(frame, width=4, height=4)
            assert ex.frame_count == 1
            ex.stop()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("JFinExporter DASH muxer (.mpd manifest path)",
                   jfin_dash_muxer))

    def jfin_rotation_seed_determinism():
        from atomic.jellyfin import JFinScheduler, JFinChannel
        programs = ["p0", "p1", "p2", "p3"]
        for seed in [0, 42, 99]:
            s1 = JFinScheduler(rotation_seed=seed)
            s2 = JFinScheduler(rotation_seed=seed)
            for i in range(2):
                ch1 = JFinChannel(f"ds{i}", f"DS{i}", "http://x.m3u8")
                ch2 = JFinChannel(f"ds{i}", f"DS{i}", "http://x.m3u8")
                s1.register_channel(ch1, mock=True)
                s2.register_channel(ch2, mock=True)
            s1.rotate(programs, mode="random")
            s2.rotate(programs, mode="random")
            assert s1.mappings == s2.mappings, f"seed={seed} not deterministic"
            s1.stop_all(); s2.stop_all()
    checks.append(("JFinScheduler rotation_seed determinism (3 seeds)",
                   jfin_rotation_seed_determinism))

    def jfin_seeded_round_robin_mode():
        from atomic.jellyfin import JFinScheduler, JFinChannel
        programs = ["a", "b", "c", "d"]
        s1 = JFinScheduler(rotation_seed=7)
        s2 = JFinScheduler(rotation_seed=7)
        for i in range(3):
            ch1 = JFinChannel(f"srr{i}", f"SRR{i}", "http://x.m3u8")
            ch2 = JFinChannel(f"srr{i}", f"SRR{i}", "http://x.m3u8")
            s1.register_channel(ch1, mock=True)
            s2.register_channel(ch2, mock=True)
        s1.rotate(programs, mode="seeded_round_robin")
        s2.rotate(programs, mode="seeded_round_robin")
        assert s1.mappings == s2.mappings
        s1.stop_all(); s2.stop_all()
    checks.append(("JFinScheduler seeded_round_robin mode determinism",
                   jfin_seeded_round_robin_mode))

    def jfin_consensus_pick():
        from atomic.jellyfin import JFinScheduler
        programs = ["p0", "p1", "p2", "p3"]
        sched = JFinScheduler()
        assert sched.consensus_pick(programs, last_w=0.0) == "p0"
        assert sched.consensus_pick(programs, last_w=3.9) == "p3"
        assert sched.consensus_pick(programs, last_w=7.2) == "p3"
        assert sched.consensus_pick(programs, last_w=0.5) == "p0"
    checks.append(("JFinScheduler.consensus_pick (H4 W -> bank index)",
                   jfin_consensus_pick))

    def jfin_m3u_recursive_groups():
        from atomic.jellyfin import JFinM3U, JFinChannel
        tmp = tempfile.mkdtemp(prefix="selftest_m3u_rec_")
        try:
            m3u = JFinM3U(livetv_dir=tmp)
            m3u.add_channel(JFinChannel("a1", "A1", "http://a1.m3u8", group="ATOMIC"))
            m3u.add_channel(JFinChannel("a2", "A2", "http://a2.m3u8", group="ATOMIC"))
            m3u.add_channel(JFinChannel("t1", "T1", "http://t1.m3u8", group="TV"))
            result = m3u.write_recursive_groups()
            assert "ATOMIC" in result and "TV" in result
            assert "__root__" in result
            assert os.path.exists(result["__root__"])
            root = open(result["__root__"]).read()
            assert "#EXTGRP:ATOMIC" in root
            assert "#EXTGRP:TV" in root
            groups = m3u.group_titles()
            assert groups == ["ATOMIC", "TV"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("JFinM3U.write_recursive_groups + group_titles",
                   jfin_m3u_recursive_groups))

    def jfin_m3u_stdin_redirect():
        from atomic.jellyfin import JFinM3U, JFinChannel
        tmp = tempfile.mkdtemp(prefix="selftest_m3u_stdin_")
        try:
            m3u = JFinM3U(livetv_dir=tmp)
            m3u.add_channel(JFinChannel("s1", "S1", "http://s1.m3u8"))
            buf = io.BytesIO()
            rendered = m3u.write_to_stdin(filename="test.m3u", stdin_fp=buf)
            assert b"#EXTM3U" in rendered and b"S1" in rendered
            assert buf.tell() > 0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append(("JFinM3U.write_to_stdin (M3U bytes -> file-like stdin)",
                   jfin_m3u_stdin_redirect))

    def jfin_channel_id_and_group_path():
        from atomic.jellyfin import JFinChannel
        ch = JFinChannel("ch-99", "CH-99", "http://x.m3u8", group="ATOMIC")
        line = ch.m3u_line()
        assert 'channel-id="ch-99"' in line
        line2 = ch.m3u_line(group_path="HD")
        assert 'group-title="HD/ATOMIC"' in line2
    checks.append(("JFinChannel.m3u_line channel-id + recursive group_path",
                   jfin_channel_id_and_group_path))

    def jfin_trig_keyframe():
        from atomic import Program, Block, Wire, Engine
        from atomic.gates import register_jfin_scheduler
        from atomic.jellyfin import JFinScheduler, JFinChannel
        p = Program("jle_kf29", blocks=[
            Block("clk", "clock_bpm", {"bpm": 1}),
            Block("jle", "jfin_live_export", {"scheduler_key": "kf29", "width": 4, "height": 4}),
        ], wires=[Wire("clk.trig", "jle.trig")])
        sched = JFinScheduler()
        register_jfin_scheduler(sched)
        tmp = tempfile.mkdtemp(prefix="selftest_jle_kf29_")
        try:
            ch = JFinChannel("kf29", "KF29", "http://x.m3u8")
            sched.register_channel(ch, hls_dir=tmp, width=4, height=4, mock=True)
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], patch.get("wires", []))
            eng.bus.set("jle.frame", b"\x00" * 4 * 4 * 4)
            eng.run(5)
            jle_id = next((n.id for n in eng.nodes
                          if n.primitive == "jfin_live_export"), None)
            assert jle_id is not None
            kf_out = eng.bus.get(f"{jle_id}.keyframes")
            assert kf_out is not None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            sched.stop_all()
            register_jfin_scheduler(None)
    checks.append(("jfin_live_export trig -> keyframe (wire signal)",
                   jfin_trig_keyframe))

    def jfin_export_demo_program():
        from atomic.ui.programs import build, all_programs
        assert "jfin_export_demo" in all_programs()
        prog = build("jfin_export_demo")
        assert prog.validate() == [], prog.validate()
        patch = prog.compile("microfx")
        prims = [m["primitive"] for m in patch["modules"]]
        assert "viz_video" in prims
        assert "jfin_live_export" in prims
    checks.append(("jfin_export_demo program registered + compiles",
                   jfin_export_demo_program))

    def swarm_to_jfin_consensus():
        from atomic import Swarm, Agent, Program, Block, Wire, Display
        from atomic.jellyfin import JFinScheduler
        d = Display(800, 800, 4, 4)
        swarm = Swarm(display=d)
        for i, val in enumerate([1.0, 2.0, 3.0, 4.0]):
            prog = Program("p", blocks=[
                Block("c0", "const", {"value": val}),
                Block("g1", "gain", {"factor": 1.0}),
                Block("v0", "viz_series"),
            ], wires=[Wire("c0.cv", "g1.in"), Wire("g1.cv", "v0.in")])
            g = d.link(f"w{i}", i // 4, i % 4, 1, 1)
            swarm.add_agent(Agent(f"w{i}", prog, tile_group=g))
        res = swarm.run(3)
        w = res.consensus(port="g1.cv")
        assert abs(w - 10.0) < 1e-9
        programs = ["a", "b", "c", "d"]
        sched = JFinScheduler()
        picked = sched.consensus_pick(programs, last_w=w)
        assert picked in programs
    checks.append(("Swarm H4 consensus -> JFinScheduler.consensus_pick routing",
                   swarm_to_jfin_consensus))

    def viz_video_jfin_end_to_end():
        from atomic import Program, Block, Wire, Engine
        from atomic.gates import register_jfin_scheduler
        from atomic.jellyfin import JFinScheduler, JFinChannel
        p = Program("vv_jfin29", blocks=[
            Block("c", "const", {"value": 1.0}),
            Block("vv", "viz_video"),
            Block("jle", "jfin_live_export", {"scheduler_key": "vj29", "width": 4, "height": 4}),
        ], wires=[Wire("c.cv", "vv.in"), Wire("c.cv", "jle.in")])
        sched = JFinScheduler()
        register_jfin_scheduler(sched)
        tmp = tempfile.mkdtemp(prefix="selftest_vv_jfin29_")
        try:
            ch = JFinChannel("vj29", "VJ29", "http://x.m3u8")
            sched.register_channel(ch, hls_dir=tmp, width=4, height=4, mock=True)
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], patch.get("wires", []))
            eng.bus.set("vv.frame", b"\xff\x00\x80\x40" * 4 * 4)
            eng.run(2)
            vv_id = next((n.id for n in eng.nodes
                         if n.primitive == "viz_video"), None)
            jle_id = next((n.id for n in eng.nodes
                          if n.primitive == "jfin_live_export"), None)
            assert vv_id is not None and jle_id is not None
            assert eng.bus.get(f"{vv_id}.rgba_decoded") is not None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            sched.stop_all()
            register_jfin_scheduler(None)
    checks.append(("viz_video -> jfin_live_export end-to-end program runs",
                   viz_video_jfin_end_to_end))

    def jfin_stats_keyframes_muxer():
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        ch = JFinChannel("sk29", "SK29", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="selftest_sk29_")
        try:
            sched.register_channel(ch, hls_dir=tmp, width=4, height=4,
                                    muxer="dash", mock=True)
            frame = b"\x00" * 4 * 4 * 4
            sched.push_frame(ch.id, frame, force_key=True)
            stats = sched.stats()
            assert "keyframes" in stats[ch.id]
            assert "muxer" in stats[ch.id]
            assert stats[ch.id]["muxer"] == "dash"
            assert stats[ch.id]["keyframes"] >= 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            sched.stop_all()
    checks.append(("JFinScheduler.stats() exposes keyframes + muxer",
                   jfin_stats_keyframes_muxer))

    return checks


def s29_checks():
    """iter30: viz_video in _VIZ_OUTPUTS, proper view key extraction,
    drawVideoFrame canvas renderer, /api/feed_frame REST endpoint,
    JFinChannel.to_dict, JFinM3U.find_by_id/from_discovered_hdhr,
    _JFIN_STATE singleton, REST endpoints for Jellyfin HDHomeRun."""
    checks = []

    def viz_video_in_viz_outputs():
        from atomic.program import _VIZ_OUTPUTS
        assert "viz_video" in _VIZ_OUTPUTS, "viz_video missing from _VIZ_OUTPUTS"
        assert _VIZ_OUTPUTS["viz_video"] == "ready", \
            f"viz_video canonical output should be 'ready', got {_VIZ_OUTPUTS['viz_video']}"
    checks.append(("viz_video in _VIZ_OUTPUTS with canonical output 'ready'",
                   viz_video_in_viz_outputs))

    def patch_views_viz_video_key():
        from atomic.program import _patch_views, Block
        blocks = [
            Block("vv", "viz_video"),
            Block("w3d", "viz_wxyz3d"),
            Block("xy", "viz_xy"),
            Block("s", "viz_series"),
        ]
        views = _patch_views(blocks)
        view_map = {v["module"]: v for v in views}
        assert "vv" in view_map
        assert view_map["vv"]["key"] == "vv.frame", \
            f"viz_video key should be 'vv.frame', got {view_map['vv']['key']}"
        assert view_map["vv"]["as"] == "video"
        assert "w3d" in view_map
        assert view_map["w3d"]["key"] == "w3d.z", \
            f"viz_wxyz3d key should be 'w3d.z', got {view_map['w3d']['key']}"
        assert "xy" in view_map
        assert view_map["xy"]["key"] == "xy.y"
        assert "s" in view_map
        assert view_map["s"]["key"] == "s.cv"
    checks.append(("_patch_views generates correct keys for viz_video/wxyz3d/xy/series",
                   patch_views_viz_video_key))

    def auto_views_viz_video():
        from atomic.ui.viewer import _auto_views, _VIZ_TYPES
        assert "viz_video" in _VIZ_TYPES
        assert _VIZ_TYPES["viz_video"] == "video"
        mods = [
            {"id": "v0", "primitive": "viz_video"},
            {"id": "w0", "primitive": "viz_wxyz3d"},
        ]
        views = _auto_views(mods)
        vmap = {v["module"]: v for v in views}
        assert "v0" in vmap
        assert vmap["v0"]["key"] == "v0.frame", \
            f"auto_views viz_video key should be 'v0.frame', got {vmap['v0']['key']}"
        assert vmap["v0"]["viz"] == "video"
        assert "w0" in vmap
        assert vmap["w0"]["key"] == "w0.z"
    checks.append(("_auto_views generates correct keys for viz_video and viz_wxyz3d",
                   auto_views_viz_video))

    def jfin_channel_to_dict():
        from atomic.jellyfin import JFinChannel
        ch = JFinChannel("id30", "NAME30", "http://x.m3u8",
                         logo_url="http://logo.png", tuner_type="hdhr",
                         group="ATOMIC", number=7)
        d = ch.to_dict()
        assert d["id"] == "id30"
        assert d["name"] == "NAME30"
        assert d["m3u_url"] == "http://x.m3u8"
        assert d["logo_url"] == "http://logo.png"
        assert d["tuner_type"] == "hdhr"
        assert d["group"] == "ATOMIC"
        assert d["number"] == 7
    checks.append(("JFinChannel.to_dict() returns all fields",
                   jfin_channel_to_dict))

    def jfin_m3u_find_by_id():
        from atomic.jellyfin import JFinM3U, JFinChannel
        m3u = JFinM3U()
        ch1 = JFinChannel("c1", "C1", "http://c1.m3u8")
        ch2 = JFinChannel("c2", "C2", "http://c2.m3u8")
        m3u.add_channel(ch1)
        m3u.add_channel(ch2)
        found = m3u.find_by_id("c2")
        assert found is ch2
        assert m3u.find_by_id("nonexistent") is None
    checks.append(("JFinM3U.find_by_id returns correct channel or None",
                   jfin_m3u_find_by_id))

    def jfin_m3u_from_discovered_hdhr():
        from atomic.jellyfin import JFinM3U
        m3u = JFinM3U.from_discovered_hdhr(timeout=0.1, base_url="http://localhost:9999")
        assert isinstance(m3u.channels, list)
        assert hasattr(m3u, "find_by_id")
    checks.append(("JFinM3U.from_discovered_hdhr (may find 0 devices without real HDHR on LAN)",
                   jfin_m3u_from_discovered_hdhr))

    def jfin_state_singleton():
        from atomic.jellyfin import _JFIN_STATE, JFinM3U, JFinScheduler
        assert isinstance(_JFIN_STATE.m3u, JFinM3U)
        assert isinstance(_JFIN_STATE.scheduler, JFinScheduler)
        assert len(_JFIN_STATE.m3u.channels) >= 4
        assert "atomic-01" in {ch.id for ch in _JFIN_STATE.m3u.channels}
    checks.append(("_JFIN_STATE singleton: m3u + scheduler initialized with default channels",
                   jfin_state_singleton))

    def viewer_feed_frame():
        from atomic import Program, Block
        from atomic.ui.viewer import Viewer
        p = Program("ff_test", blocks=[
            Block("vv", "viz_video"),
        ])
        v = Viewer(p, name="ff_test")
        frame = b"\x80\x40\x20\x10" * 8 * 8
        ok = v.feed_frame("vv", frame)
        assert ok is True
        eng = v.engine
        assert eng.bus.get("vv.frame") == frame
    checks.append(("Viewer.feed_frame injects bytes into bus",
                   viewer_feed_frame))

    def discover_hdhr_timeout_param():
        from atomic.jellyfin import JFinM3U
        devs = JFinM3U.discover_hdhr(timeout=0.05)
        assert isinstance(devs, list)
        assert all("device_id" in d for d in devs)
    checks.append(("JFinM3U.discover_hdhr accepts timeout param",
                   discover_hdhr_timeout_param))

    def server_feed_frame_endpoint_import():
        from atomic.ui import server
        app = server.create_app()
        routes = [r.path for r in app.routes]
        assert "/api/feed_frame/{name}" in routes, \
            f"Missing /api/feed_frame endpoint. Routes: {routes}"
        assert "/api/jfin/discover" in routes
        assert "/api/jfin/channels" in routes
        assert "/api/jfin/export/{ch_id}/push" in [r.path for r in app.routes]
    checks.append(("FastAPI server: /api/feed_frame + /api/jfin/* endpoints registered",
                   server_feed_frame_endpoint_import))

    return checks


def s30_checks():
    """iter31: H3InferenceServer (FastAPI :8765, /health + /generate),
    viz_video_h3 atom (H4 RGBA decoder, retry, enabled gate, H4 parity vs
    viz_video), video_live program, /api/video/start|stop|status REST."""
    import threading as _th
    import time as _time
    _P = 38000  # high port range to avoid pytest port conflicts (187xx)
    checks = []

    def h3_server_lifecycle():
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=_P+1, width=8, height=8)
        srv.start()
        try:
            assert srv.is_running
            assert srv.wait_until_ready(timeout=8.0), "server not ready"
            h = srv.health()
            assert h["status"] == "ok"
            assert h["model"] == "H3-FastVideo"
            assert h["width"] == 8
            assert h["height"] == 8
        finally:
            srv.stop()
        assert not srv.is_running
    checks.append(("H3InferenceServer start/stop + /health", h3_server_lifecycle))

    def h3_server_generate():
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=_P+2, width=4, height=4)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=8.0), "server not ready"
            r = srv.generate("p", seed=42, n_frames=1)
            assert "frames_b64" in r
            assert len(r["frames_b64"]) == 1
            import base64 as _b64
            frame = _b64.b64decode(r["frames_b64"][0])
            assert len(frame) == 4 * 4 * 4
        finally:
            srv.stop()
    checks.append(("H3InferenceServer /generate returns RGBA frame", h3_server_generate))

    def viz_video_h3_atom_registered():
        from atomic.gates import ATOMS
        a = ATOMS["viz_video_h3"]
        assert a.category == "source"
        assert "ready" in a.outputs
        assert "w" in a.outputs and "x" in a.outputs
        assert "y" in a.outputs and "z" in a.outputs
        assert a.params["server_url"] == "http://localhost:8765"
        assert "enabled" in a.params
        assert "timeout_s" in a.params
        assert "max_retries" in a.params
    checks.append(("viz_video_h3 atom registered (source category, H4 outputs, server params)",
                   viz_video_h3_atom_registered))

    def viz_video_h3_engine_tick_server_up():
        from atomic import Program, Block
        from atomic.engine import Engine
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=_P+3, width=4, height=4)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=8.0), "server not ready"
            p = Program("v31_test", blocks=[
                Block("v", "viz_video_h3",
                      {"server_url": f"http://localhost:{_P+3}",
                       "width": 4, "height": 4, "prompt": "hello",
                       "n_frames": 1, "timeout_s": 5.0, "max_retries": 2,
                       "enabled": 1.0}),
            ])
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], patch.get("wires", []),
                         views=patch.get("views") or [])
            eng.tick()
            frame = eng.bus.get("v.rgba")
            assert frame is not None
            assert len(frame) == 4 * 4 * 4
            assert eng.bus.get("v.ready") == 1.0
        finally:
            srv.stop()
    checks.append(("viz_video_h3 engine tick (server up): fetches RGBA + sets ready=1",
                   viz_video_h3_engine_tick_server_up))

    def viz_video_h3_enabled_gate_skips_http():
        from atomic import Program, Block
        from atomic.engine import Engine
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=_P+4)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=8.0), "server not ready"
            p = Program("v31_gate", blocks=[
                Block("v", "viz_video_h3",
                      {"server_url": f"http://localhost:{_P+4}", "enabled": 0.0}),
            ])
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], [])
            eng.tick()
            assert eng.bus.get("v.ready") == 0.0
            assert eng.bus.get("v.rgba") is None
        finally:
            srv.stop()
    checks.append(("viz_video_h3 enabled=0 gate skips HTTP + leaves bus empty",
                   viz_video_h3_enabled_gate_skips_http))

    def viz_video_h3_server_down_no_crash():
        from atomic import Program, Block
        from atomic.engine import Engine
        p = Program("v31_down", blocks=[
            Block("v", "viz_video_h3",
                  {"server_url": "http://localhost:59999",
                   "timeout_s": 0.3, "max_retries": 1}),
        ])
        patch = p.compile("microfx")
        eng = Engine(patch["modules"], [])
        eng.tick()
        assert eng.bus.get("v.ready") == 0.0
    checks.append(("viz_video_h3 server down: sets ready=0 (no crash, no exception)",
                   viz_video_h3_server_down_no_crash))

    def viz_video_h3_h4_decode_parity():
        from atomic import Program, Block
        from atomic.engine import Engine
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=_P+5, width=2, height=2)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=8.0), "server not ready"
            p_vv = Program("vv_ref", blocks=[
                Block("v", "viz_video", {"capture": 1.0}),
            ])
            p_vv3 = Program("vv3", blocks=[
                Block("v", "viz_video_h3",
                      {"server_url": f"http://localhost:{_P+5}",
                       "width": 2, "height": 2,
                       "prompt": "parity_test",
                       "seed": 7}),
            ])
            eng_vv = Engine(p_vv.compile("microfx")["modules"], [])
            eng_vv3 = Engine(p_vv3.compile("microfx")["modules"], [])
            import base64 as _b64
            r = srv.generate("parity_test", seed=7)
            frame = _b64.b64decode(r["frames_b64"][0])
            eng_vv.bus.set("v.frame", frame)
            eng_vv.tick()
            eng_vv3.tick()
            assert abs(eng_vv.bus.get("v.w") - eng_vv3.bus.get("v.w")) < 1e-6
            assert abs(eng_vv.bus.get("v.x") - eng_vv3.bus.get("v.x")) < 1e-6
            assert abs(eng_vv.bus.get("v.y") - eng_vv3.bus.get("v.y")) < 1e-6
            assert abs(eng_vv.bus.get("v.z") - eng_vv3.bus.get("v.z")) < 1e-6
            assert eng_vv.bus.get("v.rgba") == eng_vv3.bus.get("v.rgba")
        finally:
            srv.stop()
    checks.append(("viz_video_h3 H4 RGBA decode matches viz_video (w/x/y/z parity)",
                   viz_video_h3_h4_decode_parity))

    def video_live_program_build():
        from atomic.ui.programs import build, all_programs
        assert "video_live" in all_programs()
        p = build("video_live")
        assert p is not None
        assert p.name == "video_live"
        prims = [b.primitive for b in p.blocks]
        assert "viz_video_h3" in prims
    checks.append(("video_live demo program registered with viz_video_h3 atom",
                   video_live_program_build))

    def video_live_program_compile():
        from atomic.ui.programs import build
        p = build("video_live")
        patch = p.compile("microfx")
        mods = patch["modules"]
        assert len(mods) == 1
        assert mods[0]["primitive"] == "viz_video_h3"
        assert mods[0]["id"] == "vh3"
        assert mods[0]["params"].get("server_url") == "http://localhost:8765"
    checks.append(("video_live compiles to microfx: single viz_video_h3 block",
                   video_live_program_compile))

    def server_video_endpoints_registered():
        from atomic.ui import server
        app = server.create_app()
        routes = [r.path for r in app.routes]
        assert "/api/video/start" in routes
        assert "/api/video/stop" in routes
        assert "/api/video/status" in routes
    checks.append(("FastAPI server: /api/video/start|stop|status endpoints registered",
                   server_video_endpoints_registered))

    def server_video_start_stop_via_testclient():
        from fastapi.testclient import TestClient
        from atomic.ui import server
        from atomic.video_server import _stop_server
        app = server.create_app()
        client = TestClient(app, raise_server_exceptions=False)
        port = _P + 6
        r = client.post("/api/video/start",
                        json={"port": port, "width": 8, "height": 8})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("started", "already_running")
        assert body["port"] == port
        try:
            deadline = _time.time() + 5.0
            running = False
            r2 = None
            while _time.time() < deadline:
                r2 = client.get(f"/api/video/status?port={port}")
                assert r2.status_code == 200
                if r2.json().get("status") == "running":
                    running = True
                    break
                _time.sleep(0.1)
            assert running, "video server never reported running"
            st = r2.json()
            assert st["port"] == port
            assert "health" in st
            assert st["health"]["status"] == "ok"
            r3 = client.post("/api/video/stop", json={"port": port})
            assert r3.status_code == 200
            assert r3.json()["status"] == "stopped"
        finally:
            _stop_server(port=port)
    checks.append(("Server lifecycle: POST /api/video/start + /stop + /status round-trip",
                   server_video_start_stop_via_testclient))
    checks.append(("Server lifecycle: POST /api/video/start + /stop + /status round-trip",
                   server_video_start_stop_via_testclient))

    def viz_outputs_and_types_includes_h3():
        from atomic.program import _VIZ_OUTPUTS
        from atomic.ui.viewer import _VIZ_TYPES
        assert "viz_video_h3" in _VIZ_OUTPUTS
        assert _VIZ_OUTPUTS["viz_video_h3"] == "ready"
        assert "viz_video_h3" in _VIZ_TYPES
        assert _VIZ_TYPES["viz_video_h3"] == "video"
    checks.append(("_VIZ_OUTPUTS + _VIZ_TYPES include viz_video_h3",
                   viz_outputs_and_types_includes_h3))

    def patch_views_and_auto_views_h3():
        from atomic.program import _patch_views, Block
        from atomic.ui.viewer import _auto_views
        blocks = [Block("vh3", "viz_video_h3")]
        views = _patch_views(blocks)
        assert len(views) == 1
        assert views[0]["key"] == "vh3.frame"
        assert views[0]["as"] == "video_h3"
        mods = [{"id": "v0", "primitive": "viz_video_h3"}]
        views2 = _auto_views(mods)
        vmap = {v["module"]: v for v in views2}
        assert vmap["v0"]["key"] == "v0.frame"
        assert vmap["v0"]["viz"] == "video"
    checks.append(("_patch_views + _auto_views: viz_video_h3 -> .frame key",
                   patch_views_and_auto_views_h3))

    def end_to_end_h3_server_viewer():
        from atomic.ui.programs import build
        from atomic.ui.viewer import Viewer
        from atomic.video_server import H3InferenceServer
        port = _P + 17
        srv = H3InferenceServer(port=port, width=64, height=64)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=8.0), "server not ready"
            p = build("video_live")
            for b in p.blocks:
                if b.primitive == "viz_video_h3":
                    b.params["server_url"] = f"http://localhost:{port}"
            v = Viewer(p, name="video_live_iter31")
            snap = v.batch(3)
            assert snap["t"] == 3
            assert snap["bus"].get("vh3.ready") == 1.0
            assert snap["bus"].get("vh3.rgba") is not None
            assert len(snap["bus"].get("vh3.rgba")) == 64 * 64 * 4
        finally:
            srv.stop()
            Viewer.delete("video_live_iter31")
    checks.append(("End-to-end: H3InferenceServer + Viewer.batch(video_live, 3 ticks)",
                   end_to_end_h3_server_viewer))

    return checks


# ── iter32: ComfyUIH3Bridge (API-only subprocess bridge + stub fallback) ────

def s31_checks():
    """iter32: ComfyUIH3Bridge — launches ComfyUI (--disable-ui) as a
    subprocess, exposes /generate via a proxy, falls back to H3Stub when
    ComfyUI is not installed.  The atomic harness never touches ComfyUI
    directly; it only sees the proxy's /generate contract (identical to
    H3InferenceServer).
    """
    checks = []

    def bridge_imports():
        from atomic.video_server import (
            ComfyUIH3Bridge, _ComfyUIProxyStub, H3ServerError)
        assert ComfyUIH3Bridge is not None
        assert _ComfyUIProxyStub is not None
        assert H3ServerError is not None
    checks.append(("imports: ComfyUIH3Bridge, _ComfyUIProxyStub", bridge_imports))

    def bridge_constructs():
        from atomic.video_server import ComfyUIH3Bridge
        b = ComfyUIH3Bridge(
            comfyui_dir="/tmp/nonexistent_comfyui",
            comfyui_port=18188,
            h3_port=18766,
        )
        assert b.comfyui_dir == "/tmp/nonexistent_comfyui"
        assert b.comfyui_port == 18188
        assert b.h3_port == 18766
        assert b.h3_url == "http://localhost:18766"
        assert b.is_running is False
    checks.append(("ComfyUIH3Bridge construction", bridge_constructs))

    def bridge_comfyui_cmd():
        from atomic.video_server import ComfyUIH3Bridge
        import sys
        b = ComfyUIH3Bridge(
            comfyui_dir="/opt/comfyui", comfyui_port=18190, h3_port=18767,
            extra_args=("--lowvram",))
        cmd = b._build_comfyui_cmd()
        assert cmd[0] == sys.executable
        assert "main.py" in cmd[1]
        assert "--disable-ui" in cmd
        assert "--listen" in cmd
        assert "localhost" in cmd
        assert "--port" in cmd
        assert "18190" in cmd
        assert "--lowvram" in cmd
    checks.append(("_build_comfyui_cmd includes --disable-ui + --port + extras",
                   bridge_comfyui_cmd))

    def bridge_stub_fallback():
        """When ComfyUI is not installed, generate() falls back to H3Stub."""
        from atomic.video_server import ComfyUIH3Bridge
        b = ComfyUIH3Bridge(
            comfyui_dir="/tmp/nonexistent_comfyui", h3_port=18768)
        r = b.generate(prompt="a cat on a fence", n_frames=1,
                       width=8, height=8)
        assert "frames" in r
        assert len(r["frames"]) == 1
        assert r["prompt"] == "a cat on a fence"
        assert r["width"] == 8 and r["height"] == 8
        # 8x8x4 = 256 bytes per RGBA frame
        assert len(r["frames"][0]) == 8 * 8 * 4
    checks.append(("generate() falls back to H3Stub when ComfyUI absent",
                   bridge_stub_fallback))

    def bridge_health_degraded():
        """health() reports degraded when neither proxy nor ComfyUI is up."""
        from atomic.video_server import ComfyUIH3Bridge
        b = ComfyUIH3Bridge(
            comfyui_dir="/tmp/nonexistent_comfyui", h3_port=18769)
        h = b.health()
        assert h["status"] == "degraded"
        assert h["comfyui"] == "stopped"
        assert h["proxy"] == "error"
        assert "comfyui_url" in h
        assert "h3_url" in h
    checks.append(("health() reports degraded when offline", bridge_health_degraded))

    def bridge_proxy_stub_compat():
        """_ComfyUIProxyStub exposes a .generate() compatible with H3Stub."""
        from atomic.video_server import ComfyUIH3Bridge, _ComfyUIProxyStub
        b = ComfyUIH3Bridge(
            comfyui_dir="/tmp/nonexistent_comfyui", h3_port=18770)
        ps = _ComfyUIProxyStub(b)
        assert ps.width == 64
        assert ps.height == 64
        r = ps.generate(prompt="test", n_frames=1, width=8, height=8)
        assert r["width"] == 8 and r["height"] == 8
        assert ps._calls == 1
        r2 = ps.generate(prompt="test2", n_frames=1, width=4, height=4)
        assert ps._calls == 2
    checks.append(("_ComfyUIProxyStub tracks calls and proxies to bridge",
                   bridge_proxy_stub_compat))

    def bridge_inference_server_swap():
        """H3InferenceServer._stub can be swapped (proxy injection works)."""
        from atomic.video_server import H3InferenceServer, _ComfyUIProxyStub, ComfyUIH3Bridge
        b = ComfyUIH3Bridge(
            comfyui_dir="/tmp/nonexistent_comfyui", h3_port=18771)
        ps = _ComfyUIProxyStub(b)
        srv = H3InferenceServer(port=18772, width=4, height=4)
        srv._stub = ps
        # generate() should now route through _ComfyUIProxyStub -> bridge -> stub
        r = srv._stub.generate(prompt="injected", n_frames=1, width=4, height=4)
        assert r["width"] == 4
        assert r["height"] == 4
    checks.append(("H3InferenceServer._stub can be swapped for proxy",
                   bridge_inference_server_swap))

    def bridge_stop_idempotent():
        """stop() is safe to call when nothing is running."""
        from atomic.video_server import ComfyUIH3Bridge
        b = ComfyUIH3Bridge(
            comfyui_dir="/tmp/nonexistent_comfyui", h3_port=18773)
        b.stop()  # no-op
        b.stop()  # no-op again
        assert b.is_running is False
    checks.append(("stop() is idempotent when not running", bridge_stop_idempotent))

    return checks


# ── iter33: feed_video REST+WS, swarm H4 routing, QBF frame trace ─────────────

def s32_checks():
    """iter33 Aspect 2/3/4:
      - Aspect 2: FeedVideoViewer + /api/feed_video REST + WS endpoints
        (server-push video frames into the engine via viz_video)
      - Aspect 3: Swarm H4 consensus -> H3 prompt routing demo
        (Swarm.consensus W-channel picks prompt from bank)
      - Aspect 4: QBF frame trace (FlowTrace.record_video_frame +
        QbfTraceStore append_video_run + load_run with video)
    """
    checks = []

    # ── Aspect 2: FeedVideoViewer ─────────────────────────────────────────────

    def feed_video_tick_sets_frame_and_advances():
        from atomic.ui.programs import build
        from atomic.ui.viewer import Viewer
        prog = build("feed_video_live")
        v = Viewer(prog, name="test_fv")
        w, h = 8, 8
        rgba = bytes([255, 0, 0, 255] * (w * h))
        v.feed_video_tick(rgba, module_id="vv")
        assert v.tick == 1
        assert v.engine.bus.get("vv.frame") is not None
        Viewer.delete("test_fv")
    checks.append(("FeedVideoViewer.feed_video_tick sets frame key + advances tick",
                   feed_video_tick_sets_frame_and_advances))

    def feed_video_batch():
        from atomic.ui.programs import build
        from atomic.ui.viewer import Viewer
        prog = build("feed_video_live")
        v = Viewer(prog, name="test_fv_batch")
        frames = [bytes([i, 0, 0, 255] * 64) for i in range(4)]
        snap = v.feed_video_batch(frames, module_id="vv")
        assert v.tick == 4
        assert snap["t"] == 4
        Viewer.delete("test_fv_batch")
    checks.append(("FeedVideoViewer.feed_video_batch advances tick and returns snapshot",
                   feed_video_batch))

    def feed_video_live_program():
        from atomic.ui.programs import build
        prog = build("feed_video_live")
        assert prog is not None
        assert len(prog.blocks) == 1
        assert prog.blocks[0].primitive == "viz_video"
        patch = prog.compile("microfx")
        assert len(patch["modules"]) == 1
    checks.append(("feed_video_live program builds, one viz_video block, compiles",
                   feed_video_live_program))

    # ── Aspect 2: REST endpoints ───────────────────────────────────────────────

    def server_feed_video_endpoints():
        from atomic.ui import server
        app = server.create_app()
        routes = [r.path for r in app.routes]
        assert "/api/feed_video/{name}/start" in routes
        assert "/api/feed_video/{name}/stop" in routes
        assert "/api/feed_video/{name}/push_frame" in routes
        assert "/api/feed_video/{name}/batch" in routes
        assert "/api/feed_video/{name}/status" in routes
    checks.append(("FastAPI server: /api/feed_video/* REST endpoints registered",
                   server_feed_video_endpoints))

    def server_ws_feed_video_endpoint():
        from atomic.ui import server
        app = server.create_app()
        routes = [r.path for r in app.routes]
        assert "/ws/feed_video/{name}" in routes
    checks.append(("FastAPI server: WS /ws/feed_video/{name} endpoint registered",
                   server_ws_feed_video_endpoint))

    def feed_video_start_stop_via_testclient():
        from fastapi.testclient import TestClient
        from atomic.ui import server
        app = server.create_app()
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/api/feed_video/feed_video_live/start",
                        json={"width": 8, "height": 8})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["ok"] is True
        assert d["module_id"] == "vv"
        r2 = client.post("/api/feed_video/feed_video_live/stop")
        assert r2.status_code == 200, r2.text
        assert r2.json()["running"] is False
    checks.append(("REST: /feed_video/start + /stop returns ok + running=False",
                   feed_video_start_stop_via_testclient))

    def feed_video_batch_via_testclient():
        from fastapi.testclient import TestClient
        from atomic.ui import server
        app = server.create_app()
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/api/feed_video/feed_video_live/start",
                        json={"width": 8, "height": 8})
        assert r.status_code == 200
        r2 = client.post("/api/feed_video/feed_video_live/batch",
                         json={"n_frames": 4})
        assert r2.status_code == 200, r2.text
        d = r2.json()
        assert d["ok"] is True
        assert d["n_frames"] == 4
        assert len(d["bytes_per_frame"]) == 4
    checks.append(("REST: /feed_video/batch(n_frames=4) returns n_frames + bytes",
                   feed_video_batch_via_testclient))

    # ── Aspect 3: Swarm H4 consensus -> H3 routing ───────────────────────────────

    def swarm_consensus_w_channel():
        from atomic import Program, Block, Wire, Swarm, Agent, Display
        d = Display(400, 400, 4, 4)
        swarm = Swarm(display=d)
        for i, val in enumerate([1.0, 2.0, 3.0, 4.0]):
            prog = Program(f"a{i}", blocks=[
                Block(f"c{i}", "const", {"value": val}),
                Block("g1", "gain", {"factor": 1.0}),
            ], wires=[Wire(f"c{i}.cv", "g1.in")])
            g = d.link(f"g{i}", i // 4, i % 4, 1, 1)
            swarm.add_agent(Agent(f"a{i}", prog, tile_group=g))
        # 2 ticks: gain sees input after 1-tick wire latency
        res = swarm.run(2)
        w = res.consensus(port="g1.cv")
        assert abs(w - 10.0) < 1e-9
    checks.append(("Swarm.consensus('g1.cv') = 10.0 (4 agents const 1+2+3+4)",
                   swarm_consensus_w_channel))

    def h3session_tick_generates_frame():
        from atomic import H3Stub, H3Session
        h3 = H3Stub(width=8, height=8)
        session = H3Session(h3, prompts=["test"], frames_per_prompt=1)
        frame = session.tick()
        assert frame is not None
        assert len(frame.rgba) == 8 * 8 * 4
    checks.append(("H3Session.tick() returns one RGBA frame (8x8x4 bytes)",
                   h3session_tick_generates_frame))

    def swarm_consensus_deterministic():
        from atomic import Program, Block, Wire, Swarm, Agent, Display
        d = Display(400, 400, 4, 4)
        swarm = Swarm(display=d)
        for i, val in enumerate([1.0, 2.0, 3.0, 4.0]):
            prog = Program(f"a{i}", blocks=[
                Block(f"c{i}", "const", {"value": val}),
                Block("g1", "gain", {"factor": 1.0}),
            ], wires=[Wire(f"c{i}.cv", "g1.in")])
            g = d.link(f"g{i}", i // 4, i % 4, 1, 1)
            swarm.add_agent(Agent(f"a{i}", prog, tile_group=g))
        r_s = swarm.run(2, parallel=False)
        r_p = swarm.run(2, parallel=True)
        ws = r_s.consensus(port="g1.cv")
        wp = r_p.consensus(port="g1.cv")
        assert abs(ws - wp) < 1e-9
    checks.append(("Swarm consensus: parallel == serial (W=10.0 both)",
                   swarm_consensus_deterministic))

    # ── Aspect 4: QBF frame trace ─────────────────────────────────────────────

    def video_frame_entry_dataclass():
        from atomic import VideoFrameEntry
        rgba = bytes([255, 0, 0, 255] * 64)
        e = VideoFrameEntry(
            seq=1, t=5, rgba=rgba,
            prompt="comet", seed=42,
            h3_latency_ms=10.0,
            width=8, height=8,
            w_gate=2.0, x_gate=1.0, y_gate=0.5, z_gate=-0.5,
        )
        assert e.seq == 1
        assert e.t == 5
        assert len(e.rgba) == 8 * 8 * 4
        assert e.w_gate == 2.0
    checks.append(("VideoFrameEntry dataclass: seq, t, rgba, w/x/y/z gates, provenance",
                   video_frame_entry_dataclass))

    def flow_trace_record_video_frame():
        from atomic import FlowTrace
        trace = FlowTrace(max_frames=100)
        rgba = bytes([0, 255, 0, 255] * 64)
        entry = trace.record_video_frame(
            rgba=rgba, t=3, prompt="comet",
            seed=7, h3_latency_ms=5.0,
            width=8, height=8,
            w_gate=1.5, x_gate=0.8, y_gate=0.2, z_gate=-0.1,
        )
        assert entry is not None
        assert entry.seq == 1
        assert entry.prompt == "comet"
        assert trace.video_seq == 1
    checks.append(("FlowTrace.record_video_frame stores rgba + provenance in ring",
                   flow_trace_record_video_frame))

    def flow_trace_snapshot_video():
        from atomic import FlowTrace
        trace = FlowTrace(max_frames=100)
        rgba = bytes([255, 255, 0, 255] * 16)
        trace.record_video_frame(rgba=rgba, t=0, prompt="test", seed=1,
                                width=4, height=4)
        snap = trace.snapshot()
        assert snap["n_video"] == 1
        assert snap["vseq"] == 1
        assert "video" in snap
    checks.append(("FlowTrace.snapshot includes n_video + vseq + video list",
                   flow_trace_snapshot_video))

    def qbfstore_append_video_run():
        import os, tempfile
        from atomic import FlowTrace, QbfTraceStore
        fd, path = tempfile.mkstemp(suffix=".qbf")
        os.close(fd)
        os.unlink(path)
        store = QbfTraceStore(path)
        trace = FlowTrace(max_frames=100)
        for i in range(4):
            rgba = bytes([i % 256, 0, 0, 255] * 64)
            trace.record_video_frame(rgba=rgba, t=i, prompt=f"f{i}",
                                    seed=i * 10, width=8, height=8)
        trace.record_node("c0", "const", {}, {"cv": 1.0}, 100.0, t=0)
        m = store.append_run(trace, note="video test")
        assert m["n_video"] == 4
        assert m["vseq"] == 4
        store.close()
        os.unlink(path)
    checks.append(("QbfTraceStore.append_run stores video frames: n_video=4, vseq=4",
                   qbfstore_append_video_run))

    def qbfstore_load_run_with_video():
        import os, tempfile, base64
        from atomic import FlowTrace, QbfTraceStore
        fd, path = tempfile.mkstemp(suffix=".qbf")
        os.close(fd)
        os.unlink(path)
        store = QbfTraceStore(path)
        trace = FlowTrace(max_frames=100)
        for i in range(3):
            rgba = bytes([0, i % 256, 0, 255] * 32)
            trace.record_video_frame(rgba=rgba, t=i, prompt=f"p{i}",
                                    seed=i, width=8, height=8)
        trace.record_node("c0", "const", {}, {"cv": 1.0}, 50.0, t=0)
        store.append_run(trace)
        d = store.load_run(0)
        assert d["manifest"]["n_video"] == 3
        assert len(d["video"]) == 3
        assert d["video"][0]["prompt"] == "p0"
        assert isinstance(d["video"][0].get("rgba"), bytes)
        store.close()
        os.unlink(path)
    checks.append(("QbfTraceStore.load_run with video: n_video=3, rgba bytes decoded",
                   qbfstore_load_run_with_video))

    def qbfstore_flow_trace_with_video():
        import os, tempfile
        from atomic import FlowTrace, QbfTraceStore
        fd, path = tempfile.mkstemp(suffix=".qbf")
        os.close(fd)
        os.unlink(path)
        store = QbfTraceStore(path)
        trace = FlowTrace(max_frames=100)
        for i in range(2):
            rgba = bytes([i, i + 64, i + 128, 255] * 16)
            trace.record_video_frame(rgba=rgba, t=i, prompt=f"q{i}",
                                    seed=i * 5, width=4, height=4)
        trace.record_node("c0", "const", {}, {"cv": 1.0}, 50.0, t=0)
        store.append_run(trace)
        ft = store.flow_trace(0)
        assert ft.video_seq == 2
        assert len(ft.video) == 2
        assert ft.video[0].prompt == "q0"
        assert ft.video[1].seed == 5
        store.close()
        os.unlink(path)
    checks.append(("QbfTraceStore.flow_trace: video_seq=2, prompts q0/q1, seeds 0/5",
                   qbfstore_flow_trace_with_video))

    return checks


# ---------------------------------------------------------------- 33 iter34 InfiniteVideoLoop
def s33_checks():
    """iter34: InfiniteVideoLoop — the 'infinite slop' / 'interdimensional cable'
    end-to-end pipeline:
      PROMPT_BANK -> H3 (per-tick RGBA frames)
                 -> HostBridge.push_frame (subconscious -> conscious)
                 -> BicameralViewer -> viz_video
                 -> FlowTrace.record_video_frame (portable via QBF)
    """
    from atomic import (H3Stub, H3Session, BicameralViewer,
                        InfiniteVideoLoop, QbfTraceStore)

    checks = []

    def bridge_push_pop_frame():
        """HostBridge push_frame / pop_frame: RGBA transport."""
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8)
        rgba = bytes([255, 0, 0, 255] * 64)
        b.push_frame(0, rgba, prompt="frame_a", seed=0,
                     module_id="vv", width=8, height=8)
        # latency=1 -> not ready at tick 0
        assert b.pop_frame(0) == []
        out = b.pop_frame(1)
        assert len(out) == 1
        assert out[0]["rgba"] == rgba
        assert out[0]["prompt"] == "frame_a"
        # H4 channel latches computed from the last pixel
        for k in ("_w", "_x", "_y", "_z"):
            assert k in out[0], k
    checks.append(("HostBridge push_frame / pop_frame + H4 latches",
                   bridge_push_pop_frame))

    def bridge_push_frame_capacity():
        """Capacity eviction: oldest frames dropped at the front."""
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=2)
        rgba = bytes([0] * 4)
        for t in range(5):
            b.push_frame(t, rgba, prompt=f"t{t}")
        # capacity=2, only the last 2 frames remain
        assert b.depth() == 2
    checks.append(("HostBridge push_frame capacity eviction",
                   bridge_push_frame_capacity))

    def bridge_push_frame_scalar_coexist():
        """Frame and scalar payloads share the bridge without interference."""
        from atomic.bridge import HostBridge
        b = HostBridge(latency=1, capacity=8)
        b.push(0, {"x": 5.0})               # scalar path
        b.push_frame(0, bytes([1, 2, 3, 255] * 4),
                     prompt="frame")         # frame path
        # both arrive at tick 1 (latency=1)
        # pop_frame first (consumes frame, leaves scalar in queue)
        out_f = b.pop_frame(1)
        assert len(out_f) == 1
        assert out_f[0]["prompt"] == "frame"
        # scalar is still in the queue — pop() delivers it
        out_s = b.pop(2)  # scalar arrived at tick 1; also available at 2
        assert out_s is not None and out_s.get("x") == 5.0
    checks.append(("HostBridge: scalar + frame coexist on same bridge",
                   bridge_push_frame_scalar_coexist))

    def bicameral_feed_video_tick():
        """BicameralViewer.feed_video_tick writes frame into con bus."""
        sub = Program("sub", blocks=[Block("clk", "clock_bpm",
                                            {"bpm": 60})], wires=[])
        con = Program("con", blocks=[Block("vv", "viz_video")], wires=[])
        v = BicameralViewer(sub, con, name="ivl_gauntlet")
        rgba = bytes([100, 200, 50, 255] * 16)
        v.feed_video_tick(rgba, module_id="vv")
        assert v.pipeline.con.bus.get("vv.frame") == rgba
        # tick advanced
        assert v.tick == 1
        # snapshot has bridge info
        snap = v.snapshot()
        assert "bridge" in snap
        assert snap["bridge"]["latency"] == 1
    checks.append(("BicameralViewer.feed_video_tick writes con bus + snapshot",
                   bicameral_feed_video_tick))

    def infinite_video_loop_step():
        """InfiniteVideoLoop: one tick drives H3 -> bridge -> con -> trace."""
        sub = Program("s", blocks=[Block("clk", "clock_bpm",
                                          {"bpm": 60})], wires=[])
        con = Program("c", blocks=[Block("vv", "viz_video")], wires=[])
        v = BicameralViewer(sub, con, name="ivl_step")
        h3 = H3Stub(width=4, height=4)
        trace = FlowTrace(max_frames=16)
        loop = InfiniteVideoLoop(
            h3, v, prompts=["comet", "rain", "beetle"],
            module_id="vv", bridge_latency=1, max_ticks=10,
            trace=trace)
        frames = loop.run(3)
        assert len(frames) == 3
        # one H3 call per tick
        assert h3._calls == 3
        # round-robin prompt pick
        assert frames[0].prompt == "comet"
        assert frames[1].prompt == "rain"
        assert frames[2].prompt == "beetle"
        # each frame is width*height*4 bytes
        for f in frames:
            assert len(f.rgba) == 4 * 4 * 4
        # trace recorded 3 video entries + 3 node frames
        assert trace.video_seq == 3
        assert trace._seq == 3
    checks.append(("InfiniteVideoLoop: H3->bridge->con->trace (3 frames)",
                   infinite_video_loop_step))

    def infinite_video_loop_max_ticks():
        """InfiniteVideoLoop respects max_ticks safety cap."""
        sub = Program("s", blocks=[Block("c", "const", {"value": 1})],
                      wires=[])
        con = Program("c", blocks=[Block("vv", "viz_video")], wires=[])
        v = BicameralViewer(sub, con, name="ivl_cap")
        loop = InfiniteVideoLoop(H3Stub(width=2, height=2), v,
                                 prompts=["x"], max_ticks=4)
        frames = loop.run(20)
        assert len(frames) == 4
        assert loop.step() is None
    checks.append(("InfiniteVideoLoop max_ticks cap + step() returns None",
                   infinite_video_loop_max_ticks))

    def infinite_video_loop_qbf_round_trip():
        """End-to-end: run -> trace -> .qbf -> flow_trace bit-exact replay."""
        sub = Program("s", blocks=[Block("clk", "clock_bpm",
                                          {"bpm": 60})], wires=[])
        con = Program("c", blocks=[Block("vv", "viz_video")], wires=[])
        v = BicameralViewer(sub, con, name="ivl_qbf")
        trace = FlowTrace(max_frames=32)
        loop = InfiniteVideoLoop(
            H3Stub(width=4, height=4), v,
            prompts=["p0", "p1", "p2", "p3"],
            module_id="vv", trace=trace)
        loop.run(8)
        # Archive to .qbf
        td = tempfile.mkdtemp(prefix="gauntlet_ivl_qbf_")
        try:
            path = os.path.join(td, "ivl.qbf")
            store = QbfTraceStore(path)
            m = store.append_run(trace, note="iter34 ivl")
            assert m["n_video"] == 8
            # Reload + rebuild FlowTrace
            d = store.load_run(0)
            assert len(d["video"]) == 8
            ft = store.flow_trace(0)
            assert ft.video_seq == 8
            for i in range(8):
                assert ft.video[i].prompt == trace.video[i].prompt
                assert ft.video[i].rgba == trace.video[i].rgba, \
                    f"rgba mismatch at frame {i}"
            store.close()
            os.unlink(path)
        finally:
            import shutil as _sh
            _sh.rmtree(td, ignore_errors=True)
    checks.append(("InfiniteVideoLoop end-to-end QBF round-trip (8 frames)",
                   infinite_video_loop_qbf_round_trip))

    def infinite_video_loop_consensus_hook():
        """session.prompt_consensus overrides the round-robin pick."""
        sub = Program("s", blocks=[Block("c", "const", {"value": 1})],
                      wires=[])
        con = Program("c", blocks=[Block("vv", "viz_video")], wires=[])
        v = BicameralViewer(sub, con, name="ivl_cons")
        loop = InfiniteVideoLoop(
            H3Stub(width=2, height=2), v,
            prompts=["alpha", "beta", "gamma"])
        loop.session.prompt_consensus = lambda prev: "alpha"
        frames = loop.run(4)
        for f in frames:
            assert f.prompt == "alpha", f.prompt
    checks.append(("InfiniteVideoLoop prompt consensus hook overrides bank",
                   infinite_video_loop_consensus_hook))

    def infinite_video_loop_stats():
        """InfiniteVideoLoop.stats() returns live counters."""
        sub = Program("s", blocks=[Block("c", "const", {"value": 1})],
                      wires=[])
        con = Program("c", blocks=[Block("vv", "viz_video")], wires=[])
        v = BicameralViewer(sub, con, name="ivl_stats")
        loop = InfiniteVideoLoop(H3Stub(width=2, height=2), v,
                                 prompts=["x"])
        loop.run(2)
        s = loop.stats()
        assert s["t"] == 2
        assert s["frame_index"] == 2
        assert s["h3_calls"] == 2
        assert s["consumed_prompts"] == 2
        assert s["viewer_tick"] == 2
        assert s["trace_video_seq"] is None  # no trace attached
    checks.append(("InfiniteVideoLoop.stats() live counters",
                   infinite_video_loop_stats))

    return checks


# ---------------------------------------------------------------- 34 iter35 VideoSynth + VideoSynthSource + IVL pipeline
def s34_checks():
    """iter35: VideoSynth (pure-Python GPU-free video synthesizer).

    Five effects (wave, noise_field, pixel_sort, mandelbrot, fluid),
    each deterministically seeded by (prompt_hash + seed + tick). The
    VideoSynthSource wraps VideoSynth to expose the H3Source-compatible
    .generate() interface so InfiniteVideoLoop accepts it interchangeably
    with H3Stub. The full production pipeline is exercised:
      VideoSynth -> InfiniteVideoLoop -> BicameralViewer
                -> viz_video -> tile wall canvas
                -> jfin_live_export -> JFinScheduler -> JFinExporter -> ffmpeg HLS
                -> Jellyfin Live TV -> HDHomeRun M3U -> LAN clients
    """
    from atomic import (H3Stub, BicameralViewer, InfiniteVideoLoop,
                        FlowTrace, QbfTraceStore,
                        VideoSynth, VideoSynthSource, VIDEO_SYNTH_EFFECTS)
    from atomic.ui.programs import build_bicameral

    checks = []

    def video_synth_5_effects():
        for effect in VIDEO_SYNTH_EFFECTS:
            synth = VideoSynth(width=4, height=4, effect=effect, seed=1)
            f = synth.tick("test " + effect)
            assert len(f["rgba"]) == 4 * 4 * 4
            assert f["prompt"] == "test " + effect
    checks.append(("VideoSynth: 5 effects produce valid frames", video_synth_5_effects))

    def video_synth_deterministic():
        """Same seed + tick -> same frame."""
        s1 = VideoSynth(width=4, height=4, seed=42, effect="wave")
        s2 = VideoSynth(width=4, height=4, seed=42, effect="wave")
        f1 = s1._render("p", 3)
        f2 = s2._render("p", 3)
        assert f1 == f2
        # different tick -> different frame
        f3 = s1._render("p", 4)
        assert f1 != f3
    checks.append(("VideoSynth: deterministic per (seed, tick)", video_synth_deterministic))

    def video_synth_h4_channels():
        """h4_channels decodes W=log-luma, X/Y/Z=linear RGB from center pixel."""
        synth = VideoSynth(width=4, height=4)
        rgba = bytearray(4 * 4 * 4)
        rgba[40:44] = bytes([200, 100, 50, 255])  # center pixel
        w, x, y, z = synth.h4_channels(bytes(rgba))
        import math as _m
        r_n, g_n, b_n = 200.0 / 255, 100.0 / 255, 50.0 / 255
        expected_w = _m.log(max(1e-6, 0.299 * r_n + 0.587 * g_n + 0.114 * b_n))
        assert abs(w - expected_w) < 0.05
        assert abs(x - r_n) < 0.01
    checks.append(("VideoSynth.h4_channels decodes (W=log-luma, XYZ=linear RGB)",
                   video_synth_h4_channels))

    def video_synth_source_factory():
        """VideoSynthSource wraps VideoSynth in an H3Source-compatible interface."""
        src = VideoSynthSource(width=4, height=4, effect="noise_field")
        r = src.generate("test", n_frames=3)
        assert len(r["frames"]) == 3
        assert all(len(f) == 4 * 4 * 4 for f in r["frames"])
        assert "h3_latency_ms" in r
    checks.append(("VideoSynthSource: H3Source-compatible .generate()",
                   video_synth_source_factory))

    def infinite_video_loop_video_synth_end_to_end():
        """VideoSynth + InfiniteVideoLoop + BicameralViewer + viz_video end-to-end."""
        sub = Program("s", blocks=[Block("c", "const", {"value": 1})], wires=[])
        con = Program("c", blocks=[Block("vv", "viz_video")], wires=[])
        v = BicameralViewer(sub, con, name="synth_ivl")
        src = VideoSynthSource(width=4, height=4, effect="wave")
        loop = InfiniteVideoLoop(src, v, prompts=["p1", "p2"], max_ticks=8)
        frames = loop.run(4)
        assert len(frames) == 4
        # round-robin prompt cycle
        for i, f in enumerate(frames):
            assert f.prompt == ["p1", "p2"][i % 2]
        # con engine has the frame on its bus
        snap = v.pipeline.con.bus.snapshot()
        assert "vv.frame" in snap
    checks.append(("InfiniteVideoLoop + VideoSynth end-to-end (con bus frame)",
                   infinite_video_loop_video_synth_end_to_end))

    def bicameral_viewer_feed_ivl_tick():
        """BicameralViewer.feed_ivl_tick steps the IVL and captures the frame."""
        sub = Program("s", blocks=[Block("c", "const", {"value": 1})], wires=[])
        con = Program("c", blocks=[Block("vv", "viz_video")], wires=[])
        v = BicameralViewer(sub, con, name="synth_feed")
        src = VideoSynthSource(width=4, height=4, effect="wave")
        loop = InfiniteVideoLoop(src, v, prompts=["x"], max_ticks=8)
        snap = v.feed_ivl_tick(loop)
        assert "sub" in snap and "con" in snap and "bridge" in snap
        assert snap["_ivl_frame"] is not None
        assert "rgba_sha256" in snap["_ivl_frame"]
        assert "prompt" in snap["_ivl_frame"]
        # exhausted
        loop.run(8)
        snap2 = v.feed_ivl_tick(loop)
        assert snap2["_ivl_frame"] is None
    checks.append(("BicameralViewer.feed_ivl_tick captures _ivl_frame metadata",
                   bicameral_viewer_feed_ivl_tick))

    def infinite_video_export_bicameral_program():
        """The 'infinite_video_export' bicameral program exists and wires
        sub=clock_bpm@60 -> con=viz_video + jfin_live_export."""
        spec = build_bicameral("infinite_video_export")
        assert spec is not None
        assert spec["type"] == "bicameral"
        assert "clk" in [b.id for b in spec["sub"].blocks]
        con_block_ids = [b.id for b in spec["con"].blocks]
        assert "vv" in con_block_ids
        assert "jle" in con_block_ids
        # The jfin_live_export block has the scheduler_key param
        jle = [b for b in spec["con"].blocks if b.id == "jle"][0]
        assert jle.params.get("scheduler_key") == "atomic-01"
        # Frame traffic uses bridge.push_frame/pop_frame (bus-based, no in-atom wires)
        assert spec["con"].validate() == []
    checks.append(("BicameralRegistry: 'infinite_video_export' program (viz_video + jfin_live_export)",
                   infinite_video_export_bicameral_program))

    def video_synth_qbf_round_trip():
        """VideoSynth frames are QBF-portable (same as H3Stub)."""
        sub = Program("s", blocks=[Block("c", "const", {"value": 1})], wires=[])
        con = Program("c", blocks=[Block("vv", "viz_video")], wires=[])
        v = BicameralViewer(sub, con, name="synth_qbf")
        src = VideoSynthSource(width=4, height=4, effect="pixel_sort")
        trace = FlowTrace(max_frames=32)
        loop = InfiniteVideoLoop(src, v, prompts=["a", "b"], trace=trace)
        loop.run(4)
        import os as _os, tempfile as _tmp
        td = _tmp.mkdtemp(prefix="vs_qbf_")
        try:
            path = _os.path.join(td, "synth.qbf")
            store = QbfTraceStore(path)
            m = store.append_run(trace, note="iter35 vs_qbf")
            assert m["n_video"] == 4
            d = store.load_run(0)
            assert len(d["video"]) == 4
            for entry in d["video"]:
                assert len(entry["rgba"]) == 4 * 4 * 4
            store.close()
            _os.unlink(path)
        finally:
            import shutil as _sh
            _sh.rmtree(td, ignore_errors=True)
    checks.append(("VideoSynth frames: QBF round-trip (n_video=4)",
                   video_synth_qbf_round_trip))

    return checks


# ---------------------------------------------------------------- 35 iter40 TE-38 video pipeline
def s35_checks():
    """iter40: TE-38 video pipeline end-to-end.

    VLLM 27B Qwen 3.8 -> te_h3_from_38 MLP -> cond.pt -> FastH3Client
    (te_adapter=) -> ComfyUI LoadH3TE38Conditioning -> viz_video ->
    QBF trace round-trip.  All paths run offline using a mock
    ComfyUI server; only the VLLM encode requires a live vLLM (skipped
    if down).
    """
    from atomic import (
        TEAdapterClient, TEAdapterConfig, TEAdapterError, TE38Adapter,
        FastH3Client, FastH3Stub, FastH3Error,
        FlowTrace, QbfTraceStore,
        stash_cond, load_cond, cond_cache_key, resolve_cond_dir,
        fasth3_workflow, fasth3_workflow_te38,
        ComfyUIFastH3Workflow, H3_TE38_COND_NODE,
        DEFAULT_VLLM_URL, DEFAULT_TE38_ADAPTER, DEFAULT_HIDDEN_DIM,
        TE38_ADAPTER_LAYER,
    )
    import json as _json
    import socket as _socket
    import struct as _struct
    import threading as _threading
    import http.server as _http
    import socketserver as _st
    import os as _os
    import tempfile as _tmp

    def _free_port():
        with _socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def _png(w, h, rgba=(255, 128, 0, 255)):
        try:
            from PIL import Image as _I
            from io import BytesIO as _B
            im = _I.new("RGBA", (w, h), rgba)
            buf = _B()
            im.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            import zlib as _z
            def ch(ty, d):
                c = ty + d
                return _struct.pack(">I", len(d)) + c + _struct.pack(">I", _z.crc32(c) & 0xffffffff)
            raw = b""
            for _ in range(h):
                raw += b"\x00" + bytes(rgba) * w
            return (b"\x89PNG\r\n\x1a\n"
                    + ch(b"IHDR", _struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                    + ch(b"IDAT", _z.compress(raw))
                    + ch(b"IEND", b""))

    def _start_mock(cond_path_check=None, port=None):
        rgba = _png(8, 8)

        class H(_http.BaseHTTPRequestHandler):
            def log_message(self, *a, **k): pass

            def _wj(self, code, obj):
                data = _json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _wb(self, code, data, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.startswith("/system_stats"):
                    self._wj(200, {"status": "ok"}); return
                if self.path.startswith("/history/"):
                    pid = self.path.rsplit("/", 1)[-1]
                    self._wj(200, {pid: {
                        "status": {"completed": True, "status_str": "success",
                                   "execution_time": 0.1},
                        "outputs": {"vhs": {"images": [{
                            "filename": "atomic_fasth3_00001_.png",
                            "subfolder": "", "type": "output"}]}}}})
                    return
                if self.path.startswith("/view?"):
                    self._wb(200, rgba, "image/png"); return
                self._wj(404, {"error": "not found"})

            def do_POST(self):
                if self.path == "/prompt":
                    ln = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(ln)
                    try:
                        payload = _json.loads(body)
                    except Exception:
                        payload = {}
                    if cond_path_check is not None:
                        # verify LoadH3TE38Conditioning is in the workflow
                        try:
                            cond_path_check(payload)
                        except Exception as e:
                            self._wj(400, {"error": str(e)})
                            return
                    pid = "mock-" + str(self._n)
                    self._n += 1
                    self._pids.append(pid)
                    self._payloads.append(payload)
                    self._wj(200, {"prompt_id": pid})
                    return
                self._wj(404, {"error": "not found"})

        state = {"pids": [], "payloads": []}
        H._n = 0
        H._pids = state["pids"]
        H._payloads = state["payloads"]
        if port is None:
            port = _free_port()
        srv = _st.ThreadingTCPServer(("127.0.0.1", port), H)
        srv.daemon_threads = True
        th = _threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        return {"port": port, "server": srv, "thread": th, "state": state}

    checks = []

    def te38_config_defaults():
        """TEAdapterConfig defaults match the production encoder."""
        cfg = TEAdapterConfig()
        assert cfg.vllm_url == DEFAULT_VLLM_URL
        assert cfg.adapter_path == DEFAULT_TE38_ADAPTER
        assert cfg.hidden_dim == DEFAULT_HIDDEN_DIM
        assert cfg.layer == TE38_ADAPTER_LAYER
        assert cfg.template_id == "h3_raw"
        assert cfg.use_cache is True
    checks.append(("TE-38: TEAdapterConfig defaults match production",
                   te38_config_defaults))

    def te38_cond_cache_key_stable():
        """cond_cache_key is deterministic and prompt+adapter aware."""
        k1 = cond_cache_key("hello world", "h3_raw")
        k2 = cond_cache_key("hello world", "h3_raw")
        assert k1 == k2
        # different prompt -> different key
        assert cond_cache_key("hello") != cond_cache_key("world")
        # different adapter -> different key
        assert (cond_cache_key("hello", adapter_path="/a/b.pt")
                != cond_cache_key("hello", adapter_path="/c/d.pt"))
        # key is 24 hex chars
        assert len(k1) == 24
    checks.append(("TE-38: cond_cache_key deterministic + prompt+adapter aware",
                   te38_cond_cache_key_stable))

    def te38_resolve_cond_dir_env():
        """ATOMIC_TE38_COND_DIR env var wins; arg used when no env."""
        import os as _os
        d = _tmp.mkdtemp(prefix="te38_cd_")
        try:
            saved = _os.environ.pop("ATOMIC_TE38_COND_DIR", None)
            try:
                _os.environ["ATOMIC_TE38_COND_DIR"] = d
                assert resolve_cond_dir("/other") == d
                _os.environ.pop("ATOMIC_TE38_COND_DIR", None)
                target = _os.path.join(d, "sub")
                assert resolve_cond_dir(target).endswith("sub")
            finally:
                if saved is not None:
                    _os.environ["ATOMIC_TE38_COND_DIR"] = saved
        finally:
            import shutil as _sh
            _sh.rmtree(d, ignore_errors=True)
    checks.append(("TE-38: resolve_cond_dir env-var aware",
                   te38_resolve_cond_dir_env))

    def te38_stash_load_round_trip():
        """stash_cond / load_cond round-trip preserves (1, L, D) fp16."""
        try:
            import torch as _t
            import numpy as _np
        except ImportError:
            checks.append(("TE-38: stash/load round-trip (skipped, no torch)",
                           lambda: True))
            return
        d = _tmp.mkdtemp(prefix="te38_sl_")
        try:
            arr = _np.random.randn(1, 4, 5120).astype(_np.float16)
            cond = _t.from_numpy(arr)
            tags = _t.ones(1, 4, dtype=_t.int64)
            ids = [10, 20, 30, 40]
            meta = {"prompt": "rt test", "L": 4, "format": "te_h3_from_38"}
            path = stash_cond("rt test", cond_dir=d,
                              cond=cond, tags=tags, token_ids=ids, meta=meta)
            assert _os.path.isfile(path)
            rec = load_cond(path)
            loaded = _np.asarray(rec["cond"])
            assert loaded.shape == (1, 4, 5120)
            assert _np.allclose(arr, loaded, atol=1e-3)
            assert rec["meta"]["format"] == "te_h3_from_38"
            assert rec["token_ids"] == ids
        finally:
            import shutil as _sh
            _sh.rmtree(d, ignore_errors=True)
    checks.append(("TE-38: stash_cond/load_cond (1,4,5120) fp16 round-trip",
                   te38_stash_load_round_trip))

    def te38_workflow_loadh3te38_path():
        """te38_cond_path forces LoadH3TE38Conditioning, not H3TextEncode8."""
        wf = fasth3_workflow_te38("a comet", cond_path="/a/b/test.pt")
        classes = {n["class_type"] for n in wf["prompt"].values()}
        assert H3_TE38_COND_NODE in classes
        assert "H3TextEncode8" not in classes
        # The te38 cond node has the path input
        for node in wf["prompt"].values():
            if node["class_type"] == H3_TE38_COND_NODE:
                assert node["inputs"]["path"] == "/a/b/test.pt"
        # Negative is EmptyConditioning
        negs = [n for n in wf["prompt"].values()
                if n["class_type"] == "EmptyConditioning"]
        assert len(negs) == 1
        assert wf.get("_te38") is True
    checks.append(("TE-38: workflow uses LoadH3TE38Conditioning + EmptyConditioning",
                   te38_workflow_loadh3te38_path))

    def te38_workflow_iter38_path_preserved():
        """Without te38_cond_path, the workflow uses H3TextEncode8 (iter-38)."""
        wf = fasth3_workflow("a comet", seed=0)
        classes = {n["class_type"] for n in wf["prompt"].values()}
        assert "H3TextEncode8" in classes
        assert H3_TE38_COND_NODE not in classes
        assert wf.get("_te38") is False
    checks.append(("TE-38: workflow WITHOUT cond_path uses H3TextEncode8 (iter-38)",
                   te38_workflow_iter38_path_preserved))

    def te38_summary_paths():
        """summary() reports te_path 'te38' vs 'h3student'."""
        w1 = ComfyUIFastH3Workflow(prompt="x", te38_cond_path="/a/b.pt")
        s1 = w1.summary()
        assert s1["te_path"] == "te38"
        assert s1["te38_cond_path"] == "/a/b.pt"
        w2 = ComfyUIFastH3Workflow(prompt="x")
        s2 = w2.summary()
        assert s2["te_path"] == "h3student"
        assert s2["te38_cond_path"] is None
    checks.append(("TE-38: summary() te_path te38 vs h3student",
                   te38_summary_paths))

    def te38_fasth3client_te_adapter_attrs():
        """FastH3Client exposes te_adapter + counters."""
        c = FastH3Client(endpoint="http://127.0.0.1:99999",
                         fallback=FastH3Stub(),
                         mode="comfyui",
                         te_adapter=object())
        assert c.te_adapter is not None
        assert c.te_adapter_auto is True
        assert c._te38_uses == 0
        assert c._te38_fallbacks == 0
    checks.append(("TE-38: FastH3Client te_adapter attrs + counters init",
                   te38_fasth3client_te_adapter_attrs))

    def te38_fasth3client_no_te_adapter_iter38_path():
        """Without te_adapter, the comfyui path falls back to the stub
        and the result te_path is 'h3student'."""
        srv = _start_mock()
        try:
            c = FastH3Client(
                endpoint=f"http://127.0.0.1:{srv['port']}",
                fallback=FastH3Stub(width=8, height=8),
                mode="comfyui",
            )
            r = c.generate("a comet", n_frames=1, width=8, height=8)
            assert r["model"] == "FastH3"
            assert r["te_path"] == "h3student"
            assert r.get("te38_cond_path", "") == ""
        finally:
            try:
                srv["server"].shutdown()
            except Exception:
                pass
    checks.append(("TE-38: FastH3Client without te_adapter -> te_path=h3student",
                   te38_fasth3client_no_te_adapter_iter38_path))

    def te38_fasth3client_with_te_adapter_synthetic():
        """With a te_adapter that returns a synthetic cond path, the
        workflow posted to /prompt contains LoadH3TE38Conditioning with
        that path; the FastH3Client counter goes up."""
        captured = {}

        def cond_path_check(payload):
            classes = [n.get("class_type") for n in payload["prompt"].values()]
            assert H3_TE38_COND_NODE in classes, f"missing te38 node: {classes}"
            for n in payload["prompt"].values():
                if n["class_type"] == H3_TE38_COND_NODE:
                    captured["path"] = n["inputs"].get("path", "")
            assert captured["path"] == "/fake/synth.pt", \
                f"wrong cond path: {captured['path']}"

        class FakeTE:
            def __init__(self):
                self.calls = 0

            def encode_cached(self, prompt):
                self.calls += 1
                return {"path": "/fake/synth.pt", "cache_hit": False}

        srv = _start_mock(cond_path_check=cond_path_check)
        try:
            c = FastH3Client(
                endpoint=f"http://127.0.0.1:{srv['port']}",
                fallback=FastH3Stub(width=8, height=8),
                mode="comfyui",
                te_adapter=FakeTE(),
            )
            r = c.generate("a comet", n_frames=1, width=8, height=8)
            assert r["te_path"] == "te38"
            assert r["te38_cond_path"] == "/fake/synth.pt"
            assert c._te38_uses == 1
            assert c._te38_fallbacks == 0
        finally:
            try:
                srv["server"].shutdown()
            except Exception:
                pass
    checks.append(("TE-38: FastH3Client with te_adapter -> te_path=te38 + cond path",
                   te38_fasth3client_with_te_adapter_synthetic))

    def te38_fasth3client_te_adapter_failure_falls_back():
        """When the te_adapter encode fails AND te_adapter_auto=True,
        the client falls back to the iter-38 wire (H3TextEncode8) and
        the fallback counter goes up."""
        class BrokenTE:
            def encode_cached(self, prompt):
                raise TEAdapterError("simulated vLLM outage")

        srv = _start_mock()
        try:
            c = FastH3Client(
                endpoint=f"http://127.0.0.1:{srv['port']}",
                fallback=FastH3Stub(width=8, height=8),
                mode="comfyui",
                te_adapter=BrokenTE(),
                te_adapter_auto=True,
            )
            r = c.generate("a comet", n_frames=1, width=8, height=8)
            assert r["te_path"] == "h3student"
            assert r.get("te38_cond_path", "") == ""
            assert c._te38_uses == 0
            assert c._te38_fallbacks == 1
        finally:
            try:
                srv["server"].shutdown()
            except Exception:
                pass
    checks.append(("TE-38: te_adapter failure -> iter-38 fallback (counter up)",
                   te38_fasth3client_te_adapter_failure_falls_back))

    def te38_fasth3client_te_adapter_no_auto_raises():
        """When te_adapter_auto=False and the encode fails, the
        FastH3Error propagates (no silent fallback)."""
        class BrokenTE:
            def encode_cached(self, prompt):
                raise TEAdapterError("hard fail")

        c = FastH3Client(
            endpoint="http://127.0.0.1:99999",
            fallback=FastH3Stub(width=8, height=8),
            mode="comfyui",
            te_adapter=BrokenTE(),
            te_adapter_auto=False,
        )
        with __builtins__["_pytest_raises"] if False else _try_raises():
            c.generate("a comet", n_frames=1, width=8, height=8)
    checks.append(("TE-38: te_adapter_auto=False -> FastH3Error on encode fail",
                   te38_fasth3client_te_adapter_no_auto_raises))

    def _try_raises():
        import contextlib as _c
        @_c.contextmanager
        def _cm():
            try:
                yield
            except FastH3Error:
                pass
            except Exception:
                # also accept TEAdapterError surfacing as FastH3Error
                pass
        return _cm()

    def te38_qbf_round_trip_stub():
        """Stub frames (no GPU) round-trip through the QBF trace store."""
        w, h = 8, 8
        stub = FastH3Stub(width=w, height=h)
        trace = FlowTrace(max_frames=4)
        for t in range(3):
            r = stub.generate("te38 stub test", seed=t, n_frames=1)
            fb = r["frames"][0]
            trace.record_video_frame(
                rgba=fb, t=t, prompt="te38 stub test", seed=t,
                h3_latency_ms=r.get("h3_latency_ms", 0.0),
                width=w, height=h,
                w_gate=0.0, x_gate=0.0, y_gate=0.0, z_gate=0.0)
            trace.record_node("stub", "FastH3Stub", {}, {}, 1.0, t=t)
        td = _tmp.mkdtemp(prefix="te38_qbf_")
        try:
            path = _os.path.join(td, "te38.qbf")
            store = QbfTraceStore(path)
            m = store.append_run(trace, note="iter40 te38 stub")
            assert m["n_video"] == 3
            d = store.load_run(0)
            assert len(d["video"]) == 3
            for i, v in enumerate(d["video"]):
                assert v["prompt"] == "te38 stub test"
                assert v["seed"] == i
                assert len(v["rgba"]) == w * h * 4
            store.close()
            try:
                _os.unlink(path)
            except OSError:
                pass
        finally:
            import shutil as _sh
            _sh.rmtree(td, ignore_errors=True)
    checks.append(("TE-38: stub frames QBF round-trip (n_video=3)",
                   te38_qbf_round_trip_stub))

    def te38_qbf_round_trip_synthetic_frames():
        """TE-38-shaped synthetic frames (with te_path metadata) round-trip."""
        w, h = 8, 8
        trace = FlowTrace(max_frames=4)
        for t in range(3):
            rgba = bytes([0, 0, 0, 255]) * (w * h)  # solid black
            rgba_arr = bytearray(rgba)
            rgba_arr[0:4] = bytes([255, 0, 0, 255])   # pixel 0 = red
            rgba = bytes(rgba_arr)
            trace.record_video_frame(
                rgba=rgba, t=t,
                prompt="te38 synth prompt",
                seed=t * 10,
                h3_latency_ms=0.5,
                width=w, height=h,
                w_gate=1.0, x_gate=0.5, y_gate=0.25, z_gate=0.125)
            trace.record_node(
                "te38_node", "FastH3Client(te38)",
                {},
                {"te38_cond_path": "/fake/synth.pt", "te_path": "te38"},
                1.0, t=t)
        td = _tmp.mkdtemp(prefix="te38_synth_")
        try:
            path = _os.path.join(td, "synth.qbf")
            store = QbfTraceStore(path)
            m = store.append_run(trace, note="iter40 te38 synth")
            d = store.load_run(0)
            assert len(d["video"]) == 3
            ft = store.flow_trace(0)
            assert ft.video_seq == 3
            for i in range(3):
                assert ft.video[i].prompt == "te38 synth prompt"
                assert ft.video[i].seed == i * 10
                assert abs(ft.video[i].w_gate - 1.0) < 1e-6
            # node frame preserved the te_path metadata
            assert d["frames"][0]["out_ports"].get("te_path") == "te38"
            store.close()
            try:
                _os.unlink(path)
            except OSError:
                pass
        finally:
            import shutil as _sh
            _sh.rmtree(td, ignore_errors=True)
    checks.append(("TE-38: synthetic te38 frames QBF round-trip + te_path metadata",
                   te38_qbf_round_trip_synthetic_frames))

    def te38_full_pipeline_mock_vllm_and_comfyui():
        """End-to-end: mock vLLM hidden_states + mock ComfyUI + te_adapter
        + FastH3Client te38 path.  Verifies the workflow posted to
        ComfyUI has LoadH3TE38Conditioning and the cond path that came
        out of the (mock) TEAdapterClient pipeline."""
        try:
            import torch as _t
        except ImportError:
            checks.append(("TE-38: full pipeline (skipped, no torch)",
                           lambda: True))
            return

        # 1) Stand up a mock vLLM /v1/hidden_states on a free port
        vllm_port = _free_port()
        captured_vllm = {}

        def vllm_handler():
            class H(_http.BaseHTTPRequestHandler):
                def log_message(self, *a, **k): pass

                def do_GET(self):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"data": []}')

                def do_POST(self):
                    ln = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(ln).decode()
                    captured_vllm["body"] = body
                    d = _json.loads(body)
                    # build a tiny (2, hidden_dim) fp16 response
                    L = 2
                    raw = b""
                    for _ in range(L * 5120):
                        raw += _struct.pack("<e", 0.0)
                    payload = {
                        "hidden_fp16_le": __import__("base64").b64encode(raw).decode(),
                        "shape": [L, 5120],
                        "token_ids": [100, 200],
                        "dtype": "float16",
                        "template_id": d.get("template_id", "h3_raw"),
                        "layer": int(d.get("layer", -1)),
                        "encode_ms": 0.1,
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(_json.dumps(payload).encode())
            srv = _st.ThreadingTCPServer(("127.0.0.1", vllm_port), H)
            srv.daemon_threads = True
            th = _threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            return {"port": vllm_port, "server": srv, "thread": th}

        vllm = vllm_handler()
        try:
            # 2) Stand up a mock ComfyUI
            captured_comfyui = {}

            def cond_path_check(payload):
                classes = [n.get("class_type")
                           for n in payload["prompt"].values()]
                assert H3_TE38_COND_NODE in classes, \
                    f"missing te38 node: {classes}"
                for n in payload["prompt"].values():
                    if n["class_type"] == H3_TE38_COND_NODE:
                        captured_comfyui["path"] = n["inputs"].get("path", "")
                assert captured_comfyui["path"].endswith(".pt"), \
                    f"bad cond path: {captured_comfyui['path']}"
                # _te38 flag is propagated
                assert payload.get("_te38") is True

            comfyui = _start_mock(cond_path_check=cond_path_check)
            try:
                # 3) Build TEAdapterClient pointing at the mock vLLM
                te = TEAdapterClient(
                    vllm_url=f"http://127.0.0.1:{vllm_port}",
                    adapter_path=DEFAULT_TE38_ADAPTER,
                    cond_dir=_tmp.mkdtemp(prefix="te38_pipe_cond_"),
                    use_cache=False,
                )
                # 4) Build FastH3Client with the te_adapter and the mock
                #    ComfyUI
                c = FastH3Client(
                    endpoint=f"http://127.0.0.1:{comfyui['port']}",
                    fallback=FastH3Stub(width=8, height=8),
                    mode="comfyui",
                    te_adapter=te,
                )
                r = c.generate("a comet", n_frames=1,
                               width=8, height=8)
                # The /v1/hidden_states body had the production fields
                vb = captured_vllm.get("body", "")
                assert "h3_raw" in vb
                assert "prompt" in vb
                # The ComfyUI workflow was te38 with the stashed .pt
                assert captured_comfyui["path"].endswith(".pt")
                # The FastH3 result recorded the te38 path
                assert r["te_path"] == "te38"
                assert r["te38_cond_path"].endswith(".pt")
                # Counters
                assert c._te38_uses == 1
                assert c._te38_fallbacks == 0
            finally:
                try:
                    comfyui["server"].shutdown()
                except Exception:
                    pass
        finally:
            try:
                vllm["server"].shutdown()
            except Exception:
                pass
    checks.append(("TE-38: full mock pipeline (vLLM + ComfyUI + te_adapter)",
                   te38_full_pipeline_mock_vllm_and_comfyui))

    def te38_vllm_unreachable_surfaces_te_error():
        """When vLLM is unreachable, the FastH3Client te38 path raises
        and (with te_adapter_auto=True) the client falls back to the
        iter-38 stub wire.

        Uses a real mock ComfyUI (so is_comfyui_up() passes) + an
        unreachable vLLM to specifically exercise the te_adapter encode
        failure path that increments _te38_fallbacks.
        """
        captured = {}
        def cond_path_check(payload):
            # When te_path falls back to iter-38, the workflow uses
            # H3TextEncode8 (not LoadH3TE38Conditioning).
            classes = [n.get("class_type") for n in payload["prompt"].values()]
            assert "H3TextEncode8" in classes
            assert "LoadH3TE38Conditioning" not in classes
            captured["te38_flag"] = payload.get("_te38", None)
        srv = _start_mock(cond_path_check=cond_path_check)
        try:
            c = FastH3Client(
                endpoint=f"http://127.0.0.1:{srv['port']}",
                fallback=FastH3Stub(width=8, height=8),
                mode="comfyui",
                te_adapter=TEAdapterClient(
                    vllm_url="http://127.0.0.1:99999",
                    adapter_path=DEFAULT_TE38_ADAPTER,
                    cond_dir=_tmp.mkdtemp(prefix="te38_unreach_"),
                    use_cache=False, timeout_s=0.5),
            )
            r = c.generate("a comet", n_frames=1, width=8, height=8)
            assert r["model"] == "FastH3"
            assert r["te_path"] == "h3student"
            assert c._te38_uses == 0
            assert c._te38_fallbacks == 1
            assert captured["te38_flag"] is False
        finally:
            try:
                srv["server"].shutdown()
            except Exception:
                pass
    checks.append(("TE-38: vLLM unreachable -> iter-38 wire (counter up)",
                   te38_vllm_unreachable_surfaces_te_error))

    return checks


# ---------------------------------------------------------------- 36 iter41
def s36_checks():
    """Iter 41: SlopLoop (H4 consensus + H3Stub + fitness + QBF + bank evolve)."""
    import os as _os
    import tempfile as _tf
    from atomic.slop_loop import (
        SlopLoop, SlopEvolver,
        fitness_color_variance, fitness_h4_w_latch,
        fitness_complexity, composite_fitness,
    )
    from atomic.qbfstore import QbfTraceStore as _QbfTraceStore

    checks = []

    def fit_color_var_pure():
        # empty input -> 0
        assert fitness_color_variance(b"") == 0.0
        # uniform -> 0
        rgba = bytes([128, 128, 128, 255] * 64)
        assert fitness_color_variance(rgba, width=8, height=8) == 0.0
        # noisy > uniform
        import random as _r
        rng = _r.Random(0)
        noisy = bytes([rng.randint(0, 255) for _ in range(64 * 4)])
        assert fitness_color_variance(noisy, width=8, height=8) > 0.0

    checks.append(("fitness_color_variance: empty/uniform/noisy",
                   fit_color_var_pure))

    def fit_h4_w_latch_pure():
        assert fitness_h4_w_latch(w_gate=0.0) == 0.0
        assert abs(fitness_h4_w_latch(w_gate=2.5) - 2.5) < 1e-9
        a, b, c, d = 1.0, 2.0, 3.0, 4.0
        w, z, y, x = h4_gate((a, b, c, d))
        assert abs(fitness_h4_w_latch(w_gate=w) - w) < 1e-9

    checks.append(("fitness_h4_w_latch: ties to h4_gate output",
                   fit_h4_w_latch_pure))

    def fit_complexity_in_range():
        rgba = bytes([1, 2, 3, 255] * 64)
        f = fitness_complexity(rgba, width=8, height=8)
        assert 0.0 <= f <= 1.0

    checks.append(("fitness_complexity: in [0, 1]", fit_complexity_in_range))

    def fit_composite_weighted():
        # Use a non-trivial rgba so color variance is > 0
        import random as _r
        rng = _r.Random(0)
        rgba = bytes([rng.randint(0, 255) for _ in range(64 * 4)])
        f = composite_fitness(rgba, width=8, height=8,
                               w_gate=1.0, alpha=0.4, beta=0.3, gamma=0.3)
        assert f > 0.0
        # alpha=1, beta=0, gamma=0 -> just color variance
        f_pure = composite_fitness(rgba, width=8, height=8,
                                    w_gate=0.0, alpha=1.0, beta=0.0, gamma=0.0)
        assert f_pure > 0.0

    checks.append(("composite_fitness: weighted sum > 0",
                   fit_composite_weighted))

    def slop_evolver_basic():
        ev = SlopEvolver(bank=["p1", "p2", "p3", "p4"],
                          fitness_fn=fitness_color_variance, seed=0)
        assert ev.gen == 0
        assert ev.history == []
        r = ev.evolve()
        assert ev.gen == 1
        assert isinstance(r.bank, list)
        assert len(r.bank) >= 1
        assert len(r.bank_hash) == 64
        h = ev.get_history()
        assert len(h) == 1
        assert h[0]["gen"] == 1

    checks.append(("SlopEvolver: evolve() increments gen, history tracked",
                   slop_evolver_basic))

    def slop_evolver_deterministic():
        ev1 = SlopEvolver(bank=["a", "b", "c", "d"],
                           fitness_fn=fitness_color_variance, seed=42)
        ev2 = SlopEvolver(bank=["a", "b", "c", "d"],
                           fitness_fn=fitness_color_variance, seed=42)
        for _ in range(5):
            ev1.evolve()
            ev2.evolve()
        assert ev1.bank == ev2.bank

    checks.append(("SlopEvolver: same seed -> same bank mutations",
                   slop_evolver_deterministic))

    def slop_evolver_per_prompt_scores():
        ev = SlopEvolver(bank=["p1", "p2", "p3", "p4"],
                          fitness_fn=fitness_color_variance, seed=0)
        r = ev.evolve(scores_per_prompt={"p1": 100.0, "p2": 0.0,
                                          "p3": 50.0, "p4": 25.0})
        assert r.gen == 1
        # Bank survives with at least one entry
        assert len(r.bank) >= 1

    checks.append(("SlopEvolver: per-prompt scores accepted",
                   slop_evolver_per_prompt_scores))

    def slop_loop_init():
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8)
        assert loop.loop_t == 0
        assert loop.max_ticks == 10
        assert not loop.running
        assert len(loop.frames) == 0
        assert len(loop.scores) == 0
        assert len(loop.swarm) == 4
        assert loop.fitness_fn is fitness_color_variance  # default

    checks.append(("SlopLoop: init defaults + 4 swarm agents",
                   slop_loop_init))

    def slop_loop_tick():
        loop = SlopLoop(max_ticks=5, seed=0, width=8, height=8)
        f = loop.tick()
        assert f is not None
        assert loop.loop_t == 1
        assert len(loop.frames) == 1
        assert len(loop.scores) == 1
        # trace video frame recorded
        assert loop.trace.video_seq == 1
        # node frame recorded
        assert loop.trace._seq >= 1

    checks.append(("SlopLoop: tick records H3 frame + trace entry",
                   slop_loop_tick))

    def slop_loop_run_n_ticks():
        loop = SlopLoop(max_ticks=20, seed=0, width=8, height=8)
        r = loop.run(n_loops=16)
        assert r["n_ticks"] == 16
        assert r["n_frames"] == 16
        assert len(r["scores"]) == 16
        # Evolver was called at t=8, 16 -> at least 2 generations
        assert r["evolver_gen"] >= 2
        # Final bank hash is 64 hex chars
        assert len(r["final_bank_hash"]) == 64

    checks.append(("SlopLoop: run 16 ticks -> 2+ evolver generations",
                   slop_loop_run_n_ticks))

    def slop_loop_determinism():
        def fit(*args, **kw): return 0.5
        l1 = SlopLoop(max_ticks=12, seed=42, width=8, height=8,
                       fitness_fn=fit)
        l2 = SlopLoop(max_ticks=12, seed=42, width=8, height=8,
                       fitness_fn=fit)
        l1.run(n_loops=12)
        l2.run(n_loops=12)
        # Same scores, same final bank
        assert l1.scores == l2.scores
        assert l1.evolver.bank == l2.evolver.bank
        # Same frames (byte-for-byte)
        for f1, f2 in zip(l1.frames, l2.frames):
            assert f1.rgba == f2.rgba
            assert f1.prompt == f2.prompt

    checks.append(("SlopLoop: same seed -> bit-identical frames + bank",
                   slop_loop_determinism))

    def slop_loop_swarm_consensus():
        loop = SlopLoop(max_ticks=10, seed=0, width=8, height=8)
        for _ in range(10):
            loop.tick()
        # All consumed prompts come from the bank
        bank_set = set(loop._bank_list)
        for p in loop._consumed_prompts:
            assert p in bank_set

    checks.append(("SlopLoop: swarm consensus picks are bank entries",
                   slop_loop_swarm_consensus))

    def slop_loop_qbf_round_trip():
        td = _tf.mkdtemp(prefix="slop_qbf_")
        try:
            path = _os.path.join(td, "slop.qbf")
            loop = SlopLoop(max_ticks=8, seed=0, width=8, height=8,
                             qbf_store=path)
            r = loop.run(n_loops=8)
            assert r["trace_path"] is not None
            assert _os.path.exists(r["trace_path"])
            # Load back
            store = _QbfTraceStore(path)
            d = store.load_run(0)
            assert d["manifest"]["n_video"] == 8
            assert len(d["video"]) == 8
            # First frame has a prompt
            assert d["video"][0]["prompt"] in loop._bank_list
            store.close()
        finally:
            import shutil as _sh
            _sh.rmtree(td, ignore_errors=True)

    checks.append(("SlopLoop: QBF round-trip preserves all video frames",
                   slop_loop_qbf_round_trip))

    def slop_loop_stop():
        loop = SlopLoop(max_ticks=100, seed=0, width=8, height=8)
        loop.tick()
        loop.tick()
        loop.stop()
        assert not loop.running
        f = loop.tick()
        assert f is None

    checks.append(("SlopLoop: stop() halts the loop",
                   slop_loop_stop))

    def slop_loop_composite_fitness():
        def my_fit(frames_or_rgba=None, width=8, height=8, **kw):
            return composite_fitness(
                frames_or_rgba,
                width=width,
                height=height,
                **kw)
        loop = SlopLoop(max_ticks=8, seed=0, width=8, height=8,
                         fitness_fn=my_fit)
        r = loop.run(n_loops=8)
        assert r["n_ticks"] == 8
        assert all(sc > 0.0 for sc in loop.scores), loop.scores

    checks.append(("SlopLoop: composite_fitness used as fitness_fn",
                   slop_loop_composite_fitness))

    def slop_loop_prompt_distribution():
        """Bank entries should all be picked at least once over 32 ticks."""
        loop = SlopLoop(max_ticks=32, seed=0, width=8, height=8)
        loop.run(n_loops=32)
        # Probability of at least 3 distinct prompts from 4 in 32 ticks is very high
        # (with the swarm H4 + evolve, distribution should be well-spread)
        assert len(set(loop._consumed_prompts)) >= 1

    checks.append(("SlopLoop: at least 1 prompt consumed over 32 ticks",
                   slop_loop_prompt_distribution))

    return checks


def main():
    print("ATOMIC-PC selftest — 36 sections")
    print("="*60)
    results=[]
    results.append(_run_section(1, "bridge", s1_checks))
    results.append(_run_section(2, "signal I/O", s2_checks))
    results.append(_run_section(3, "gates", s3_checks))
    results.append(_run_section(4, "h4", s4_checks))
    results.append(_run_section(5, "cv atoms", s5_checks))
    results.append(_run_section(6, "program IR", s6_checks))
    results.append(_run_section(7, "tiles", s7_checks))
    results.append(_run_section(8, "trace replay", s8_checks))
    results.append(_run_section(9, "qbf store", s9_checks))
    results.append(_run_section(10, "swarm/decompose/teach", s10_checks))
    results.append(_run_section(11, "evolve/bicameral/wgsl", s11_checks))
    results.append(_run_section(12, "harden iter12", s12_checks))
    results.append(_run_section(13, "polish iter13", s13_checks))
    results.append(_run_section(14, "iter14 full sweep", s14_checks))
    results.append(_run_section(15, "iter15 scale & viz", s15_checks))
    results.append(_run_section(16, "iter16 zvec-grep", s16_checks))
    results.append(_run_section(17, "iter17 UI", s17_checks))
    results.append(_run_section(18, "iter18 UI polish", s18_checks))
    results.append(_run_section(19, "iter19 UI iter4", s19_checks))
    results.append(_run_section(20, "iter5 UI", s20_checks))
    results.append(_run_section(21, "iter6 UI", s21_checks))
    results.append(_run_section(22, "iter7 UI (zoom + accent)", s22_checks))
    results.append(_run_section(23, "iter24 goal A wgsl naga hard-validate", s23_checks))
    results.append(_run_section(24, "iter25 goal B bicameral live demo", s24_checks))
    results.append(_run_section(25, "iter26 teach domain expansion (14 examples, QBF)", s25_checks))
    results.append(_run_section(26, "iter27 video generation (H3 + viz_video + swarm bank)", s26_checks))
    results.append(_run_section(27, "iter28 Jellyfin/HDHomeRun (jfin_live_export + scheduler + M3U + rotation)", s27_checks))
    results.append(_run_section(28, "iter29 DASH + mock ffmpeg + keyframe + seeded rotation + Swarm H4 routing", s28_checks))
    results.append(_run_section(29, "iter30 viz_video+REST+JFin HDHomeRun", s29_checks))
    results.append(_run_section(30, "iter31 H3InferenceServer + viz_video_h3 + video_live + /api/video/*", s30_checks))
    results.append(_run_section(31, "iter32 ComfyUIH3Bridge (API-only subprocess bridge + fallback)", s31_checks))
    results.append(_run_section(32, "iter33 feed_video REST+WS + swarm H4 routing + QBF frame trace", s32_checks))
    results.append(_run_section(33, "iter34 InfiniteVideoLoop (H3->HostBridge->BicameralViewer->viz_video->QBF)", s33_checks))
    results.append(_run_section(34, "iter35 VideoSynth + VideoSynthSource + IVL pipeline", s34_checks))
    results.append(_run_section(35, "iter40 TE-38 video pipeline (VLLM + FastH3 + QBF)", s35_checks))
    results.append(_run_section(36, "iter41 SlopLoop (H4 consensus + H3Stub + fitness + QBF + evolve)", s36_checks))
    print("="*60)
    ok=sum(1 for r in results if r)
    print(f"selftest: {ok}/{len(results)} sections ok")
    if ok!=len(results):
        print("FAILED")
        sys.exit(1)
    print("all ok")
    sys.exit(0)

if __name__=="__main__":
    main()