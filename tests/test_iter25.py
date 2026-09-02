"""iter25 — Goal B: bicameral live demo.

Two engines wired through HostBridge:
  sub = clock_bpm@60   (the subconscious / GPU1)
  con = accum -> smooth -> viz_series  (the conscious / GPU0)

BicameralPipeline.tick():
  sub.tick() -> sample sub.bus -> bridge.push -> bridge.pop
  -> inject into con node.inputs -> con.tick()

Tests:
  1. Final accum == 2.0 after 90 ticks at 60 bpm.
  2. Bridge depth == 1 (latency=1, one beat in flight).
  3. BicameralViewer snapshot has sub/con/bridge keys.
  4. UI server exposes /api/bicameral endpoints.
  5. /api/programs includes bicameral key.
"""
from __future__ import annotations

from atomic import Block, Program, Wire
from atomic.bridge import BicameralPipeline


def _sub_clock():
    return Program("clock_source",
                   blocks=[Block("clk", "clock_bpm", {"bpm": 60})],
                   wires=[])


def _con_counter():
    return Program("counter_display",
                   blocks=[Block("cnt", "accum", {"per_tick": 1}),
                           Block("sm", "smooth", {"alpha": 0.1}),
                           Block("v0", "viz_series")],
                   wires=[Wire("cnt.acc", "sm.in"),
                          Wire("sm.cv", "v0.in")])


def test_bicameral_accum_2():
    pipe = BicameralPipeline(_sub_clock(), _con_counter(),
                             bridge_map=[("clk.trig", "cnt.in")],
                             bridge_latency=1)
    pipe.run(90)
    assert pipe.con.bus.snapshot().get("cnt.acc") == 2.0


def test_bicameral_bridge_depth():
    pipe = BicameralPipeline(_sub_clock(), _con_counter(),
                             bridge_map=[("clk.trig", "cnt.in")],
                             bridge_latency=1)
    for _ in range(90):
        pipe.tick()
    assert pipe.bridge.depth() == 1
    assert pipe.bridge._pushed >= 1


def test_bicameral_viewer_snapshot():
    from atomic.ui.bicameral_viewer import BicameralViewer
    from atomic.ui.programs import build_bicameral
    spec = build_bicameral("bicameral_clock")
    assert spec is not None
    bv = BicameralViewer(spec["sub"], spec["con"],
                         bridge_map=spec["bridge_map"],
                         bridge_latency=spec["bridge_latency"],
                         name="test")
    snap = bv.batch(5)
    assert "sub" in snap and "con" in snap
    assert "bridge" in snap
    assert "depth" in snap["bridge"]
    assert "latency" in snap["bridge"]


def test_ui_bicameral_endpoints():
    from fastapi.testclient import TestClient
    from atomic.ui.server import create_app
    app = create_app()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/bicameral")
    assert r.status_code == 200
    assert "bicameral_clock" in r.json()["programs"]
    r2 = c.post("/api/bicameral/bicameral_clock/batch", json={"ticks": 10})
    assert r2.status_code == 200
    bd = r2.json()["bridge"]
    assert bd["pushed"] == 10
    assert bd["latency"] == 1


def test_ui_programs_includes_bicameral():
    from fastapi.testclient import TestClient
    from atomic.ui.server import create_app
    app = create_app()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/programs")
    assert "bicameral" in r.json()
    assert "bicameral_clock" in r.json()["bicameral"]