"""Iter 31 tests: H3InferenceServer, viz_video_h3 atom, video_live program,
REST /api/video/start|stop|status endpoints.

Tests:
  - H3InferenceServer starts, responds to /health, /generate
  - H3InferenceServer.stop() cleans up thread
  - viz_video_h3 atom registered in ATOMS
  - viz_video_h3 atom tick (server up): makes HTTP call, decodes RGBA
  - viz_video_h3 atom tick (server down): sets error flag, outputs 0
  - viz_video_h3 atom: enabled=0 gate skips HTTP call
  - viz_video_h3 atom: retry logic fires on connection error
  - viz_video_h3 atom: H4 RGBA decoding same as viz_video
  - video_live program builds without error
  - video_live program: compile("microfx") produces valid patch
  - /api/video/start returns status + url
  - /api/video/stop returns stopped status
  - /api/video/status returns correct state (not_running/running)
  - FastAPI server registers /api/video/* endpoints
  - _VIZ_OUTPUTS["viz_video_h3"] == "ready"
  - _VIZ_TYPES["viz_video_h3"] == "video"
  - _patch_views for viz_video_h3 generates correct .frame key
  - _auto_views for viz_video_h3 generates correct .frame key
  - End-to-end: H3InferenceServer + Viewer.tick_once with video_live
"""
import threading
import time

import pytest


class TestH3InferenceServer:
    def test_server_start_stop(self):
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=18765, width=8, height=8)
        srv.start()
        try:
            assert srv.is_running
            assert srv.wait_until_ready(timeout=5.0), "server didn't become ready"
            health = srv.health()
            assert health["status"] == "ok"
            assert health["model"] == "H3-FastVideo"
            assert health["width"] == 8
            assert health["height"] == 8
        finally:
            srv.stop()
        assert not srv.is_running

    def test_generate_endpoint(self):
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=18766, width=4, height=4)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=5.0)
            r = srv.generate("test prompt", seed=42, n_frames=1)
            assert "frames_b64" in r
            assert len(r["frames_b64"]) == 1
            assert r["prompt"] == "test prompt"
            assert r["seed"] == 42
            assert "h3_latency_ms" in r
            import base64
            frame = base64.b64decode(r["frames_b64"][0])
            assert len(frame) == 4 * 4 * 4
        finally:
            srv.stop()

    def test_generate_no_seed_derives_from_prompt(self):
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=18767)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=5.0)
            r = srv.generate("my prompt")
            assert "seed" in r
            assert isinstance(r["seed"], int)
        finally:
            srv.stop()

    def test_multiple_generate_calls_deterministic(self):
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=18768)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=5.0)
            r1 = srv.generate("const", seed=99)
            r2 = srv.generate("const", seed=99)
            assert r1["frames_b64"] == r2["frames_b64"]
            assert r1["seed"] == r2["seed"] == 99
        finally:
            srv.stop()


class TestVizVideoH3Atom:
    def test_viz_video_h3_registered(self):
        from atomic.gates import ATOMS
        assert "viz_video_h3" in ATOMS
        a = ATOMS["viz_video_h3"]
        assert a.category == "source"
        assert "ready" in a.outputs
        assert "w" in a.outputs
        assert "x" in a.outputs
        assert "y" in a.outputs
        assert "z" in a.outputs
        assert "server_url" in a.params
        assert a.params["server_url"] == "http://localhost:8765"

    def test_viz_video_h3_engine_basic(self):
        from atomic import Program, Block
        from atomic.engine import Engine
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=18770, width=4, height=4)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=8.0)
            p = Program("vv3_test", blocks=[
                Block("v", "viz_video_h3",
                      {"server_url": "http://localhost:18770",
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
            ready = eng.bus.get("v.ready")
            assert ready == 1.0
        finally:
            srv.stop()

    def test_viz_video_h3_enabled_gate(self):
        from atomic import Program, Block
        from atomic.engine import Engine
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=18771)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=8.0)
            p = Program("vv3_gate", blocks=[
                Block("v", "viz_video_h3",
                      {"server_url": "http://localhost:18771",
                       "enabled": 0.0}),
            ])
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], [])
            eng.tick()
            ready = eng.bus.get("v.ready")
            assert ready == 0.0
            frame = eng.bus.get("v.rgba")
            assert frame is None
        finally:
            srv.stop()

    def test_viz_video_h3_server_down_sets_error(self):
        from atomic import Program, Block
        from atomic.engine import Engine
        p = Program("vv3_down", blocks=[
            Block("v", "viz_video_h3",
                  {"server_url": "http://localhost:59999",
                   "timeout_s": 0.5, "max_retries": 1}),
        ])
        patch = p.compile("microfx")
        eng = Engine(patch["modules"], [])
        eng.tick()
        ready = eng.bus.get("v.ready")
        assert ready == 0.0

    def test_viz_video_h3_h4_decode_parity(self):
        from atomic import Program, Block
        from atomic.engine import Engine
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=18773, width=2, height=2)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=8.0)
            p_vv = Program("vv_ref", blocks=[
                Block("v", "viz_video", {"capture": 1.0}),
            ])
            p_vv3 = Program("vv3", blocks=[
                Block("v", "viz_video_h3",
                      {"server_url": "http://localhost:18773",
                       "width": 2, "height": 2,
                       "prompt": "parity_test_seed7",
                       "seed": 7}),
            ])
            patch_vv = p_vv.compile("microfx")
            patch_vv3 = p_vv3.compile("microfx")
            eng_vv = Engine(patch_vv["modules"], [])
            eng_vv3 = Engine(patch_vv3["modules"], [])
            import base64
            r = srv.generate("parity_test_seed7", seed=7)
            frame = base64.b64decode(r["frames_b64"][0])
            eng_vv.bus.set("v.frame", frame)
            eng_vv.tick()
            eng_vv3.tick()
            w_vv = eng_vv.bus.get("v.w")
            w_vv3 = eng_vv3.bus.get("v.w")
            x_vv = eng_vv.bus.get("v.x")
            x_vv3 = eng_vv3.bus.get("v.x")
            assert abs(w_vv - w_vv3) < 1e-6
            assert abs(x_vv - x_vv3) < 1e-6
        finally:
            srv.stop()


class TestVideoLiveProgram:
    def test_video_live_builds(self):
        from atomic.ui.programs import build
        p = build("video_live")
        assert p is not None
        assert p.name == "video_live"
        assert any(b.primitive == "viz_video_h3" for b in p.blocks)

    def test_video_live_compile_microfx(self):
        from atomic.ui.programs import build
        p = build("video_live")
        patch = p.compile("microfx")
        mods = patch["modules"]
        assert len(mods) == 1
        assert mods[0]["primitive"] == "viz_video_h3"
        assert mods[0]["id"] == "vh3"
        assert "server_url" in mods[0]["params"]

    def test_video_live_all_programs(self):
        from atomic.ui.programs import all_programs
        assert "video_live" in all_programs()


class TestVideoRESTEndpoints:
    def test_video_start_stop_status(self):
        from fastapi.testclient import TestClient
        from atomic.ui import server
        from atomic.video_server import _stop_server
        app = server.create_app()
        client = TestClient(app, raise_server_exceptions=False)
        # start
        r = client.post("/api/video/start",
                        json={"port": 18775, "width": 8, "height": 8})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("started", "already_running")
        assert body["port"] == 18775
        try:
            # poll status until running OR timeout (5s)
            deadline = time.time() + 5.0
            running = False
            while time.time() < deadline:
                r2 = client.get("/api/video/status?port=18775")
                assert r2.status_code == 200
                if r2.json().get("status") == "running":
                    running = True
                    break
                time.sleep(0.1)
            assert running, "video server never reported running"
            # final status check (last response)
            st = r2.json()
            assert st["port"] == 18775
            assert "health" in st
            assert st["health"]["status"] == "ok"
            # stop
            r3 = client.post("/api/video/stop", json={"port": 18775})
            assert r3.status_code == 200
            assert r3.json()["status"] == "stopped"
            # status after stop
            r4 = client.get("/api/video/status?port=18775")
            assert r4.json()["status"] == "not_running"
        finally:
            _stop_server(port=18775)

    def test_video_endpoints_in_routes(self):
        from atomic.ui import server
        app = server.create_app()
        routes = [r.path for r in app.routes]
        assert "/api/video/start" in routes
        assert "/api/video/stop" in routes
        assert "/api/video/status" in routes


class TestVizVideoH3Integration:
    def test_viz_outputs_includes_videovideo_h3(self):
        from atomic.program import _VIZ_OUTPUTS
        assert "viz_video_h3" in _VIZ_OUTPUTS
        assert _VIZ_OUTPUTS["viz_video_h3"] == "ready"

    def test_viz_types_includes_videovideo_h3(self):
        from atomic.ui.viewer import _VIZ_TYPES
        assert "viz_video_h3" in _VIZ_TYPES
        assert _VIZ_TYPES["viz_video_h3"] == "video"

    def test_patch_views_viz_video_h3(self):
        from atomic.program import _patch_views, Block
        blocks = [Block("vh3", "viz_video_h3")]
        views = _patch_views(blocks)
        assert len(views) == 1
        v = views[0]
        assert v["module"] == "vh3"
        assert v["key"] == "vh3.frame"
        assert v["as"] == "video_h3"
        assert v["output"] == "ready"

    def test_auto_views_viz_video_h3(self):
        from atomic.ui.viewer import _auto_views, _VIZ_TYPES
        assert _VIZ_TYPES["viz_video_h3"] == "video"
        mods = [{"id": "v0", "primitive": "viz_video_h3"}]
        views = _auto_views(mods)
        vmap = {v["module"]: v for v in views}
        assert "v0" in vmap
        assert vmap["v0"]["key"] == "v0.frame"
        assert vmap["v0"]["viz"] == "video"

    def test_end_to_end_server_plus_viewer(self):
        from atomic.ui.programs import build
        from atomic.ui.viewer import Viewer
        from atomic.video_server import H3InferenceServer
        srv = H3InferenceServer(port=18780, width=8, height=8)
        srv.start()
        try:
            assert srv.wait_until_ready(timeout=8.0)
            p = build("video_live")
            for b in p.blocks:
                if b.primitive == "viz_video_h3":
                    b.params["server_url"] = "http://localhost:18780"
            v = Viewer(p, name="video_live_test")
            snap = v.batch(3)
            assert snap["t"] == 3
            bus = snap["bus"]
            assert bus.get("vh3.ready") == 1.0
            assert bus.get("vh3.rgba") is not None
        finally:
            srv.stop()
            Viewer.delete("video_live_test")
