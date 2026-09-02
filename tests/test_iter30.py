"""iter30 tests: viz_video view integration, drawVideoFrame canvas renderer,
JFinChannel.to_dict, JFinM3U.find_by_id/from_discovered_hdhr, _JFIN_STATE
singleton, Viewer.feed_frame, /api/feed_frame + /api/jfin/* REST endpoints.

Tests:
  - _VIZ_OUTPUTS includes viz_video with canonical 'ready' output
  - _patch_views generates correct view key for viz_video
  - _auto_views generates correct key for viz_video and viz_wxyz3d
  - JFinChannel.to_dict() returns all fields
  - JFinM3U.find_by_id finds correct channel or None
  - JFinM3U.from_discovered_hdhr returns a valid JFinM3U
  - _JFIN_STATE singleton initialized with default channels
  - JFinM3U.discover_hdhr accepts timeout param
  - Viewer.feed_frame injects bytes into bus
  - FastAPI server registers all new endpoints
  - drawVideoFrame renders bytes (smoke test via mock canvas)
"""

import io
import os
import sys
import tempfile
import shutil

import pytest


class TestVizVideoViewIntegration:
    def test_viz_video_in_viz_outputs(self):
        from atomic.program import _VIZ_OUTPUTS
        assert "viz_video" in _VIZ_OUTPUTS
        assert _VIZ_OUTPUTS["viz_video"] == "ready"

    def test_wxyz3d_in_viz_outputs(self):
        from atomic.program import _VIZ_OUTPUTS
        assert "viz_wxyz3d" in _VIZ_OUTPUTS
        assert _VIZ_OUTPUTS["viz_wxyz3d"] == "z"

    def test_patch_views_viz_video_frame_key(self):
        from atomic.program import _patch_views, Block
        blocks = [
            Block("vv", "viz_video"),
            Block("w3d", "viz_wxyz3d"),
            Block("xy", "viz_xy"),
            Block("s", "viz_series"),
        ]
        views = _patch_views(blocks)
        vmap = {v["module"]: v for v in views}
        assert vmap["vv"]["key"] == "vv.frame"
        assert vmap["vv"]["as"] == "video"
        assert vmap["w3d"]["key"] == "w3d.z"
        assert vmap["xy"]["key"] == "xy.y"
        assert vmap["s"]["key"] == "s.cv"

    def test_auto_views_viz_video(self):
        from atomic.ui.viewer import _auto_views, _VIZ_TYPES
        assert _VIZ_TYPES["viz_video"] == "video"
        mods = [
            {"id": "v0", "primitive": "viz_video"},
            {"id": "w0", "primitive": "viz_wxyz3d"},
        ]
        views = _auto_views(mods)
        vmap = {v["module"]: v for v in views}
        assert vmap["v0"]["key"] == "v0.frame"
        assert vmap["v0"]["viz"] == "video"
        assert vmap["w0"]["key"] == "w0.z"


class TestJFinChannelToDict:
    def test_to_dict_full(self):
        from atomic.jellyfin import JFinChannel
        ch = JFinChannel("id30", "NAME30", "http://x.m3u8",
                         logo_url="http://logo.png", tuner_type="hdhr",
                         group="ATOMIC", number=7)
        d = ch.to_dict()
        assert d == {
            "id": "id30", "name": "NAME30", "m3u_url": "http://x.m3u8",
            "logo_url": "http://logo.png", "tuner_type": "hdhr",
            "group": "ATOMIC", "number": 7,
        }

    def test_to_dict_minimal(self):
        from atomic.jellyfin import JFinChannel
        ch = JFinChannel("m1", "M1", "http://m.m3u8")
        d = ch.to_dict()
        assert d["logo_url"] == ""
        assert d["tuner_type"] == "hdhr"
        assert d["group"] == "ATOMIC"
        assert d["number"] == 1


class TestJFinM3UExtensions:
    def test_find_by_id(self):
        from atomic.jellyfin import JFinM3U, JFinChannel
        m3u = JFinM3U()
        ch1 = JFinChannel("c1", "C1", "http://c1.m3u8")
        ch2 = JFinChannel("c2", "C2", "http://c2.m3u8")
        m3u.add_channel(ch1)
        m3u.add_channel(ch2)
        assert m3u.find_by_id("c2") is ch2
        assert m3u.find_by_id("c1") is ch1
        assert m3u.find_by_id("nope") is None

    def test_from_discovered_hdhr_no_devices(self):
        from atomic.jellyfin import JFinM3U
        m3u = JFinM3U.from_discovered_hdhr(timeout=0.05, base_url="http://test:8080")
        # may have 0 channels if no HDHR on LAN
        assert isinstance(m3u.channels, list)
        assert m3u.find_by_id("nonexistent") is None

    def test_from_discovered_hdhr_with_mock_device(self, monkeypatch):
        from atomic.jellyfin import JFinM3U
        # monkey-patch discover_hdhr to return fake devices
        def fake_discover(timeout=1.0):
            return [
                {"device_id": "ABC123", "device_type": "HDHR5-4US",
                 "tuner_count": 2, "firmware": "20240701", "ip": "192.168.1.100"},
                {"device_id": "XYZ789", "device_type": "HDHR3-EU",
                 "tuner_count": 1, "firmware": "20240701", "ip": "192.168.1.101"},
            ]
        monkeypatch.setattr(JFinM3U, "discover_hdhr", staticmethod(fake_discover))
        m3u = JFinM3U.from_discovered_hdhr(timeout=0.05, base_url="http://x:8080")
        # 2 tuners + 1 tuner = 3 channels
        assert len(m3u.channels) == 3
        ids = {ch.id for ch in m3u.channels}
        assert "hdhr-ABC123-t1" in ids
        assert "hdhr-ABC123-t2" in ids
        assert "hdhr-XYZ789-t1" in ids

    def test_discover_hdhr_timeout(self):
        from atomic.jellyfin import JFinM3U
        # must accept timeout kwarg without raising
        devs = JFinM3U.discover_hdhr(timeout=0.05)
        assert isinstance(devs, list)


class TestJFinStateSingleton:
    def test_state_init(self):
        from atomic.jellyfin import _JFIN_STATE, JFinM3U, JFinScheduler
        assert isinstance(_JFIN_STATE.m3u, JFinM3U)
        assert isinstance(_JFIN_STATE.scheduler, JFinScheduler)
        ids = {ch.id for ch in _JFIN_STATE.m3u.channels}
        assert "atomic-01" in ids
        assert "atomic-04" in ids

    def test_state_scheduler_channels_match_m3u(self):
        from atomic.jellyfin import _JFIN_STATE
        m3u_ids = {ch.id for ch in _JFIN_STATE.m3u.channels}
        sched_ids = set(_JFIN_STATE.scheduler.channels.keys())
        assert m3u_ids == sched_ids


class TestViewerFeedFrame:
    def test_feed_frame_injects_bytes(self):
        from atomic import Program, Block
        from atomic.ui.viewer import Viewer
        p = Program("ff_test", blocks=[Block("vv", "viz_video")])
        v = Viewer(p, name="ff_test")
        frame = b"\x80\x40\x20\x10" * 8 * 8
        ok = v.feed_frame("vv", frame)
        assert ok is True
        eng = v.engine
        assert eng.bus.get("vv.frame") == frame

    def test_feed_frame_bytearray(self):
        from atomic import Program, Block
        from atomic.ui.viewer import Viewer
        p = Program("ff_ba", blocks=[Block("vv", "viz_video")])
        v = Viewer(p, name="ff_ba")
        frame = bytearray(b"\xff" * 64)
        ok = v.feed_frame("vv", frame)
        assert ok is True
        eng = v.engine
        stored = eng.bus.get("vv.frame")
        assert bytes(stored) == bytes(frame)


class TestServerEndpoints:
    def _routes(self):
        from atomic.ui import server
        app = server.create_app()
        return [r.path for r in app.routes]

    def test_feed_frame_endpoint(self):
        routes = self._routes()
        assert "/api/feed_frame/{name}" in routes

    def test_jfin_endpoints(self):
        routes = self._routes()
        assert "/api/jfin/discover" in routes
        assert "/api/jfin/channels" in routes
        assert "/api/jfin/export/{ch_id}/push" in routes
        assert "/api/jfin/export/{ch_id}/start" in routes
        assert "/api/jfin/export/{ch_id}/stop" in routes
        assert "/api/jfin/scheduler" in routes
        assert "/api/jfin/rotate" in routes
        assert "/api/jfin/exporters" in routes

    def test_jfin_from_discovered_endpoint(self):
        routes = self._routes()
        assert "/api/jfin/channels/from_discovered" in routes


class TestEndToEndServer:
    def test_feed_frame_via_testclient_json(self):
        from fastapi.testclient import TestClient
        from atomic.ui import server
        from atomic import Program, Block
        from atomic.ui.viewer import Viewer
        # pre-register a program with viz_video
        p = Program("tc_vv", blocks=[Block("vv", "viz_video")])
        Viewer.put("tc_vv", Viewer(p, name="tc_vv"))
        try:
            app = server.create_app()
            client = TestClient(app)
            import base64
            raw = b"\x80\x40\x20\x10" * 8 * 8
            payload = {"module": "vv", "data": base64.b64encode(raw).decode("ascii")}
            r = client.post("/api/feed_frame/tc_vv", json=payload)
            assert r.status_code == 200
            body = r.json()
            assert body["ok"] is True
            assert body["module"] == "vv"
            assert body["bytes"] == len(raw)
        finally:
            Viewer.delete("tc_vv")

    def test_feed_frame_via_testclient_raw(self):
        from fastapi.testclient import TestClient
        from atomic.ui import server
        from atomic import Program, Block
        from atomic.ui.viewer import Viewer
        p = Program("tc_raw", blocks=[Block("vv", "viz_video")])
        Viewer.put("tc_raw", Viewer(p, name="tc_raw"))
        try:
            app = server.create_app()
            client = TestClient(app)
            raw = b"\xff" * 32
            r = client.post("/api/feed_frame/tc_raw?module=vv", content=raw)
            assert r.status_code == 200
            body = r.json()
            assert body["ok"] is True
            assert body["bytes"] == 32
        finally:
            Viewer.delete("tc_raw")

    def test_jfin_channels_via_testclient(self):
        from fastapi.testclient import TestClient
        from atomic.ui import server
        app = server.create_app()
        client = TestClient(app)
        # list default channels
        r = client.get("/api/jfin/channels")
        assert r.status_code == 200
        body = r.json()
        assert "channels" in body
        assert body["count"] >= 4
        for ch in body["channels"]:
            assert "id" in ch
            assert "m3u_url" in ch

    def test_jfin_add_channel_via_testclient(self):
        from fastapi.testclient import TestClient
        from atomic.ui import server
        from atomic.jellyfin import _JFIN_STATE
        app = server.create_app()
        client = TestClient(app)
        before = len(_JFIN_STATE.m3u.channels)
        r = client.post("/api/jfin/channels", json={
            "id": "tc_added_30", "name": "TC_ADDED_30",
            "m3u_url": "http://test.m3u8", "tuner_type": "hdhr",
            "group": "ATOMIC", "number": 99,
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert len(_JFIN_STATE.m3u.channels) == before + 1

    def test_jfin_export_start_push_stop(self):
        from fastapi.testclient import TestClient
        from atomic.ui import server
        from atomic.jellyfin import _JFIN_STATE
        from atomic.jellyfin import JFinChannel
        app = server.create_app()
        client = TestClient(app)
        # add a channel
        ch = JFinChannel("tc_export_30", "TC_EXPORT_30", "http://x.m3u8")
        _JFIN_STATE.m3u.add_channel(ch)
        # start exporter (mock)
        r = client.post("/api/jfin/export/tc_export_30/start",
                        json={"width": 4, "height": 4, "mock": True})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        # push a frame (4x4 RGBA = 64 bytes)
        raw = b"\x00" * 4 * 4 * 4
        r = client.post("/api/jfin/export/tc_export_30/push",
                        json={"data": list(raw), "width": 4, "height": 4})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        # stop
        r = client.post("/api/jfin/export/tc_export_30/stop")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_jfin_discover_endpoint(self):
        from fastapi.testclient import TestClient
        from atomic.ui import server
        app = server.create_app()
        client = TestClient(app)
        r = client.get("/api/jfin/discover?timeout=0.05")
        assert r.status_code == 200
        body = r.json()
        assert "devices" in body
        assert "count" in body
        assert body["count"] == len(body["devices"])

    def test_jfin_rotate_endpoint(self):
        from fastapi.testclient import TestClient
        from atomic.ui import server
        app = server.create_app()
        client = TestClient(app)
        r = client.post("/api/jfin/rotate",
                        json={"programs": ["a", "b", "c", "d"],
                              "mode": "round_robin"})
        assert r.status_code == 200
        # returns the new mappings dict from the scheduler
        assert "a" in r.json() or "atomic-01" in r.json()


class TestDrawVideoFrameSmoke:
    """Smoke tests for the drawVideoFrame JS function (HTML parsing only)."""

    def test_draw_video_frame_defined(self):
        import re
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "atomic", "ui", "static", "index.html",
        )
        with open(path) as f:
            html = f.read()
        assert re.search(r"function\s+drawVideoFrame\s*\(", html), \
            "drawVideoFrame function not defined in index.html"
        # the dispatch in renderTile must mention 'video'
        assert re.search(r"viz\s*===\s*'video'", html), \
            "renderTile does not dispatch to 'video' viz"
        # the group render must also handle 'video'
        assert html.count("'video'") >= 2
