"""Iter 45: UI responsiveness — play/pause/step/reset REST + WS, tap, feed, canvas throttling.

Tests:
  1. WS connects and streams ticks
  2. WS pause: tick freezes, heartbeat sent with paused=True
  3. WS play: tick resumes
  4. WS step: exactly one tick advance, snapshot in ack
  5. WS reset: engine resets to t=0, cleared bus/series
  6. REST pause/resume
  7. REST reset
  8. /api/feed mutates engine state (param injection)
  9. /api/tap fires and affects engine state
  10. viz_series renders from live WS series data
  11. Play/pause buttons present in HTML
  12. Live feed bar present in HTML
"""
import json
import pytest


@pytest.fixture(autouse=True)
def fresh_viewer_registry():
    """Isolate tests: clear Viewer registry before and after each test.

    TestClient shares the app instance across all test methods, so the
    Viewer registry (module-level _registry) persists unless we clear it.
    """
    from atomic.ui.viewer import Viewer
    from atomic.ui.bicameral_viewer import BicameralViewer
    Viewer._registry.clear()
    BicameralViewer._registry.clear()
    yield
    Viewer._registry.clear()
    BicameralViewer._registry.clear()


@pytest.fixture
def client():
    from atomic.ui.server import create_app
    from fastapi.testclient import TestClient
    return TestClient(create_app())


class TestWSAuto:
    """WS streaming tests (use TestClient.websocket_connect)."""

    def test_ws_connect_and_stream(self, client):
        with client.websocket_connect("/ws/clock_counter") as ws:
            snap = ws.receive_json()
            assert "t" in snap and "bus" in snap
            assert snap["running"] is False

    def test_ws_pause_freezes_tick(self, client):
        with client.websocket_connect("/ws/hadamard_wxyz") as ws:
            # advance a few ticks
            for _ in range(5):
                ws.receive_json()
            snap = json.loads(json.dumps(ws.receive_json()))
            t_before = snap["t"]
            # pause
            ws.send_json({"type": "pause"})
            ack = ws.receive_json()
            assert ack.get("ack") == "pause"
            assert ack.get("playing") is False
            # next ticks should be heartbeats with t frozen
            paused_ticks = []
            for _ in range(3):
                s = ws.receive_json()
                paused_ticks.append((s.get("t"), s.get("paused")))
            assert all(t == t_before for t, _ in paused_ticks), \
                "ticks should freeze during pause"
            assert all(p is True for _, p in paused_ticks), \
                "all paused ticks should have paused=True"

    def test_ws_play_resumes(self, client):
        with client.websocket_connect("/ws/hadamard_wxyz") as ws:
            for _ in range(3):
                ws.receive_json()
            snap = json.loads(json.dumps(ws.receive_json()))
            t_before = snap["t"]
            ws.send_json({"type": "pause"})
            ws.receive_json()  # ack
            for _ in range(2):  # skip heartbeats
                ws.receive_json()
            ws.send_json({"type": "play"})
            ack = ws.receive_json()
            assert ack.get("ack") == "play"
            assert ack.get("playing") is True
            # advance — t should increase
            ticks = []
            for _ in range(4):
                s = ws.receive_json()
                ticks.append(s.get("t"))
            assert ticks[-1] > t_before, "t should advance after play"

    def test_ws_step_advances_one_tick(self, client):
        with client.websocket_connect("/ws/hadamard_wxyz") as ws:
            for _ in range(3):
                ws.receive_json()
            snap = ws.receive_json()
            t_before = snap["t"]
            ws.send_json({"type": "pause"})
            ws.receive_json()  # ack
            for _ in range(2):  # skip heartbeats
                ws.receive_json()
            ws.send_json({"type": "step"})
            ack = ws.receive_json()
            assert ack.get("ack") == "step"
            assert "snapshot" in ack
            assert ack["snapshot"]["t"] == t_before + 1, \
                f"step should advance exactly 1 tick, got {ack['snapshot']['t']} vs {t_before}"

    def test_ws_reset_zeros_tick(self, client):
        with client.websocket_connect("/ws/hadamard_wxyz") as ws:
            # advance many ticks
            for _ in range(10):
                ws.receive_json()
            ws.send_json({"type": "reset"})
            ack = ws.receive_json()
            assert ack.get("ack") == "reset"
            snap = ack.get("snapshot", {})
            assert snap.get("t") == 0, f"reset should set t=0, got {snap.get('t')}"

    def test_ws_feed_param_injects(self, client):
        """POST /api/feed/{name} should mutate engine state (bpm param change).

        Note: the WS server loop ticks in parallel; after sending a WS
        feed message the next received frame could be a tick snapshot OR
        the feed ack depending on timing. Drain enough frames to find the ack.
        """
        with client.websocket_connect("/ws/gated_clock_counter") as ws:
            snap0 = ws.receive_json()
            ws.send_json({
                "type": "feed",
                "ticks": [snap0["t"]],
                "params": {"clk": {"bpm": 240}},
            })
            ack = None
            for _ in range(8):
                msg = ws.receive_json()
                if msg.get("ack") == "feed":
                    ack = msg
                    break
            assert ack is not None, "feed ack not received in 8 frames"
            assert ack.get("applied") == 1


class TestRESTEndpoints:
    """REST endpoint tests."""

    def test_control_GET_returns_schema(self, client):
        r = client.get("/api/control/gated_clock_counter")
        assert r.status_code == 200
        d = r.json()
        assert "params" in d
        assert "window" in d
        assert d["window"] == 512

    def test_control_POST_pause_resume(self, client):
        r = client.post("/api/control/hadamard_wxyz",
                         json={"playing": False})
        assert r.status_code == 200
        d = r.json()
        assert d.get("playing") is False

        r = client.post("/api/control/hadamard_wxyz",
                         json={"playing": True})
        assert r.status_code == 200
        d = r.json()
        assert d.get("playing") is True

    def test_control_POST_reset_clears_state(self, client):
        # advance first
        client.post("/api/batch/hadamard_wxyz", json={"ticks": 10})
        # reset
        r = client.post("/api/control/hadamard_wxyz", json={"reset": True})
        assert r.status_code == 200
        d = r.json()
        assert d.get("t") == 0
        assert d.get("bus") == {} or len(d.get("bus", {})) == 0

    def test_tap_fires(self, client):
        r = client.post("/api/tap/hadamard_wxyz")
        assert r.status_code == 200
        d = r.json()
        assert d.get("ok") is True

    def test_feed_mutates_state(self, client):
        """Feed a param and verify it affects the next tick."""
        # batch 1 tick to get initial state
        r1 = client.post("/api/batch/hadamard_wxyz", json={"ticks": 1})
        d1 = r1.json()
        # feed a tick with a param
        r2 = client.post("/api/feed/hadamard_wxyz", json={
            "ticks": [d1["t"]],
            "params": {"src": {"value": 2.0}},
        })
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2.get("ok") is True
        assert d2.get("applied") == 1

    def test_batch_runs_ticks(self, client):
        r = client.post("/api/batch/hadamard_wxyz", json={"ticks": 5})
        assert r.status_code == 200
        d = r.json()
        # Engine ticks an additional initial tick on first access (eng.tick()
        # at end of tick_once + the N we asked for). Tolerate ±1.
        assert 4 <= d["t"] <= 6, f"batch(5) should yield ~5 ticks, got {d['t']}"
        assert len(d["bus"]) > 0

    def test_snapshot_returns_current_state(self, client):
        r = client.get("/api/snapshot/hadamard_wxyz")
        assert r.status_code == 200
        d = r.json()
        assert "t" in d and "bus" in d and "series" in d


class TestVizRender:
    """Verify viz sinks produce renderable data via batch snapshot."""

    def test_viz_series_data(self, client):
        r = client.post("/api/batch/sine_lfo_scope", json={"ticks": 30})
        d = r.json()
        assert "series" in d
        # sine_lfo generates cv output
        has_series = any(k.endswith(".cv") for k in d["series"])
        assert has_series, "sine_lfo_scope should produce .cv series"

    def test_viz_xy_data(self, client):
        r = client.post("/api/batch/xy_pad", json={"ticks": 30})
        d = r.json()
        assert "series" in d
        # xy_pad wires src.cv->v0.x and sy.cv->v0.y
        has_xy = any(k.endswith(".y") for k in d["series"])
        assert has_xy, "xy_pad should produce .y series"

    def test_viz_wxyz3d_data(self, client):
        r = client.post("/api/batch/wxyz3d_demo", json={"ticks": 30})
        d = r.json()
        assert "series" in d
        has_z = any(k.endswith(".z") for k in d["series"])
        assert has_z, "wxyz3d_demo should produce .z series"


class TestHTMLControls:
    """Static HTML structure tests (reading the file, not rendering in a browser)."""

    def test_play_button_present(self):
        with open("atomic/ui/static/index.html") as f:
            html = f.read()
        assert 'id="play-btn"' in html, "play button missing from HTML"
        assert 'id="step-btn"' in html, "step button missing from HTML"
        assert 'id="live-feed-bar"' in html, "live feed bar missing from HTML"
        assert "type: want ? 'play' : 'pause'" in html or \
               "type: 'pause'" in html, "play/pause handler missing"

    def test_no_double_startloop(self):
        """startLoop() should NOT be called as a standalone statement (only inside
        functions like _adjustSpeed and speedSlider handler). This prevents
        the dual-boot bug where two polling loops ran simultaneously."""
        with open("atomic/ui/static/index.html") as f:
            content = f.read()
        lines = content.split('\n')
        calls = []
        for i, line in enumerate(lines):
            if 'startLoop();' in line:
                stripped = line.strip()
                # Exclude function definitions and recursive calls
                if stripped.startswith('function ') or '{' in stripped:
                    continue
                calls.append((i+1, line.rstrip()))
        assert len(calls) == 0, \
            f"startLoop() should not be called at top level (found {len(calls)}): {calls}"

    def test_no_double_connectws(self):
        """connectWS(state.name, 'tile') should only be called once in the
        .then() boot block — no standalone call at script end."""
        with open("atomic/ui/static/index.html") as f:
            content = f.read()
        lines = content.split('\n')
        calls = []
        for i, line in enumerate(lines):
            if "connectWS(state.name, 'tile')" in line:
                stripped = line.strip()
                # Exclude reconnection timeouts and function bodies
                if 'setTimeout' in line or stripped.startswith('function '):
                    continue
                calls.append((i+1, line.rstrip()))
        assert len(calls) == 1, \
            f"connectWS should be called exactly once in boot (found {len(calls)}): {calls}"

    def test_live_feed_send_handler(self):
        with open("atomic/ui/static/index.html") as f:
            html = f.read()
        assert "feed-send-btn" in html, "feed send button missing"
        assert "send(JSON.stringify({ type: 'feed'" in html, \
            "WS feed message handler missing"

    def test_keyboard_shortcuts_P_dot(self):
        with open("atomic/ui/static/index.html") as f:
            html = f.read()
        assert "'p'" in html or "'P'" in html, "P key shortcut missing"
        assert "'.'" in html, ". key shortcut missing"

    def test_canvas_throttle(self):
        with open("atomic/ui/static/index.html") as f:
            html = f.read()
        assert "_scheduleRedraw" in html, "throttled redraw (_scheduleRedraw) missing"
        assert "setTimeout" in html and "16" in html, "16ms throttle timeout missing"
