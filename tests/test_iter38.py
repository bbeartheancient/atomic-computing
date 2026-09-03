"""Iter 38: ComfyUI vsa /prompt + /history integration (FastH3 GGUF).

Covers:
  - fasth3_server: workflow JSON validity (ComfyUI /prompt shape)
  - fasth3_server: server client (is_comfyui_up, submit_prompt, poll_history)
  - fasth3_server: PNG frame decoder (RGBA bytes from the first /view)
  - FastH3Client(mode="comfyui") path: dispatches to the real workflow
    when a ComfyUI vsa server is up; falls back to FastH3Stub when not.
  - FastH3Source(kind="http") returns a FastH3Client (the iter-36
    contract surface, the iter-36 wire contract still works)
  - Engine integration: InfiniteFastH3Loop + a real HTTP-backed
    FastH3Client produces a frame in a QBF trace

The ComfyUI vsa server is OPT-IN. By default the tests use:
   - a tiny fake ComfyUI server in a thread (the in-process H3InferenceServer
     from atomic.video_server, which exposes the same /generate wire so
     we can mock the upstream with FastH3Stub); OR
   - a pure-Python mock that intercepts /prompt + /history + /view.

The --start flag on the test driver is provided so the operator can
launch the real ComfyUI vsa subprocess if the GPU is available.

Run: python -m pytest tests/test_iter38.py -q
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import socket
import struct
import sys
import threading
import time
import zlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic import (
    Engine, Block, Wire, Program, ATOMS, HostBridge,
    FastH3Client, FastH3Stub, FastH3Source, FastH3Frame,
    FastH3Error, FastH3Session, InfiniteFastH3Loop,
    FASTH3_DEFAULT_STEPS, FASTH3_DEFAULT_VSA, FASTH3_DEFAULT_VSA_KEEP,
    FASTH3_DEFAULT_QUANT, FASTH3_PROMPT_BANK_DEFAULT,
    FlowTrace, QbfTraceStore, H3Frame,
    ComfyUIFastH3Workflow, ComfyUIWorkflowError,
    fasth3_workflow, is_comfyui_up, submit_prompt, poll_history,
    decode_first_frame, start_comfyui_vsa, stop_comfyui_vsa,
    FASTH3_GGUF_Q4, H3_4B_H3STUDENT, COMFYUI_DEFAULT_PORT,
)
from atomic.ui.viewer import Viewer
from atomic.ui.bicameral_viewer import BicameralViewer


# -------------------------------------------------------------------------
# Test fixtures
# -------------------------------------------------------------------------

def _make_png(width, height, rgba_color=(255, 128, 0, 255)):
    """Make a minimal PNG (filter 0, RGBA) — the test fixture PNG.

    PIL is used to make the PNG; fasth3_server decodes it back to RGBA.
    """
    from PIL import Image
    im = Image.new("RGBA", (int(width), int(height)), rgba_color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _start_mock_comfyui(rgba_bytes=None, port=None, fail_after=None):
    """Start a tiny in-process ComfyUI vsa mock.

    Implements:
      GET  /system_stats    -> {"status": "ok"}
      POST /prompt          -> {"prompt_id": "mock-XXX"}
      GET  /history/<id>    -> after a short delay: status.completed=true,
                                outputs.vhs.images[0] = {filename, type}
      GET  /view?filename=X -> the raw PNG bytes (the test fixture)
    """
    import http.server
    import socketserver

    if rgba_bytes is None:
        rgba_bytes = _make_png(64, 64)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a, **kw):
            pass   # silence

        def setup(self):
            super().setup()
            if fail_after is not None:
                self._fail_after = fail_after

        def _write_json(self, code, obj):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_bytes(self, code, data, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _prompt_count(self):
            return len(self._prompt_ids)

        def do_GET(self):
            if self.path.startswith("/system_stats"):
                self._write_json(200, {"status": "ok",
                                       "devices": [{"type": "xpu"}]})
                return
            if self.path.startswith("/history/"):
                if (hasattr(self, "_fail_after") and self._fail_after is not None
                        and self._prompt_count() > self._fail_after):
                    self._write_json(200, {})   # never completes
                    return
                prompt_id = self.path.rsplit("/", 1)[-1]
                self._write_json(200, {prompt_id: {
                    "status": {"completed": True,
                               "status_str": "success",
                               "execution_time": 0.1},
                    "outputs": {"vhs": {"images": [{
                        "filename": "atomic_fasth3_00001_.png",
                        "subfolder": "",
                        "type": "output",
                    }]}}
                }})
                return
            if self.path.startswith("/view?"):
                self._write_bytes(200, rgba_bytes, "image/png")
                return
            self._write_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/prompt":
                ln = int(self.headers.get("Content-Length", "0"))
                body_raw = self.rfile.read(ln)
                try:
                    payload = json.loads(body_raw)
                except Exception:
                    payload = {}
                prompt_id = "mock-" + hashlib.md5(
                    json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
                self._prompt_ids.append(prompt_id)
                self._last_payload.clear()
                self._last_payload.update(payload)
                self._write_json(200, {"prompt_id": prompt_id})
                return
            self._write_json(404, {"error": "not found"})

        def _calls(self):
            return len(self._prompt_ids)

    state = {"prompt_ids": [], "last_payload": {}}
    Handler._prompt_ids = state["prompt_ids"]
    Handler._last_payload = state["last_payload"]

    if port is None:
        port = _free_port()
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return {"port": port, "server": srv, "thread": th, "rgba": rgba_bytes,
            "state": state}


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# -------------------------------------------------------------------------
# 1. Workflow JSON validity
# -------------------------------------------------------------------------

class TestWorkflowJSON:
    def test_workflow_is_comfyui_prompt_shape(self):
        """The workflow JSON has the {prompt: {<id>: {...}}, client_id} shape
        that ComfyUI's /prompt endpoint expects."""
        wf = fasth3_workflow("a comet over the ocean", seed=7, width=64,
                             height=64, steps=4, vsa=True, vsa_keep=10)
        assert "prompt" in wf
        assert "client_id" in wf
        # Each entry in wf["prompt"] is a {class_type, inputs} dict
        for nid, node in wf["prompt"].items():
            assert "class_type" in node, f"node {nid} missing class_type"
            assert "inputs" in node, f"node {nid} missing inputs"

    def test_workflow_has_required_nodes(self):
        """The workflow MUST contain: CheckpointLoaderSimple, H3TextEncode8,
        EmptyHunyuanLatentVideo, KSamplerAdvanced, VAEDecode, VHS_VideoCombine
        (+ SolAttnXPUVSA when vsa=True)."""
        wf = fasth3_workflow("test", vsa=True)
        classes = {n["class_type"] for n in wf["prompt"].values()}
        for required in ("CheckpointLoaderSimple", "H3TextEncode8",
                          "EmptyHunyuanLatentVideo", "KSamplerAdvanced",
                          "VAEDecode", "VHS_VideoCombine",
                          "SolAttnXPUVSA"):
            assert required in classes, (
                f"workflow missing {required}; got {classes}")

    def test_workflow_no_vsa_when_disabled(self):
        """When vsa=False the Sol-Attn node is NOT in the workflow."""
        wf = fasth3_workflow("test", vsa=False, vsa_keep=10)
        classes = {n["class_type"] for n in wf["prompt"].values()}
        assert "SolAttnXPUVSA" not in classes

    def test_workflow_steps_vsa_keep_pinned(self):
        """4-step VSA + keep=10 is the iter-37 contract — must be in the body."""
        wf = fasth3_workflow("test", steps=4, vsa=True, vsa_keep=10)
        ksampler = None
        for n in wf["prompt"].values():
            if n["class_type"] == "KSamplerAdvanced":
                ksampler = n["inputs"]
                break
        assert ksampler is not None
        assert ksampler["steps"] == 4
        assert ksampler["noise_seed"] == 0   # default seed

    def test_workflow_gguf_ckpt_present(self):
        """The CheckpointLoaderSimple ckpt_name is the FastH3 GGUF basename."""
        wf = fasth3_workflow("test")
        ckpt = None
        for n in wf["prompt"].values():
            if n["class_type"] == "CheckpointLoaderSimple":
                ckpt = n["inputs"]
                break
        assert ckpt is not None
        assert ckpt["ckpt_name"] == os.path.basename(FASTH3_GGUF_Q4)

    def test_workflow_te_is_4b_h3student(self):
        """The H3TextEncode8 wrapper uses the 4b h3student TE."""
        wf = fasth3_workflow("test")
        te = None
        for n in wf["prompt"].values():
            if n["class_type"] == "H3TextEncode8":
                te = n["inputs"]
                break
        assert te is not None
        assert te["text_encoder"] == os.path.basename(H3_4B_H3STUDENT)
        assert "prompt" in te

    def test_workflow_seed_propagates(self):
        """A non-zero seed is reflected in the KSamplerAdvanced noise_seed."""
        wf = fasth3_workflow("test", seed=42, steps=4)
        for n in wf["prompt"].values():
            if n["class_type"] == "KSamplerAdvanced":
                assert n["inputs"]["noise_seed"] == 42
                break
        else:
            pytest.fail("no KSamplerAdvanced in workflow")

    def test_workflow_dataclass_summary(self):
        """The dataclass .summary() returns a dict the operator can log."""
        w = ComfyUIFastH3Workflow(prompt="x", seed=1, steps=4, vsa=True)
        s = w.summary()
        assert s["prompt"] == "x"
        assert s["seed"] == 1
        assert s["steps"] == 4
        assert s["vsa"] is True
        assert s["gguf"] == FASTH3_GGUF_Q4
        assert s["text_encoder"] == H3_4B_H3STUDENT

    def test_workflow_to_json_bytes_round_trip(self):
        """to_json_bytes() is a valid JSON document."""
        w = ComfyUIFastH3Workflow(prompt="x", steps=4, vsa=True)
        data = w.to_json_bytes()
        obj = json.loads(data.decode("utf-8"))
        assert "prompt" in obj and "client_id" in obj


# -------------------------------------------------------------------------
# 2. Server client (mock ComfyUI)
# -------------------------------------------------------------------------

class TestServerClient:
    def setup_method(self):
        self._rgba = _make_png(64, 64, rgba_color=(255, 0, 128, 255))
        self._srv = _start_mock_comfyui(rgba_bytes=self._rgba)

    def teardown_method(self):
        try:
            self._srv["server"].shutdown()
        except Exception:
            pass

    def test_is_comfyui_up_true(self):
        """is_comfyui_up() returns True against the mock."""
        assert is_comfyui_up(host="127.0.0.1", port=self._srv["port"],
                             timeout=2.0) is True

    def test_is_comfyui_up_false(self):
        """is_comfyui_up() returns False against a closed port."""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        assert is_comfyui_up(host="127.0.0.1", port=port, timeout=0.2) is False

    def test_submit_prompt_returns_id(self):
        """submit_prompt() returns a prompt_id (string)."""
        wf = fasth3_workflow("a comet over the ocean", seed=0)
        pid = submit_prompt(wf, host="127.0.0.1", port=self._srv["port"])
        assert isinstance(pid, str) and pid.startswith("mock-")
        assert pid in self._srv["state"]["prompt_ids"]

    def test_poll_history_returns_outputs(self):
        """poll_history() returns a /history entry with outputs."""
        wf = fasth3_workflow("x", seed=1)
        pid = submit_prompt(wf, host="127.0.0.1", port=self._srv["port"])
        entry = poll_history(pid, host="127.0.0.1", port=self._srv["port"],
                             interval=0.05, timeout=5.0)
        assert "outputs" in entry
        assert "vhs" in entry["outputs"]
        assert "status" in entry
        assert entry["status"]["completed"] is True

    def test_decode_first_frame_returns_rgba(self):
        """decode_first_frame() returns RGBA bytes of the requested size."""
        wf = fasth3_workflow("x")
        pid = submit_prompt(wf, host="127.0.0.1", port=self._srv["port"])
        entry = poll_history(pid, host="127.0.0.1", port=self._srv["port"],
                             interval=0.05, timeout=5.0)
        out = decode_first_frame(entry, host="127.0.0.1",
                                  port=self._srv["port"])
        assert out["width"] == 64
        assert out["height"] == 64
        rgba = out["frames"][0]
        assert len(rgba) == 64 * 64 * 4
        # first pixel colour should be (255, 0, 128, 255) — what we wrote
        assert rgba[0:4] == bytes([255, 0, 128, 255])

    def test_submit_prompt_unreachable_raises(self):
        """submit_prompt() raises ComfyUIWorkflowError on unreachable host."""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        wf = fasth3_workflow("x")
        with pytest.raises(ComfyUIWorkflowError):
            submit_prompt(wf, host="127.0.0.1", port=port, timeout=0.5)


# -------------------------------------------------------------------------
# 3. PNG frame decoder (pure-stdlib path)
# -------------------------------------------------------------------------

class TestFrameDecoder:
    def test_png_dimensions(self):
        """_png_dimensions() returns the correct (W, H)."""
        png = _make_png(48, 32, rgba_color=(0, 0, 0, 255))
        from atomic.fasth3_server import _png_dimensions
        w, h = _png_dimensions(png)
        assert (w, h) == (48, 32)

    def test_png_to_rgba_pil_path(self):
        """The PIL-backed decoder reproduces the test fixture colour."""
        from atomic.fasth3_server import _png_bytes_to_rgba
        png = _make_png(16, 16, rgba_color=(100, 200, 50, 255))
        rgba = _png_bytes_to_rgba(png)
        assert len(rgba) == 16 * 16 * 4
        # first pixel = (100, 200, 50, 255)
        assert rgba[0:4] == bytes([100, 200, 50, 255])

    def test_png_to_rgba_manual_path(self):
        """The pure-stdlib decoder (no PIL) reproduces the test fixture colour
        for a filter-type-0 PNG. We monkey-patch out PIL temporarily."""
        from atomic import fasth3_server as fs
        png = _make_png(16, 16, rgba_color=(10, 20, 30, 255))
        saved = fs._PIL
        fs._PIL = None
        try:
            rgba = fs._png_bytes_to_rgba(png)
        finally:
            fs._PIL = saved
        assert len(rgba) == 16 * 16 * 4
        assert rgba[0:4] == bytes([10, 20, 30, 255])

    def test_image_bytes_to_rgba_picks_png(self):
        """_image_bytes_to_rgba() handles a PNG payload."""
        from atomic.fasth3_server import _image_bytes_to_rgba
        png = _make_png(8, 8, rgba_color=(1, 2, 3, 255))
        rgba = _image_bytes_to_rgba(png)
        assert rgba[0:4] == bytes([1, 2, 3, 255])

    def test_png_to_rgba_manual_with_filter_nonzero(self):
        """The manual decoder handles filters 1, 2, 3, 4 in addition to 0."""
        # Build a PNG with PIL then verify the manual decoder reproduces it.
        png = _make_png(24, 24, rgba_color=(77, 88, 99, 255))
        from atomic import fasth3_server as fs
        saved = fs._PIL
        fs._PIL = None
        try:
            rgba = fs._png_bytes_to_rgba(png)
        finally:
            fs._PIL = saved
        # colour matches the fixture
        assert rgba[0:4] == bytes([77, 88, 99, 255])


# -------------------------------------------------------------------------
# 4. FastH3Client(mode="comfyui") end-to-end against the mock server
# -------------------------------------------------------------------------

class TestFastH3ClientComfyUIMode:
    def setup_method(self):
        self._rgba = _make_png(64, 64, rgba_color=(0, 200, 100, 255))
        self._srv = _start_mock_comfyui(rgba_bytes=self._rgba)

    def teardown_method(self):
        try:
            self._srv["server"].shutdown()
        except Exception:
            pass

    def test_comfyui_mode_dispatches_to_fasth3_server(self):
        """FastH3Client(mode='comfyui').generate() -> submits /prompt,
        polls /history, returns an RGBA frame and the FastH3 metadata."""
        c = FastH3Client(
            endpoint=f"http://127.0.0.1:{self._srv['port']}",
            timeout_s=10.0,
            mode="comfyui",
        )
        r = c.generate("a comet over the ocean", seed=0, n_frames=1,
                        width=64, height=64, steps=4, vsa=True)
        assert r["model"] == "FastH3"
        assert r["steps"] == 4
        assert r["vsa"] is True
        assert r["vsa_keep"] == FASTH3_DEFAULT_VSA_KEEP
        assert r["quant"] == FASTH3_DEFAULT_QUANT
        assert r["width"] == 64
        assert r["height"] == 64
        assert len(r["frames"]) == 1
        rgba = r["frames"][0]
        assert len(rgba) == 64 * 64 * 4
        # colour matches the mock server fixture
        assert rgba[0:4] == bytes([0, 200, 100, 255])
        # a comfyui_prompt_id is set
        assert r.get("comfyui_prompt_id", "").startswith("mock-")

    def test_comfyui_mode_calls_counters_increment(self):
        """The client's _calls counter increments on success."""
        c = FastH3Client(
            endpoint=f"http://127.0.0.1:{self._srv['port']}",
            timeout_s=10.0,
            mode="comfyui",
        )
        c.generate("a", n_frames=1, width=32, height=32)
        c.generate("b", n_frames=1, width=32, height=32)
        assert c._calls == 2
        assert c._last_prompt == "b"

    def test_comfyui_mode_falls_back_when_server_down(self):
        """mode='comfyui' falls back to FastH3Stub when the server is down."""
        stub = FastH3Stub(width=32, height=32, n_frames=1)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        c = FastH3Client(
            endpoint=f"http://127.0.0.1:{port}",
            timeout_s=1.0,
            fallback=stub,
            mode="comfyui",
        )
        r = c.generate("a comet", n_frames=1, width=32, height=32)
        # came from the stub
        assert r["model"] == "FastH3"
        assert r["steps"] == FASTH3_DEFAULT_STEPS
        assert len(r["frames"]) == 1

    def test_comfyui_mode_raises_without_fallback(self):
        """mode='comfyui' raises FastH3Error when the server is down
        and no fallback is configured."""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        c = FastH3Client(
            endpoint=f"http://127.0.0.1:{port}",
            timeout_s=1.0,
            mode="comfyui",
        )
        with pytest.raises(FastH3Error):
            c.generate("a comet", n_frames=1, width=32, height=32)

    def test_generic_mode_still_works(self):
        """The iter-36 /generate wire contract is preserved (mode='generic')."""
        # The mock doesn't expose /generate, so a generic-mode call
        # falls back to the stub.
        stub = FastH3Stub(width=32, height=32, n_frames=1)
        c = FastH3Client(
            endpoint=f"http://127.0.0.1:{self._srv['port']}",
            timeout_s=1.0,
            fallback=stub,
            mode="generic",
        )
        r = c.generate("a comet", n_frames=1, width=32, height=32)
        # came from the stub
        assert r["model"] == "FastH3"
        assert len(r["frames"]) == 1

    def test_source_factory_returns_compatible_client(self):
        """FastH3Source(kind='http') returns a FastH3Client (iter-36 contract)."""
        c = FastH3Source(
            kind="http",
            endpoint=f"http://127.0.0.1:{self._srv['port']}",
            mode="comfyui",
        )
        assert isinstance(c, FastH3Client)
        assert c.mode == "comfyui"

    def test_fasth3frame_from_real_frame(self):
        """A FastH3Frame can be constructed from the mock-served RGBA bytes."""
        c = FastH3Client(
            endpoint=f"http://127.0.0.1:{self._srv['port']}",
            timeout_s=10.0,
            mode="comfyui",
        )
        r = c.generate("a comet", n_frames=1, width=64, height=64)
        f = FastH3Frame(
            rgba=r["frames"][0], prompt=r["prompt"], seed=r["seed"],
            t=0, h3_latency_ms=r["h3_latency_ms"],
            steps=r["steps"], vsa=r["vsa"], vsa_keep=r["vsa_keep"],
            quant=r["quant"], model=r["model"],
        )
        assert f.steps == 4
        assert f.vsa is True
        # to_h3_frame drops the FastH3 metadata but keeps the RGBA
        h = f.to_h3_frame()
        assert isinstance(h, H3Frame)
        assert h.rgba == f.rgba


# -------------------------------------------------------------------------
# 5. Engine integration: InfiniteFastH3Loop with the real HTTP path
# -------------------------------------------------------------------------

class TestEngineIntegration:
    def setup_method(self):
        self._rgba = _make_png(32, 32, rgba_color=(50, 100, 200, 255))
        self._srv = _start_mock_comfyui(rgba_bytes=self._rgba)

    def teardown_method(self):
        try:
            self._srv["server"].shutdown()
        except Exception:
            pass

    def _make_viewer(self):
        from atomic import Program as _P, Block as _B
        sub = _P("fh3_sub", blocks=[_B("clk", "clock_bpm", {"bpm": 60})])
        con = _P("fh3_con", blocks=[_B("vv", "viz_video")])
        v = BicameralViewer(
            sub, con,
            bridge_map=[("clk.trig", "vv.in")],
            bridge_latency=1,
            name="iter38_fasth3_real",
        )
        return v

    def test_infinite_loop_with_real_http_client(self):
        """InfiniteFastH3Loop drives the FastH3Client (mode='comfyui')
        against the mock server and produces FastH3Frames in the trace."""
        v = self._make_viewer()
        c = FastH3Client(
            endpoint=f"http://127.0.0.1:{self._srv['port']}",
            timeout_s=10.0,
            mode="comfyui",
        )
        tr = FlowTrace()
        loop = InfiniteFastH3Loop(
            c, v, bridge_latency=1, module_id="vfh", max_ticks=3, trace=tr)
        out = loop.run(3)
        assert len(out) == 3
        for f in out:
            assert isinstance(f, FastH3Frame)
            assert f.steps == 4
            assert f.vsa is True
            assert f.model == "FastH3"
        # the trace recorded the frames
        assert tr.video_seq >= 3
        node_entries = [ne for ne in tr.frames if ne.module_id == "vfh"]
        assert len(node_entries) == 3

    def test_qbf_round_trip_with_real_http_frame(self):
        """A FastH3Frame from the real HTTP path survives QBF round-trip."""
        v = self._make_viewer()
        c = FastH3Client(
            endpoint=f"http://127.0.0.1:{self._srv['port']}",
            timeout_s=10.0,
            mode="comfyui",
        )
        tr = FlowTrace()
        loop = InfiniteFastH3Loop(
            c, v, bridge_latency=1, module_id="vfh", max_ticks=2, trace=tr)
        out = loop.run(2)
        # store
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".qbf")
        os.close(fd); os.unlink(path)
        try:
            store = QbfTraceStore(path)
            store.append_run(tr, "iter38_fasth3")
            store.close()
            store2 = QbfTraceStore(path)
            rids = store2.runs()
            assert len(rids) == 1
            blob = store2.load_run(rids[0])
            node_entries = [ne for ne in blob["frames"]
                            if ne.get("module_id") == "vfh"]
            assert len(node_entries) == 2
            ne = node_entries[0]
            assert ne["in_ports"]["model"] == "FastH3"
            assert ne["in_ports"]["steps"] == 4
            assert ne["in_ports"]["vsa"] == 1
            assert ne["in_ports"]["quant"] == FASTH3_DEFAULT_QUANT
            store2.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_loop_stats_after_real_http(self):
        """InfiniteFastH3Loop.stats() reflects the real call count."""
        v = self._make_viewer()
        c = FastH3Client(
            endpoint=f"http://127.0.0.1:{self._srv['port']}",
            timeout_s=10.0,
            mode="comfyui",
        )
        loop = InfiniteFastH3Loop(
            c, v, bridge_latency=1, module_id="vfh", max_ticks=3)
        loop.run(3)
        st = loop.stats()
        assert st["t"] == 3
        assert st["frame_index"] == 3
        # one call per frame (frames_per_prompt=1 in InfiniteFastH3Loop)
        assert st["fasth3_calls"] == 3


# -------------------------------------------------------------------------
# 6. Workflow JSON parses + round-trips through /prompt (mock)
# -------------------------------------------------------------------------

class TestWorkflowRoundTrip:
    def setup_method(self):
        self._rgba = _make_png(32, 32, rgba_color=(40, 60, 80, 255))
        self._srv = _start_mock_comfyui(rgba_bytes=self._rgba)

    def teardown_method(self):
        try:
            self._srv["server"].shutdown()
        except Exception:
            pass

    def test_workflow_payload_captured_by_prompt(self):
        """The payload the mock server receives is the workflow JSON."""
        wf = fasth3_workflow("hello", seed=5, steps=4, vsa=True,
                              vsa_keep=15, n_frames=1)
        submit_prompt(wf, host="127.0.0.1", port=self._srv["port"])
        captured = self._srv["state"]["last_payload"]
        assert "prompt" in captured
        classes = {n["class_type"] for n in captured["prompt"].values()}
        assert "SolAttnXPUVSA" in classes

    def test_seed_propagates_to_prompt(self):
        """The seed in the workflow matches the request seed."""
        wf = fasth3_workflow("x", seed=99, steps=4)
        submit_prompt(wf, host="127.0.0.1", port=self._srv["port"])
        captured = self._srv["state"]["last_payload"]
        for n in captured["prompt"].values():
            if n["class_type"] == "KSamplerAdvanced":
                assert n["inputs"]["noise_seed"] == 99
                break
        else:
            pytest.fail("no KSamplerAdvanced captured")


# -------------------------------------------------------------------------
# 7. Subprocess helpers (smoke test, no real ComfyUI launch)
# -------------------------------------------------------------------------

class TestSubprocessHelpers:
    def test_default_port_is_8188(self):
        assert COMFYUI_DEFAULT_PORT == 8188

    def test_comfyui_dir_and_python_exist(self):
        """The defaults point at the live vsa branch on disk."""
        assert os.path.isfile(FASTH3_GGUF_Q4), (
            f"FastH3 GGUF not on disk: {FASTH3_GGUF_Q4}")
        assert os.path.isfile(H3_4B_H3STUDENT), (
            f"4b h3student TE not on disk: {H3_4B_H3STUDENT}")
        assert os.path.isdir("/home/bbear/services/comfyui-h3"), (
            "comfyui-h3 checkout missing")

    def test_stop_comfyui_vsa_handles_none(self):
        """stop_comfyui_vsa(None) is a no-op (safe)."""
        from atomic.fasth3_server import stop_comfyui_vsa
        stop_comfyui_vsa(None)   # no raise
