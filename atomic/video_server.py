"""H3InferenceServer: a local FastAPI server that serves H3 video generation.

This module provides H3InferenceServer — a runnable FastAPI app on localhost:8765
that implements the same contract as the real H3 FastVideo GPU1 service:

  GET  /health          -> {"status": "ok", "model": "H3-FastVideo", "running": True}
  POST /generate        -> {"frames_b64": [...], "prompt": ..., "seed": ...,
                            "h3_latency_ms": ..., "width": W, "height": H}

The server is backed by H3Stub (deterministic, GPU-free) so the full
end-to-end pipeline (viz_video_h3 atom -> tile wall -> QBF trace) is
exercisable without a B200 GPU. The operator replaces the stub with a
real H3 FastVideo process on GPU1 when the hardware is available.

Lifecycle:
  server = H3InferenceServer(width=64, height=64, n_frames=1)
  server.start()   # spawn uvicorn in a background thread
  ... use it ...
  server.stop()     # kill the thread
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
import struct
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

__all__ = [
    "H3InferenceServer",
    "H3VideoApp",
    "H3ServerError",
    "ComfyUIH3Bridge",
    "_H3_STUB_REGISTRY",
]


_H3_STUB_REGISTRY: dict[str, object] = {}
_REGISTRY_PORT: dict[int, "H3InferenceServer"] = {}


# -------------------------------------------------------------------------
# Local stub (copied from video.py to avoid circular imports; identical logic)
# -------------------------------------------------------------------------

class _LocalStub:
    """Deterministic local generator (no GPU). Renders RGBA frames whose
    pixels are a function of (prompt, seed, frame_index)."""

    def __init__(self, width=64, height=64, n_frames=1, latency_ms=0.0):
        self.width = int(width)
        self.height = int(height)
        self.n_frames = int(n_frames)
        self.latency_ms = float(latency_ms)
        self._calls = 0

    def _seed_for(self, prompt, seed):
        if seed is None:
            seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        return int(seed)

    def _render(self, prompt, seed, frame_index, width=None, height=None):
        import math as _m
        w = int(width) if width else self.width
        h = int(height) if height else self.height
        combined = (int(seed) * 1_000_003) ^ (int(frame_index) * 2654435761) & 0xFFFFFFFF
        rng = random.Random(combined)
        n = w * h
        prompt_bytes = (prompt or "").encode("utf-8")
        prompt_hash = int(hashlib.sha256(prompt_bytes).hexdigest()[:16], 16)
        r0 = (prompt_hash >> 8) & 0xFF
        g0 = (prompt_hash >> 16) & 0xFF
        b0 = (prompt_hash >> 24) & 0xFF
        out = bytearray(n * 4)
        for i in range(n):
            x = i % w
            y = i // w
            cx, cy = w / 2.0, h / 2.0
            d = _m.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            wave = _m.sin((d / max(1.0, w)) * 6.2832 - frame_index * 0.6)
            r = max(0, min(255, int(r0 + 60 * wave + rng.randint(-25, 25))))
            g = max(0, min(255, int(g0 + 40 * (-wave) + rng.randint(-20, 20))))
            b = max(0, min(255, int(b0 + 50 * wave + rng.randint(-20, 20))))
            j = i * 4
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            out[j + 3] = 255
        return bytes(out)

    def generate(self, prompt, seed=None, n_frames=None,
                 width=None, height=None):
        t0 = time.perf_counter()
        seed = self._seed_for(prompt, seed)
        n = int(n_frames) if n_frames else self.n_frames
        w = int(width) if width else self.width
        h = int(height) if height else self.height
        frames = [self._render(prompt, seed, k, w, h) for k in range(n)]
        elapsed = (time.perf_counter() - t0) * 1000.0 + self.latency_ms
        self._calls += 1
        return {
            "prompt": prompt, "seed": seed, "frames": frames,
            "h3_latency_ms": elapsed, "width": w, "height": h,
        }


# -------------------------------------------------------------------------
# FastAPI app (the server itself)
# -------------------------------------------------------------------------

def H3VideoApp(stub=None):
    """Build the FastAPI app. `stub` is a _LocalStub (or compatible)."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError:
        raise ImportError("H3InferenceServer requires fastapi. Install: pip install fastapi uvicorn")

    app = FastAPI(title="H3-Inference-Server", version="0.1")

    if stub is None:
        stub = _LocalStub()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": "H3-FastVideo",
            "running": True,
            "width": stub.width,
            "height": stub.height,
            "calls": stub._calls,
        }

    @app.post("/generate")
    async def generate(payload: dict):
        prompt = str(payload.get("prompt", ""))
        seed = int(payload["seed"]) if "seed" in payload else None
        n_frames = int(payload.get("n_frames", 1))
        width = int(payload.get("width", stub.width))
        height = int(payload.get("height", stub.height))
        steps = int(payload.get("steps", 4))

        r = stub.generate(
            prompt=prompt, seed=seed, n_frames=n_frames,
            width=width, height=height,
        )
        frames_b64 = [base64.b64encode(f).decode("ascii") for f in r["frames"]]
        return {
            "frames_b64": frames_b64,
            "prompt": r["prompt"],
            "seed": r["seed"],
            "h3_latency_ms": round(r["h3_latency_ms"], 2),
            "width": r["width"],
            "height": r["height"],
            "steps": steps,
        }

    return app


class H3InferenceServer:
    """Runnable H3 FastVideo stub server on localhost:8765.

    Lifecycle:
      server = H3InferenceServer(port=8765, width=64, height=64)
      server.start()   # background thread
      server.stop()    # join thread
    """

    def __init__(self, port: int = 8765,
                 width: int = 64, height: int = 64,
                 n_frames: int = 1, latency_ms: float = 0.0,
                 host: str = "localhost"):
        self.port = int(port)
        self.host = str(host)
        self.width = int(width)
        self.height = int(height)
        self.n_frames = int(n_frames)
        self.latency_ms = float(latency_ms)
        self._stub = _LocalStub(width=width, height=height,
                                n_frames=n_frames, latency_ms=latency_ms)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._uv_server = None
        self._running = False
        self._server_url = f"http://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        return self._server_url

    def start(self, blocking: bool = False):
        """Start the server.

        If blocking=True, run in the current thread (for subprocess mode).
        Otherwise spawn a background thread and return immediately.
        Raises OSError if the port is already in use.
        """
        if self._running:
            return
        if blocking:
            try:
                import uvicorn
            except ImportError:
                raise ImportError("H3InferenceServer requires uvicorn. Install: pip install uvicorn")
            app = H3VideoApp(stub=self._stub)
            cfg = uvicorn.Config(app, host=self.host, port=self.port,
                                 log_level="warning")
            server = uvicorn.Server(cfg)
            self._running = True
            server.run()
            self._running = False
        else:
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_thread,
                name=f"H3InferenceServer-{self.port}",
                daemon=True,
            )
            self._thread.start()
            self._running = True
            _REGISTRY_PORT[self.port] = self
            # wait for uvicorn to bind the port + serve /health
            if not self.wait_until_ready(timeout=5.0):
                self._running = False
                _REGISTRY_PORT.pop(self.port, None)
                raise OSError(f"port {self.port} failed to start server")

    def _run_thread(self):
        try:
            import uvicorn
        except ImportError:
            import sys
            print("H3InferenceServer requires uvicorn. Install: pip install uvicorn",
                  file=sys.stderr)
            self._running = False
            return
        app = H3VideoApp(stub=self._stub)
        cfg = uvicorn.Config(app, host=self.host, port=self.port,
                             log_level="warning")
        srv = uvicorn.Server(cfg)
        self._uv_server = srv
        try:
            srv.run()
        except Exception:
            pass
        self._running = False
        self._uv_server = None

    def _port_bound(self) -> bool:
        """Quick socket check: is something listening on self.port?"""
        import socket as _s
        try:
            with _s.create_connection((self.host, self.port), timeout=0.1):
                return True
        except Exception:
            return False

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Poll /health until it returns ok, or timeout (returns False)."""
        import time as _t
        deadline = _t.perf_counter() + timeout
        while _t.perf_counter() < deadline:
            if self._port_bound():
                try:
                    h = self.health()
                    if h.get("status") == "ok":
                        return True
                except Exception:
                    pass
            _t.sleep(0.1)
        return False

    def stop(self):
        """Stop the background server thread."""
        if not self._running:
            return
        self._stop_event.set()
        if self._uv_server is not None:
            try:
                self._uv_server.should_exit = True
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._running = False
        _REGISTRY_PORT.pop(self.port, None)

    @property
    def is_running(self) -> bool:
        return self._running

    def health(self) -> dict:
        """Ping /health and return the JSON response."""
        url = f"{self._server_url}/health"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate(self, prompt: str, seed: int | None = None,
                 n_frames: int | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 steps: int = 4,
                 timeout_s: float = 60.0) -> dict:
        """Call POST /generate on the running server.

        Returns the JSON dict with frames_b64 list.
        Raises H3ServerError on failure.
        """
        url = f"{self._server_url}/generate"
        body = {
            "prompt": prompt,
            "n_frames": n_frames if n_frames is not None else self.n_frames,
            "width": width if width is not None else self.width,
            "height": height if height is not None else self.height,
            "steps": steps,
        }
        if seed is not None:
            body["seed"] = int(seed)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise H3ServerError(f"/generate failed: {e}") from e

    def __repr__(self):
        return (f"H3InferenceServer(port={self.port} width={self.width} "
                f"height={self.height} running={self._running})")


class H3ServerError(RuntimeError):
    pass


# -------------------------------------------------------------------------
# ComfyUI Subprocess Bridge (API-only, iter 32)
# -------------------------------------------------------------------------
# Launches ComfyUI as a subprocess with MiniMax H3 FastVideo, exposing
# the same /generate contract as H3Stub so the harness never knows which
# backend is running.  The ComfyUI web UI is NOT started (--disable-ui);
# only the API server (:8188) is used.  Falls back to the local stub
# if ComfyUI is unavailable (the "degraded but live" mode).
#
# Topology:
#   atomic_ui :18094  <--  H3InferenceServer :8765  <--  ComfyUI :8188
#                                  |                         |
#                            H3Stub (fallback)         H3 FastVideo (GPU1)
#
# The atomic harness never talks to :8188 directly; it only talks to
# :8765, which proxies to :8188 when the bridge is active.
#
# ComfyUI launch args (API-only, no web UI):
#   python main.py --disable-ui --listen 0.0.0.0 --port 8188
#   --extra-model-paths /path/to/models
#
# The bridge polls /history or /system_stats to detect readiness.

class ComfyUIH3Bridge:
    """Subprocess bridge: launches ComfyUI (API-only) and proxies to /generate.

    Args:
        comfyui_dir:    path to the ComfyUI installation
        comfyui_port:   port ComfyUI listens on (default 8188)
        comfyui_host:   host ComfyUI binds to (default localhost)
        h3_port:        port this bridge listens on (default 8766, avoids
                        collision with H3InferenceServer :8765)
        h3_host:        host for the proxy server (default localhost)
        extra_args:     extra command-line args for ComfyUI launch
        stub:           _LocalStub instance for degraded-but-live fallback
        launch_timeout: seconds to wait for ComfyUI to become ready
        poll_interval:  seconds between readiness polls
    """

    def __init__(self,
                 comfyui_dir: str = "/home/bbear/comfyui",
                 comfyui_port: int = 8188,
                 comfyui_host: str = "localhost",
                 h3_port: int = 8766,
                 h3_host: str = "localhost",
                 extra_args: tuple[str, ...] = (),
                 stub: "_LocalStub | None" = None,
                 launch_timeout: float = 120.0,
                 poll_interval: float = 1.0):
        self.comfyui_dir = str(comfyui_dir)
        self.comfyui_port = int(comfyui_port)
        self.comfyui_host = str(comfyui_host)
        self.h3_port = int(h3_port)
        self.h3_host = str(h3_host)
        self.extra_args = tuple(extra_args)
        self.stub = stub or _LocalStub()
        self.launch_timeout = float(launch_timeout)
        self.poll_interval = float(poll_interval)

        self._proc: "subprocess.Popen[str] | None" = None
        self._proxy_server: "H3InferenceServer | None" = None
        self._running = False
        self._comfyui_url = f"http://{comfyui_host}:{comfyui_port}"
        self._h3_url = f"http://{h3_host}:{h3_port}"

    @property
    def h3_url(self) -> str:
        return self._h3_url

    @property
    def is_running(self) -> bool:
        return self._running

    # -- ComfyUI process lifecycle -------------------------------------------

    def _comfyui_ready(self) -> bool:
        """Ping ComfyUI's system-stats endpoint to check readiness."""
        import urllib.request as _ur
        url = f"{self._comfyui_url}/system_stats"
        req = _ur.Request(url, headers={"Accept": "application/json"})
        try:
            with _ur.urlopen(req, timeout=5.0) as r:
                return r.status == 200
        except Exception:
            return False

    def _wait_for_comfyui(self, timeout: float) -> bool:
        import time as _t
        deadline = _t.perf_counter() + timeout
        while _t.perf_counter() < deadline:
            if self._comfyui_ready():
                return True
            _t.sleep(self.poll_interval)
        return False

    def _build_comfyui_cmd(self) -> list[str]:
        import sys
        python = sys.executable
        main_py = f"{self.comfyui_dir}/main.py"
        return [
            python, main_py,
            "--disable-ui",
            "--listen", self.comfyui_host,
            "--port", str(self.comfyui_port),
            "--disable-metrics",
        ] + list(self.extra_args)

    def launch_comfyui(self) -> bool:
        """Start ComfyUI as a subprocess (API-only). Returns True if started."""
        import subprocess, sys, os
        if self._proc is not None:
            return True  # already running
        cmd = self._build_comfyui_cmd()
        env = dict(os.environ)
        env["COMFYUI_PORT"] = str(self.comfyui_port)
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=self.comfyui_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            self._proc = None
            return False
        # wait for ready
        return self._wait_for_comfyui(self.launch_timeout)

    def stop_comfyui(self):
        """Terminate the ComfyUI subprocess."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    # -- Proxy /generate to ComfyUI H3 FastVideo --------------------------

    def _generate_via_comfyui(self, prompt: str, seed: int | None = None,
                               n_frames: int = 1,
                               width: int = 64, height: int = 64,
                               steps: int = 4) -> dict:
        """POST /generate -> ComfyUI H3 FastVideo API.

        The ComfyUI H3 FastVideo custom node is assumed to expose a
        /custom/h3_fastvideo/generate endpoint.  If the endpoint is not
        found (404), falls back to the local stub so the harness stays live.
        """
        import json as _json
        import base64 as _b64
        import urllib.request as _ur
        import urllib.error as _ue

        # ComfyUI H3 FastVideo custom node endpoint (iter 32 contract)
        url = f"{self._comfyui_url}/custom/h3_fastvideo/generate"
        body = {
            "prompt": prompt,
            "seed": seed,
            "n_frames": n_frames,
            "width": width,
            "height": height,
            "steps": steps,
        }
        data = _json.dumps(body).encode("utf-8")
        req = _ur.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _ur.urlopen(req, timeout=60.0) as resp:
                r = _json.loads(resp.read().decode("utf-8"))
            frames_b64 = r.get("frames_b64", [])
            frames = [_b64.b64decode(f) for f in frames_b64]
            return {
                "prompt": prompt,
                "seed": r.get("seed", seed or 0),
                "frames": frames,
                "h3_latency_ms": r.get("h3_latency_ms", 0.0),
                "width": width,
                "height": height,
            }
        except (_ue.HTTPError, _ue.URLError, OSError) as e:
            # ComfyUI H3 node not found or not ready -> fallback to stub
            return self.stub.generate(
                prompt=prompt, seed=seed, n_frames=n_frames,
                width=width, height=height,
            )

    def start(self, blocking: bool = False):
        """Start the proxy server that exposes /generate (proxies to ComfyUI).

        If blocking=True, run in the current thread (for subprocess mode).
        Otherwise spawn a background thread and return immediately.
        """
        # Use H3InferenceServer as the proxy, but inject the ComfyUI generator
        proxy_stub = _ComfyUIProxyStub(self)
        srv = H3InferenceServer(
            port=self.h3_port,
            width=64, height=64, n_frames=1,
            host=self.h3_host,
        )
        srv._stub = proxy_stub  # bypass constructor stub

        if blocking:
            srv.start(blocking=True)
        else:
            srv.start(blocking=False)
        self._proxy_server = srv
        self._running = True

    def stop(self):
        """Stop the proxy server and terminate ComfyUI."""
        self.stop_comfyui()
        if self._proxy_server is not None:
            self._proxy_server.stop()
            self._proxy_server = None
        self._running = False

    def generate(self, prompt: str, seed: int | None = None,
                 n_frames: int = 1, width: int = 64, height: int = 64,
                 steps: int = 4) -> dict:
        """Call generate on the active backend (ComfyUI or stub)."""
        if self._proc is not None and self._comfyui_ready():
            return self._generate_via_comfyui(
                prompt, seed, n_frames, width, height, steps)
        return self.stub.generate(prompt, seed, n_frames, width, height)

    def health(self) -> dict:
        """Return health of both the proxy server and ComfyUI."""
        comfyui_ok = self._comfyui_ready() if self._proc else False
        proxy_ok = (self._proxy_server.health().get("status") == "ok"
                   if self._proxy_server else False)
        return {
            "status": "ok" if proxy_ok else "degraded",
            "comfyui": "running" if comfyui_ok else "stopped",
            "proxy": "ok" if proxy_ok else "error",
            "comfyui_url": self._comfyui_url,
            "h3_url": self._h3_url,
        }

    def __repr__(self):
        return (f"ComfyUIH3Bridge(comfyui={self._comfyui_url} "
                f"h3={self._h3_url} running={self._running})")


class _ComfyUIProxyStub:
    """A stub-compatible object that proxies /generate to ComfyUIH3Bridge."""

    def __init__(self, bridge: "ComfyUIH3Bridge"):
        self._bridge = bridge
        self.width = 64
        self.height = 64
        self.n_frames = 1
        self._calls = 0

    def generate(self, prompt: str, seed=None, n_frames=None,
                 width=None, height=None):
        self._calls += 1
        return self._bridge.generate(
            prompt=prompt, seed=seed,
            n_frames=n_frames or self.n_frames,
            width=width or self.width,
            height=height or self.height,
        )


# -------------------------------------------------------------------------
# Global server management (for /api/video/start|stop)
# -------------------------------------------------------------------------

def _get_global_server(port: int = 8765) -> Optional[H3InferenceServer]:
    return _REGISTRY_PORT.get(port)


def _start_server(port: int = 8765, width: int = 64, height: int = 64,
                  n_frames: int = 1) -> dict:
    """Start (or return) the global H3InferenceServer on the given port."""
    existing = _REGISTRY_PORT.get(port)
    if existing is not None and existing.is_running:
        return {"status": "already_running", "port": port, "url": existing.url}

    srv = H3InferenceServer(port=port, width=width, height=height, n_frames=n_frames)
    try:
        srv.start(blocking=False)
        return {"status": "started", "port": port, "url": srv.url}
    except OSError as e:
        if e.errno == 98 or e.errno == 48 or "Address already in use" in str(e):
            return {"status": "already_running", "port": port,
                    "url": f"http://localhost:{port}"}
        raise


def _stop_server(port: int = 8765) -> dict:
    """Stop the global H3InferenceServer on the given port."""
    srv = _REGISTRY_PORT.get(port)
    if srv is None:
        return {"status": "not_running", "port": port}
    srv.stop()
    return {"status": "stopped", "port": port}
