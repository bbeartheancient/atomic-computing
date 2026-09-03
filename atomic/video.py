"""H3 video generation (iter 27): the GPU1 service bridge.

H3 + FastVideo live in ComfyUI space; the harness treats them as an
EXTERNAL GPU1 service. The atomic side only sees
`frames: List[bytes]` per tick (one RGBA frame per tick). The bridge
contract is intentionally minimal so a stub or a real H3 backend
plugs in interchangeably:

  1) HTTP bridge (default): POST /generate {prompt, seed, width, height,
     steps, n_frames, ...} -> JSON {frames_b64: [..], h3_latency_ms, prompt}
     The harness never imports torch / diffusers / comfy -- it shells
     out via urllib to a running H3 server, just like a worker.
  2) local stub (no H3): `H3Stub(width=64, height=64, n_frames=8)`,
     synthesizes deterministic RGBA frames from a seeded RNG over
     the prompt embedding. Lets the harness prove the wire contract
     end-to-end without GPU.
  3) file source: a directory of pre-rendered .rgba files, served in
     alphabetical order (the "replay an old generation" path).

This module is the operator's "middle" plumbing: the *real* H3 model
is GPU1 (H3 FastVideo on a B70, separate from the duty VLLM on GPU0);
the harness drives both via a per-tick tick() that pushes fresh frames
into the conscious engine through a HostBridge.

Public API:
  H3Client   HTTP wrapper. `generate(prompt) -> {frames, prompt, latency_ms}`.
  H3Stub     Local deterministic generator (no GPU).
  H3File     File source (replay a previous generation).
  H3Source   Factory: {http|file|stub}.

  H3Session  The "infinite slop" loop: a tick-indexed queue of prompts
             (the swarm prompt bank), one H3 call per tick, frames
             pushed into a HostBridge. The renderer reads them via a
             viz_video sink.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import struct
import time
import urllib.error
import urllib.request

__all__ = [
    "H3Client", "H3Stub", "H3File", "H3Source",
    "H3Session", "H3Frame",
    "PROMPT_BANK_DEFAULT",
    "InfiniteVideoLoop",
    # FastH3 (iter 36)
    "FastH3Client", "FastH3Stub", "FastH3Source",
    "FastH3Error", "FastH3Frame",
    "FastH3Session",
    "FASTH3_PROMPT_BANK_DEFAULT", "FASTH3_DEFAULT_STEPS",
    "FASTH3_DEFAULT_VSA", "FASTH3_DEFAULT_VSA_KEEP",
    "FASTH3_DEFAULT_QUANT",
    "InfiniteFastH3Loop",
]


PROMPT_BANK_DEFAULT = (
    "a slow orbit around a frozen comet",
    "neon city under heavy rain, far future",
    "macro shot of a beetle on wet bark",
    "two black cats sleeping in a beam of warm light",
    "saturn-rings dissolve into ribbons of light",
    "an ancient library where the books are glowing",
    "a single leaf falling through fog",
    "a car driving down an empty desert highway at dawn",
    "fish-eye view of a crowded subway station",
    "a candle flame fighting against a draft",
    "pan across a server farm, blue indicator lights",
    "a wooden boat on still water at golden hour",
    "a slow zoom into the pupil of a human eye",
    "a child chasing soap bubbles in a park",
    "a cyberpunk alley with steam and reflected signs",
    "a herd of antelope crossing a dry riverbed",
    "a city rooftop garden with bees and flowers",
    "an old radio with vacuum tubes glowing orange",
    "rain on a window with city lights behind it",
    "a single paper airplane in a vast white space",
)


class H3Frame:
    """One decoded RGBA frame, with provenance metadata."""

    __slots__ = ("rgba", "prompt", "seed", "t", "h3_latency_ms")

    def __init__(self, rgba, prompt="", seed=0, t=0, h3_latency_ms=0):
        self.rgba = bytes(rgba) if rgba is not None else b""
        self.prompt = str(prompt)
        self.seed = int(seed)
        self.t = int(t)
        self.h3_latency_ms = float(h3_latency_ms)

    @property
    def size_bytes(self):
        return len(self.rgba)

    @property
    def sha256(self):
        return hashlib.sha256(self.rgba).hexdigest()

    def __repr__(self):
        return "H3Frame(t=%d seed=%d %dB '%s')" % (
            self.t, self.seed, self.size_bytes, self.prompt[:32])


# -- the local stub: deterministic, GPU-free --------------------------------

class H3Stub:
    """Deterministic local generator (no GPU). Renders RGBA frames whose
    pixels are a function of (prompt, seed, frame_index). Two callers with
    the same prompt + seed get the same frames (the 'infinite slop' is
    repeatable; the swarm bank is queryable)."""

    def __init__(self, width=64, height=64, n_frames=8, latency_ms=0.0):
        self.width = int(width)
        self.height = int(height)
        self.n_frames = int(n_frames)
        self.latency_ms = float(latency_ms)
        self._calls = 0
        self._frames_emitted = 0
        self._last_prompt = ""

    def _seed_for(self, prompt, seed):
        if seed is None:
            seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        return int(seed)

    def _render(self, prompt, seed, frame_index, width=None, height=None):
        w = int(width) if width else self.width
        h = int(height) if height else self.height
        # Python 3.14 random.Random() rejects tuples; combine via xor
        combined = (int(seed) * 1_000_003) ^ (int(frame_index) * 2654435761) & 0xFFFFFFFF
        rng = random.Random(combined)
        n = w * h
        # emit raw RGBA bytes (4 channels) -- a strong gradient + noise
        # mix that visibly varies per frame_index and prompt seed.
        import math as _m
        prompt_bytes = (prompt or "").encode("utf-8")
        prompt_hash = int(hashlib.sha256(prompt_bytes).hexdigest()[:16], 16)
        r0 = (prompt_hash >> 8) & 0xFF
        g0 = (prompt_hash >> 16) & 0xFF
        b0 = (prompt_hash >> 24) & 0xFF
        a0 = 255
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
            a = a0
            j = i * 4
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            out[j + 3] = a
        return bytes(out)

    def generate(self, prompt, seed=None, n_frames=None,
                 width=None, height=None):
        t0 = time.perf_counter()
        seed = self._seed_for(prompt, seed)
        n = int(n_frames) if n_frames else self.n_frames
        frames = [self._render(prompt, seed, k, width, height) for k in range(n)]
        elapsed = (time.perf_counter() - t0) * 1000.0 + self.latency_ms
        self._calls += 1
        self._frames_emitted += n
        self._last_prompt = prompt
        return {
            "prompt": prompt, "seed": seed, "frames": frames,
            "h3_latency_ms": elapsed, "width": self.width, "height": self.height,
        }


# -- the HTTP bridge: H3 FastVideo on GPU1 (B70) ----------------------------

class H3Client:
    """HTTP wrapper for the H3 FastVideo server (GPU1).

    `endpoint` is the base URL (e.g. 'http://gpu1.local:8765'). The
    server is expected to expose:
        POST /generate  body={"prompt":..., "seed":..., "n_frames":...,
                              "width":..., "height":..., "steps":...}
        response JSON {"frames_b64": [..], "prompt", "seed",
                       "h3_latency_ms", "width", "height"}

    Failure handling: a 5xx / connection error raises H3Error; a
    caller's tick handler can choose to fall back to H3Stub (the
    'degraded but live' mode)."""

    def __init__(self, endpoint="http://localhost:8765", timeout_s=60.0,
                 fallback=None):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.fallback = fallback  # optional H3Stub for degraded mode
        self._calls = 0
        self._errors = 0
        self._last_prompt = ""

    def _post(self, path, body):
        url = self.endpoint + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            self._errors += 1
            raise H3Error("H3 %s: %s" % (path, e)) from e

    def generate(self, prompt, seed=None, n_frames=8,
                 width=64, height=64, steps=4):
        body = {"prompt": prompt, "n_frames": int(n_frames),
                "width": int(width), "height": int(height),
                "steps": int(steps)}
        if seed is not None:
            body["seed"] = int(seed)
        t0 = time.perf_counter()
        try:
            r = self._post("/generate", body)
        except H3Error:
            if self.fallback is not None:
                r = self.fallback.generate(prompt, seed=seed, n_frames=n_frames,
                                           width=width, height=height)
            else:
                raise
        else:
            r["frames"] = [base64.b64decode(b) for b in r.get("frames_b64", [])]
        r["h3_latency_ms"] = float(r.get("h3_latency_ms", 0.0))
        if r["h3_latency_ms"] == 0.0:
            r["h3_latency_ms"] = (time.perf_counter() - t0) * 1000.0
        if "seed" not in r:
            r["seed"] = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        self._calls += 1
        self._last_prompt = prompt
        return r


class H3Error(RuntimeError):
    pass


# -- file source: replay a previous generation -----------------------------

class H3File:
    """Reads pre-rendered RGBA frames from a directory (alphabetical order).
    Each file's raw bytes are one frame. Used for replay / archival."""

    def __init__(self, path):
        self.path = str(path)
        if not os.path.isdir(self.path):
            raise H3Error("H3File: not a directory: %s" % self.path)
        self._files = sorted(f for f in os.listdir(self.path)
                             if f.endswith(".rgba") or f.endswith(".bin"))
        if not self._files:
            raise H3Error("H3File: no .rgba files in %s" % self.path)
        self._idx = 0
        self._calls = 0
        self._last_prompt = ""

    def generate(self, prompt="", seed=None, n_frames=None, width=None, height=None):
        n = int(n_frames) if n_frames else len(self._files)
        frames = []
        for _ in range(n):
            fn = self._files[self._idx % len(self._files)]
            self._idx += 1
            with open(os.path.join(self.path, fn), "rb") as f:
                frames.append(f.read())
        self._calls += 1
        self._last_prompt = prompt
        return {"prompt": prompt, "seed": int(seed or 0), "frames": frames,
                "h3_latency_ms": 0.0, "width": 0, "height": 0}


# -- factory ---------------------------------------------------------------

def H3Source(kind="stub", **kw):
    """Return an H3 generator by name (iter 27, factory)."""
    if kind == "stub":
        return H3Stub(**kw)
    if kind == "http":
        return H3Client(**kw)
    if kind == "file":
        return H3File(**kw)
    raise H3Error("H3Source: unknown kind: %s" % kind)


# -- the "infinite slop" session: per-tick H3 -> HostBridge -> viz_video ---

class H3Session:
    """The infinite-livestream loop: one tick = one H3 frame.

    On each tick, the session:
      1) consumes a prompt from a bank (or a Swarm's H4-consensus pick);
      2) calls h3.generate(prompt) and pops one frame;
      3) pushes the frame into the host bridge (subconscious -> conscious);
      4) records a H3Frame in its per-tick ring (the dma_trace view).
    The conscious engine sees the frame on its bus `bridge.frame` key
    after `bridge_latency` ticks; a viz_video sink reads it and exposes
    it to the UI renderer."""

    def __init__(self, h3, prompts=None, frames_per_prompt=8,
                 bridge=None, max_ticks=10_000, start_seed=0,
                 prompt_consensus=None):
        self.h3 = h3
        self.prompts = list(prompts) if prompts is not None else list(PROMPT_BANK_DEFAULT)
        self.frames_per_prompt = int(frames_per_prompt)
        self.bridge = bridge  # optional HostBridge (created by caller)
        self.max_ticks = int(max_ticks)
        self._t = 0
        self._seed = int(start_seed)
        # a queue of fresh frames still to be pushed
        self._pending = []  # list of H3Frame
        # per-tick ring of consumed H3Frames (the dma_trace view)
        self._ring = []  # FIFO bounded by max_ticks
        # consumed-prompt ring (the swarm can see what the bank emitted)
        self._consumed_prompts = []
        # optional Swarm-style consensus hook: a callable
        # (prev_prompts) -> next_prompt. Used to let the H4 consensus
        # of a swarm's scalar outputs route the prompt bank.
        self.prompt_consensus = prompt_consensus

    @property
    def t(self):
        return self._t

    def _next_prompt(self):
        # 1) swarm consensus pick (if registered)
        if self.prompt_consensus is not None and self._consumed_prompts:
            return self.prompt_consensus(self._consumed_prompts)
        # 2) round-robin over the bank
        if not self.prompts:
            return "loop"
        idx = (self._t // self.frames_per_prompt) % len(self.prompts)
        return self.prompts[idx]

    def tick(self):
        if self._t >= self.max_ticks:
            return None
        # ensure we have a fresh batch queued for the current prompt
        if not self._pending:
            prompt = self._next_prompt()
            r = self.h3.generate(prompt, seed=self._seed)
            self._seed += 1
            self._consumed_prompts.append(prompt)
            if len(self._consumed_prompts) > 64:
                self._consumed_prompts.pop(0)
            for i, rgba in enumerate(r["frames"]):
                self._pending.append(H3Frame(
                    rgba=rgba, prompt=prompt, seed=r.get("seed", 0),
                    t=self._t, h3_latency_ms=r.get("h3_latency_ms", 0.0)))
        # pop one frame
        frame = self._pending.pop(0)
        frame.t = self._t
        # push into the bridge (sub -> host -> con)
        if self.bridge is not None:
            self.bridge.push(self._t, {"frame": frame.rgba,
                                       "prompt": frame.prompt,
                                       "seed": float(frame.seed)})
        # record in the per-tick ring (the dma_trace / QBF view)
        self._ring.append(frame)
        if len(self._ring) > self.max_ticks:
            self._ring.pop(0)
        self._t += 1
        return frame

    def run(self, ticks):
        for _ in range(int(ticks)):
            self.tick()
        return self._ring

    @property
    def consumed_prompts(self):
        return list(self._consumed_prompts)

    def latest(self):
        return self._ring[-1] if self._ring else None

    def frames_for_prompt(self, prompt):
        return [f for f in self._ring if f.prompt == prompt]

    def stats(self):
        h3 = self.h3
        return {
            "t": self._t,
            "consumed_prompts": len(self._consumed_prompts),
            "frames_in_ring": len(self._ring),
            "h3_calls": getattr(h3, "_calls", None),
            "h3_frames_emitted": getattr(h3, "_frames_emitted", None),
            "h3_last_prompt": getattr(h3, "_last_prompt", ""),
            "bridge_depth": self.bridge.depth() if self.bridge else 0,
        }

    def __repr__(self):
        return "H3Session(t=%d consumed=%d ring=%d)" % (
            self._t, len(self._consumed_prompts), len(self._ring))


# -- the "infinite slop" loop: H3Session -> HostBridge -> BicameralViewer -----
# iter 34: this is the end-to-end pipeline the operator's queue
# describes as "infinite livestream" or "interdimensional cable":
#
#   PROMPT_BANK -> H3Stub / H3Client (per-tick frames)
#                -> HostBridge.push_frame (subconscious -> conscious)
#                -> BicameralViewer.feed_video_tick (injects into viz_video)
#                -> Tile wall renders the decoded frame
#                -> FlowTrace.record_video_frame (portable via QBF)
#
# The InfiniteVideoLoop class wires all of the above into a single
# object so an operator can run it directly without hand-wiring. It
# is a single-tick loop: each call to step() drives one H3 frame
# through the bridge, the bicameral pipeline's conscious engine
# renders it via viz_video, and the trace observer records the frame
# metadata. The trace is QBF-portable: every recorded frame can be
# replayed by a future H3 session bit-exact.

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .trace import FlowTrace


class InfiniteVideoLoop:
    """End-to-end pipeline: H3 session -> HostBridge -> BicameralViewer.

    The operator-facing "infinite slop" loop. On every step() call:
      1) H3Session.tick() generates a new RGBA frame.
      2) HostBridge.push_frame() enqueues it for the conscious engine.
      3) The bicameral pipeline ticks; the conscious engine sees the
         frame on bus[<con_module_id>.frame] after `bridge_latency` ticks.
      4) The viz_video atom decodes the frame via the H(4) gate.
      5) If a FlowTrace observer is attached, the frame is recorded
         as a VideoFrameEntry (portable via QBF).
      6) stats() returns the live counters.

    Args:
        h3: an H3Stub / H3Client / H3File instance.
        viewer: a BicameralViewer (the conscious engine; its con_program
                must contain a viz_video block at module_id).
        bridge_map: list of (sub_key, con_key) pairs for scalar bridge
                    traffic. Frame traffic is auto-wired.
        bridge_latency: tick latency (default 1, matches the wire latch).
        prompts: optional prompt bank (default PROMPT_BANK_DEFAULT).
        module_id: the viz_video block id in the con_program (default "vv").
        max_ticks: safety cap on the loop.
        trace: optional FlowTrace observer.
    """

    def __init__(self, h3, viewer, bridge_map=None, bridge_latency=1,
                 prompts=None, module_id="vv", max_ticks=10_000,
                 trace: Optional["FlowTrace"] = None):
        self.h3 = h3
        self.viewer = viewer
        self.bridge_map = list(bridge_map or [])
        self.bridge_latency = int(bridge_latency)
        self.module_id = str(module_id)
        self.max_ticks = int(max_ticks)
        self._t = 0
        self._frame_index = 0
        # Reuse the viewer's bridge so bridge depths / stats stay in sync.
        self.bridge = viewer.bridge
        # The H3Session drives the prompts + h3 backend.
        self.session = H3Session(
            h3, prompts=prompts, frames_per_prompt=1,
            bridge=self.bridge, max_ticks=max_ticks,
            start_seed=0,
        )
        self.trace = trace

    @property
    def t(self):
        return self._t

    @property
    def frame_index(self):
        return self._frame_index

    def _next_prompt(self):
        # The InfiniteVideoLoop drives its own prompt index. Each tick
        # consumes one prompt from the bank (round-robin); the swarm
        # consensus hook (if set) can override the pick via
        # session.prompt_consensus.
        if (self.session.prompt_consensus is not None
                and self.session._consumed_prompts):
            return self.session.prompt_consensus(
                self.session._consumed_prompts)
        if not self.session.prompts:
            return "loop"
        idx = self._t % len(self.session.prompts)
        return self.session.prompts[idx]

    def step(self):
        """One tick: H3 -> bridge -> bicameral pipeline -> viz_video -> trace.

        Returns the H3Frame that was rendered, or None if the loop is
        exhausted.
        """
        if self._t >= self.max_ticks:
            return None
        t = self._t
        # 1) Generate one frame per tick from the prompt bank. We use
        #    the H3Session's own prompt pick so swarm consensus hooks
        #    work the same way they do in standalone H3Session use.
        #    Each tick advances the prompt (round-robin or consensus).
        prompt = self._next_prompt()
        r = self.h3.generate(prompt, seed=self.session._seed,
                             n_frames=1)
        self.session._seed += 1
        self.session._consumed_prompts.append(prompt)
        if len(self.session._consumed_prompts) > 64:
            self.session._consumed_prompts.pop(0)
        rgba = r["frames"][0]
        frame = H3Frame(
            rgba=rgba, prompt=prompt, seed=r.get("seed", 0),
            t=t, h3_latency_ms=r.get("h3_latency_ms", 0.0))
        # 2) push into the dedicated frame path (subconscious -> conscious)
        self.bridge.push_frame(
            t, frame.rgba,
            prompt=frame.prompt, seed=frame.seed,
            module_id=self.module_id,
            width=0, height=0,
        )
        # 3) record in session ring (kept for parity with H3Session)
        self.session._ring.append(frame)
        if len(self.session._ring) > self.max_ticks:
            self.session._ring.pop(0)
        # 4) drive the conscious engine: pop any frames whose arrival
        #    is <= current tick, write the LATEST into bus, then tick.
        # The bridge arrival is push_tick + latency, so pop with
        # t+latency to drain everything we pushed.
        frames_ready = self.bridge.pop_frame(t + self.bridge_latency)
        latest = frames_ready[-1] if frames_ready else None
        if latest:
            rgba_bytes = latest.get("rgba", b"")
            if rgba_bytes:
                # Inject into the conscious engine's bus
                pipe = self.viewer.pipeline
                key = self.module_id + ".frame"
                pipe.con.bus.set(key, bytes(rgba_bytes))
        # 5) tick the bicameral pipeline (con sees the frame on its bus)
        self.viewer.pipeline.tick()
        self.viewer._tick_count = self.viewer.pipeline._t
        self.viewer._depth_history.append(self.bridge.depth())
        if len(self.viewer._depth_history) > 512:
            self.viewer._depth_history.pop(0)
        # 6) record the frame in the trace (portable)
        if self.trace is not None:
            w_g = latest.get("_w", 0.0) if latest else 0.0
            x_g = latest.get("_x", 0.0) if latest else 0.0
            y_g = latest.get("_y", 0.0) if latest else 0.0
            z_g = latest.get("_z", 0.0) if latest else 0.0
            self.trace.record_video_frame(
                rgba=frame.rgba, t=t,
                prompt=frame.prompt, seed=frame.seed,
                h3_latency_ms=frame.h3_latency_ms,
                width=0, height=0,
                w_gate=w_g, x_gate=x_g, y_gate=y_g, z_gate=z_g,
            )
            # also record a node frame so the trace is a valid .qbf payload
            # (append_run refuses empty-trace archives; the video entries
            # already carry the canonical frame payload, so the node
            # frame is just a small scalar indicator of bytes-rendered).
            self.trace.record_node(
                module_id=self.module_id, module_type="viz_video",
                in_ports={"frame_bytes": len(frame.rgba)},
                out_ports={"rgba_decoded": len(frame.rgba)},
                latency_us=int(frame.h3_latency_ms * 1000),
                t=t, tag="infinite_video_loop",
            )
        self._t += 1
        self._frame_index += 1
        return frame

    def run(self, ticks):
        """Run `ticks` steps. Returns the list of H3Frame objects rendered."""
        out = []
        for _ in range(int(ticks)):
            f = self.step()
            if f is None:
                break
            out.append(f)
        return out

    def stats(self):
        return {
            "t": self._t,
            "frame_index": self._frame_index,
            "session_t": self.session._t,
            "frames_in_ring": len(self.session._ring),
            "consumed_prompts": len(self.session._consumed_prompts),
            "h3_calls": getattr(self.h3, "_calls", None),
            "bridge_depth": self.bridge.depth(),
            "bridge_pushed": self.bridge._pushed,
            "bridge_popped": self.bridge._popped,
            "trace_video_seq": self.trace.video_seq if self.trace is not None else None,
            "viewer_tick": self.viewer._tick_count,
        }

    def __repr__(self):
        return ("InfiniteVideoLoop(t=%d frames=%d bridge_depth=%d)"
                % (self._t, self._frame_index, self.bridge.depth()))


# =============================================================================
# iter 36 — FastH3 (FastVideo GGUF, 4-step VSA) checkpoint
# =============================================================================
#
# Iter 36 deliberately does NOT build a ComfyUI-VSA-on-XPU integration.
# The model card lists three unmerged pieces, all Windows + CUDA + VS 2022:
#   1. ComfyUI on kijai's `vsa` branch (H3 model code that keeps gate weights)
#   2. comfy-kitchen's `sol_attn` branch, compiled — CUDA sparse-attention
#   3. A test node from comfy-kitchen PR #117 that turns VSA on at runtime
# Plus 2 VAEs + the qwen3vl-32B text encoder. The vendored comfy_kitchen here
# (in OlympusServer/h3-pyoverrides) is 0.2.26 with cuda/hip/triton/eager
# backends; there is NO Intel/SYCL sol_attn equivalent.
#
# Status check (this turn):
#   - /home/bbear/models/diffusion_models/FastH3-comfy-Q5_K_M.gguf: 14 GB on
#     disk (target 25 GB) — transfer still in progress
#   - /home/bbear/models/diffusion_models/FastH3-comfy-Q4_K_M.gguf: 9.2 GB on
#     disk (target 21 GB) — transfer still in progress
#   - minimax_h3_video_vae_fp16.safetensors + minimax_h3_audio_vae_fp32.safetensors
#     already in /home/bbear/models/vae/
#   - Text encoder adapter: qwen3vl-32B-MiniMax-H3-Q2_K.gguf (8.5 GB)
#     in /home/bbear/models/text_encoders/ — too aggressive a quant for 4-step
#     VSA (the Q2 K-tier can corrupt the timestep embedder the model card
#     warns about; an fp8 / Q4 tier would be safer when the model card
#     recommends "kept at high precision through quantization").
#   - ComfyUI: master (v0.33.1); comfyui-h3: master (v0.33.1) — neither has
#     kijai/vsa checked out, neither has the sol_attn node installed.
#
# What this iter DOES build:
#   * FastH3Client / FastH3Stub — wire contract identical in shape to H3
#     (HTTP /generate -> {frames_b64, h3_latency_ms, prompt, seed}) but
#     tagged with the FastH3-specific knobs (steps=4, vsa=on, quant, vae
#     selection). The bridge DOES NOT CARE whether the back-end is ComfyUI
#     running kijai/vsa, llama.cpp with the GGUF, or a deterministic stub.
#   * FastH3Bridge — same host-RAM wire as HostBridge.push_frame; one extra
#     metadata tag so QBF traces can distinguish FastH3 frames from base H3.
#   * FastH3Session / InfiniteFastH3Loop — the "infinite slop" loop shape
#     again, parameterized for FastH3 so an operator can swap from base H3
#     to FastH3 by changing one constructor call.
#   * Tests proving the wire contract: stub -> bridge -> viz_fasth3_video ->
#     QBF round-trip is bit-exact and replay-safe.
#
# The fast-path the operator wants ("actual frames") requires:
#   a) wait for the Q5_K_M / Q4_K_M GUF to finish downloading
#   b) checkout kijai/vsa in /home/bbear/services/comfyui or comfyui-h3
#   c) get a higher-precision text encoder (Q4_K_M or bf16 of the 32B TE
#      — the current Q2_K is below the card's recommendation)
#   d) build comfy-kitchen sol_attn for the chosen CUDA arch — and then
#      patch a SYCL port for the B70 (separate effort; the existing
#      comfy_kitchen 0.2.26 has no XPU backend at all).
# These steps are NOT in iter 36; the harness-side contract is.
# =============================================================================


# -- iter 36: FastH3 wire contract constants ---------------------------------

FASTH3_PROMPT_BANK_DEFAULT = PROMPT_BANK_DEFAULT  # same bank, same shape
FASTH3_DEFAULT_STEPS = 4       # 4-step distilled (guidance baked in; CFG 1.0)
FASTH3_DEFAULT_VSA = True      # sparse attention ON — required by these GUF
FASTH3_DEFAULT_VSA_KEEP = 10   # 10% kept = 90% sparsity (card-recommended)
FASTH3_DEFAULT_QUANT = "Q5_K_M"  # 25 GB target on disk; Q4_K_M is 21 GB


# -- iter 36: FastH3Client (HTTP bridge) -----------------------------------

class FastH3Error(RuntimeError):
    pass


class FastH3Stub:
    """Deterministic local stub for FastH3 (no GPU, no real model).

    Same wire shape as H3Stub: prompt -> List[bytes] (one RGBA frame
    per tick). FastH3-specific: stamps every call with steps=4 +
    vsa=ON + the quant tier so a downstream consumer can verify
    the contract independently. Two callers with the same prompt+seed
    see the same frames (so QBF traces are replay-safe).
    """

    def __init__(self, width=64, height=64, n_frames=8, latency_ms=0.0,
                 steps=FASTH3_DEFAULT_STEPS, vsa=FASTH3_DEFAULT_VSA,
                 vsa_keep=FASTH3_DEFAULT_VSA_KEEP,
                 quant=FASTH3_DEFAULT_QUANT):
        self.width = int(width)
        self.height = int(height)
        self.n_frames = int(n_frames)
        self.latency_ms = float(latency_ms)
        self.steps = int(steps)
        self.vsa = bool(vsa)
        self.vsa_keep = int(vsa_keep)
        self.quant = str(quant)
        self._calls = 0
        self._frames_emitted = 0
        self._last_prompt = ""

    def _seed_for(self, prompt, seed):
        if seed is None:
            seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        return int(seed)

    def _render(self, prompt, seed, frame_index, width=None, height=None):
        # Identical visual contract to H3Stub so existing viz_video / UI
        # renderers keep working unchanged. The FastH3-specific metadata
        # is on the H3Frame, not in the pixels.
        w = int(width) if width else self.width
        h = int(height) if height else self.height
        combined = ((int(seed) * 1_000_003) ^ (int(frame_index) * 2654435761)) & 0xFFFFFFFF
        rng = random.Random(combined)
        n = w * h
        import math as _m
        prompt_bytes = (prompt or "").encode("utf-8")
        prompt_hash = int(hashlib.sha256(prompt_bytes).hexdigest()[:16], 16)
        r0 = (prompt_hash >> 8) & 0xFF
        g0 = (prompt_hash >> 16) & 0xFF
        b0 = (prompt_hash >> 24) & 0xFF
        a0 = 255
        out = bytearray(n * 4)
        for i in range(n):
            x = i % w
            y = i // w
            cx, cy = w / 2.0, h / 2.0
            d = _m.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            # FastH3 trains with a steeper sparsity: bump the wave freq
            # slightly so stub output is visually distinct from H3Stub.
            wave = _m.sin((d / max(1.0, w)) * 8.2467 - frame_index * 0.85)
            r = max(0, min(255, int(r0 + 70 * wave + rng.randint(-25, 25))))
            g = max(0, min(255, int(g0 + 50 * (-wave) + rng.randint(-20, 20))))
            b = max(0, min(255, int(b0 + 60 * wave + rng.randint(-20, 20))))
            a = a0
            j = i * 4
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            out[j + 3] = a
        return bytes(out)

    def generate(self, prompt, seed=None, n_frames=None,
                 width=None, height=None, steps=None, vsa=None):
        t0 = time.perf_counter()
        seed = self._seed_for(prompt, seed)
        n = int(n_frames) if n_frames else self.n_frames
        frames = [self._render(prompt, seed, k, width, height) for k in range(n)]
        elapsed = (time.perf_counter() - t0) * 1000.0 + self.latency_ms
        self._calls += 1
        self._frames_emitted += n
        self._last_prompt = prompt
        return {
            "prompt": prompt, "seed": seed, "frames": frames,
            "h3_latency_ms": elapsed,
            "width": int(width or self.width),
            "height": int(height or self.height),
            # FastH3-specific: 4-step + VSA contract
            "steps": int(steps if steps is not None else self.steps),
            "vsa": bool(vsa if vsa is not None else self.vsa),
            "vsa_keep": int(self.vsa_keep),
            "quant": self.quant,
            "model": "FastH3",
        }


class FastH3Client:
    """HTTP wrapper for a FastH3 server (the kijai/vsa ComfyUI backend
    on GPU1, or a llama.cpp GGUF backend, or any conformant server).

    Two dispatch modes (iter 38):
        mode="comfyui"  (default): POST /prompt with the minimal FastH3
          vsa workflow JSON, poll /history until done, decode the first
          RGBA frame.  Falls back to FastH3Stub when the server is
          unreachable.
        mode="generic"  (iter-36 contract): POST /generate and expect
          the server to own the generation pipeline.  Falls back to
          FastH3Stub when unreachable.

    Falls back to FastH3Stub when the server is unreachable (the
    'degraded but live' mode). The fallback's contract is bit-identical,
    so a QBF trace recorded against the stub replays identically against
    the real server (modulo the actual pixel bytes).
    """

    def __init__(self, endpoint="http://localhost:8188", timeout_s=60.0,
                 fallback=None,
                 steps=FASTH3_DEFAULT_STEPS, vsa=FASTH3_DEFAULT_VSA,
                 vsa_keep=FASTH3_DEFAULT_VSA_KEEP,
                 quant=FASTH3_DEFAULT_QUANT,
                 mode="generic",
                 te_adapter=None,   # iter 39: TEAdapterClient or None
                 te_adapter_auto=True):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.fallback = fallback  # optional FastH3Stub
        self.steps = int(steps)
        self.vsa = bool(vsa)
        self.vsa_keep = int(vsa_keep)
        self.quant = str(quant)
        self.mode = str(mode)
        self._calls = 0
        self._errors = 0
        self._last_prompt = ""
        # iter 39: when te_adapter is set, every generate() call first
        # encodes the prompt via TEAdapterClient (VLLM Qwen 3.8 27B ->
        # te_h3_from_38 MLP) and stashes a .pt in cond_dir. The workflow
        # then uses LoadH3TE38Conditioning instead of H3TextEncode8. If
        # te_adapter_auto is True (the default) and the encode fails or
        # the vllm is down, we fall back to the iter-38 wire (H3TextEncode8).
        self.te_adapter = te_adapter
        self.te_adapter_auto = bool(te_adapter_auto)
        self._te38_uses = 0
        self._te38_fallbacks = 0

    def _post(self, path, body):
        url = self.endpoint + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            self._errors += 1
            raise FastH3Error("FastH3 %s: %s" % (path, e)) from e

    def _generate_comfyui(self, prompt, seed, n_frames, width, height,
                          steps, vsa):
        """iter 38: dispatch through atomic.fasth3_server (real ComfyUI
        vsa /prompt + /history path).

        iter 39: if `te_adapter` is set, encode the prompt via the
        27B Qwen 3.8 -> 5120-d path and pass the .pt path into the
        workflow as LoadH3TE38Conditioning.  Falls back to the
        iter-38 H3TextEncode8 (4b h3student) wire when the encode
        fails or the vllm is down.
        """
        from atomic.fasth3_server import (
            fasth3_workflow, submit_prompt, poll_history,
            decode_first_frame, is_comfyui_up,
        )
        # The endpoint base is e.g. http://localhost:8188; the helpers
        # default to 127.0.0.1:8188 so we re-parse for non-default.
        host = "127.0.0.1"
        port = 8188
        try:
            from urllib.parse import urlparse
            u = urlparse(self.endpoint)
            if u.hostname:
                host = u.hostname
            if u.port is not None:
                try:
                    port = int(u.port)
                except (ValueError, OverflowError):
                    pass
        except Exception:
            pass
        if not is_comfyui_up(host=host, port=port, timeout=1.0):
            raise FastH3Error("ComfyUI vsa not reachable at %s:%d" % (host, port))
        # iter 39: try the TE-38 path first; fall back to the
        # iter-38 H3TextEncode8 wire on any failure.
        te38_cond_path = None
        if self.te_adapter is not None and self.te_adapter_auto:
            try:
                rec = self.te_adapter.encode_cached(prompt)
                te38_cond_path = rec.get("path")
                if te38_cond_path:
                    self._te38_uses += 1
            except Exception:
                if self.te_adapter_auto:
                    self._te38_fallbacks += 1
                    te38_cond_path = None
                else:
                    raise
        workflow = fasth3_workflow(
            prompt=prompt, seed=seed or 0, width=width, height=height,
            steps=steps, vsa=vsa, vsa_keep=self.vsa_keep,
            n_frames=int(n_frames),
            te38_cond_path=te38_cond_path)
        prompt_id = submit_prompt(workflow, host=host, port=port,
                                  timeout=10.0)
        history = poll_history(prompt_id, host=host, port=port,
                               interval=0.5,
                               timeout=max(self.timeout_s, 30.0))
        decoded = decode_first_frame(history, host=host, port=port)
        r = {
            "prompt": prompt,
            "seed": int(seed or 0),
            "frames": decoded["frames"],
            "width": int(decoded.get("width") or width),
            "height": int(decoded.get("height") or height),
            "h3_latency_ms": float(history.get("status", {}).get("execution_time", 0.0) * 1000.0),
            "steps": int(steps),
            "vsa": bool(vsa),
            "vsa_keep": int(self.vsa_keep),
            "quant": self.quant,
            "model": "FastH3",
            "comfyui_prompt_id": prompt_id,
            "view_filename": decoded.get("view_filename", ""),
            # iter 39: TE path is observable downstream
            "te38_cond_path": te38_cond_path or "",
            "te_path": "te38" if te38_cond_path else "h3student",
        }
        return r

    def generate(self, prompt, seed=None, n_frames=8,
                 width=64, height=64, steps=None, vsa=None):
        body = {"prompt": prompt, "n_frames": int(n_frames),
                "width": int(width), "height": int(height),
                "steps": int(steps if steps is not None else self.steps),
                "vsa": bool(vsa if vsa is not None else self.vsa),
                "vsa_keep": int(self.vsa_keep),
                "quant": self.quant,
                "model": "FastH3"}
        if seed is not None:
            body["seed"] = int(seed)
        t0 = time.perf_counter()
        # iter 38 dispatch: comfyui mode uses the real ComfyUI vsa
        # workflow + history path; generic mode preserves the
        # iter-36 /generate wire so the existing tests still pass.
        if self.mode == "comfyui":
            try:
                r = self._generate_comfyui(
                    prompt, seed, int(n_frames), int(width), int(height),
                    body["steps"], body["vsa"])
            except FastH3Error:
                if self.fallback is not None:
                    r = self.fallback.generate(prompt, seed=seed,
                                               n_frames=int(n_frames),
                                               width=width, height=height,
                                               steps=body["steps"],
                                               vsa=body["vsa"])
                else:
                    raise
        else:
            try:
                r = self._post("/generate", body)
            except FastH3Error:
                if self.fallback is not None:
                    r = self.fallback.generate(prompt, seed=seed,
                                               n_frames=int(n_frames),
                                               width=width, height=height,
                                               steps=body["steps"], vsa=body["vsa"])
                else:
                    raise
            else:
                r["frames"] = [base64.b64decode(b) for b in r.get("frames_b64", [])]
        r["h3_latency_ms"] = float(r.get("h3_latency_ms", 0.0))
        if r["h3_latency_ms"] == 0.0:
            r["h3_latency_ms"] = (time.perf_counter() - t0) * 1000.0
        if "seed" not in r:
            r["seed"] = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        # guarantee FastH3 fields even if the server omitted them
        r.setdefault("steps", self.steps)
        r.setdefault("vsa", self.vsa)
        r.setdefault("vsa_keep", self.vsa_keep)
        r.setdefault("quant", self.quant)
        r.setdefault("model", "FastH3")
        self._calls += 1
        self._last_prompt = prompt
        return r


def FastH3Source(kind="stub", **kw):
    """Factory: kind in {stub, http}."""
    if kind == "stub":
        return FastH3Stub(**kw)
    if kind == "http":
        return FastH3Client(**kw)
    raise FastH3Error("FastH3Source: unknown kind: %s" % kind)


# -- iter 36: FastH3Frame (H3Frame + FastH3 metadata) -----------------------

class FastH3Frame(H3Frame):
    """H3Frame + FastH3-specific provenance (steps, vsa, quant, model).

    Wire-compatible with H3Frame: every existing H3 consumer (H3Session,
    InfiniteVideoLoop, viz_video, viz_video_h3, QBF trace) sees a normal
    H3Frame with rgba + prompt + seed + h3_latency_ms. The extra fields
    are opt-in for FastH3-aware consumers (viz_fasth3_video).
    """

    __slots__ = ("steps", "vsa", "vsa_keep", "quant", "model")

    def __init__(self, rgba, prompt="", seed=0, t=0, h3_latency_ms=0,
                 steps=FASTH3_DEFAULT_STEPS, vsa=FASTH3_DEFAULT_VSA,
                 vsa_keep=FASTH3_DEFAULT_VSA_KEEP,
                 quant=FASTH3_DEFAULT_QUANT, model="FastH3"):
        super().__init__(rgba, prompt=prompt, seed=seed, t=t,
                         h3_latency_ms=h3_latency_ms)
        self.steps = int(steps)
        self.vsa = bool(vsa)
        self.vsa_keep = int(vsa_keep)
        self.quant = str(quant)
        self.model = str(model)

    def provenance(self):
        """FastH3 provenance dict: round-trippable through QBF tags."""
        return {
            "model": self.model,
            "steps": self.steps,
            "vsa": int(self.vsa),
            "vsa_keep": self.vsa_keep,
            "quant": self.quant,
            "prompt": self.prompt,
            "seed": self.seed,
            "h3_latency_ms": self.h3_latency_ms,
            "t": self.t,
            "size_bytes": self.size_bytes,
            "rgba_sha256": self.sha256,
        }

    def to_h3_frame(self):
        """Promote to a plain H3Frame (drop FastH3 metadata)."""
        return H3Frame(rgba=self.rgba, prompt=self.prompt,
                       seed=self.seed, t=self.t,
                       h3_latency_ms=self.h3_latency_ms)


# -- iter 36: FastH3Session (one-tick = one FastH3 frame) ------------------

class FastH3Session:
    """The FastH3 infinite-livestream loop: one tick = one frame.

    Same shape as H3Session, but the frame is a FastH3Frame (with
    steps/vsa/quant metadata). Tick stream:
          1) consume a prompt from a bank (or a Swarm's H4 consensus);
          2) call fasth3.generate(prompt) -> pop one FastH3Frame;
          3) push the frame into the host bridge (sub -> host -> con);
          4) record in the per-tick ring (dma_trace view).

    The conscious engine sees the frame on bus[bridge.frame] (or
    bus[<module_id>.frame] via a HostBridge.push_frame) after
    `bridge_latency` ticks; the viz_fasth3_video sink decodes + renders.
    """

    def __init__(self, fasth3, prompts=None, frames_per_prompt=8,
                 bridge=None, max_ticks=10_000, start_seed=0,
                 prompt_consensus=None):
        self.fasth3 = fasth3
        self.prompts = list(prompts) if prompts is not None else list(FASTH3_PROMPT_BANK_DEFAULT)
        self.frames_per_prompt = int(frames_per_prompt)
        self.bridge = bridge
        self.max_ticks = int(max_ticks)
        self._t = 0
        self._seed = int(start_seed)
        self._pending = []          # list of FastH3Frame
        self._ring = []             # bounded FIFO
        self._consumed_prompts = []
        self.prompt_consensus = prompt_consensus

    @property
    def t(self):
        return self._t

    def _next_prompt(self):
        if self.prompt_consensus is not None and self._consumed_prompts:
            return self.prompt_consensus(self._consumed_prompts)
        if not self.prompts:
            return "loop"
        idx = (self._t // self.frames_per_prompt) % len(self.prompts)
        return self.prompts[idx]

    def tick(self):
        if self._t >= self.max_ticks:
            return None
        if not self._pending:
            prompt = self._next_prompt()
            r = self.fasth3.generate(prompt, seed=self._seed)
            self._seed += 1
            self._consumed_prompts.append(prompt)
            if len(self._consumed_prompts) > 64:
                self._consumed_prompts.pop(0)
            for i, rgba in enumerate(r["frames"]):
                self._pending.append(FastH3Frame(
                    rgba=rgba, prompt=prompt, seed=r.get("seed", 0),
                    t=self._t, h3_latency_ms=r.get("h3_latency_ms", 0.0),
                    steps=r.get("steps", FASTH3_DEFAULT_STEPS),
                    vsa=r.get("vsa", FASTH3_DEFAULT_VSA),
                    vsa_keep=r.get("vsa_keep", FASTH3_DEFAULT_VSA_KEEP),
                    quant=r.get("quant", FASTH3_DEFAULT_QUANT),
                    model=r.get("model", "FastH3")))
        frame = self._pending.pop(0)
        frame.t = self._t
        if self.bridge is not None:
            # Use the HostBridge push_frame path so viz_video and
            # viz_fasth3_video can both decode. Provenance is carried
            # separately in self._ring + the trace.
            try:
                self.bridge.push_frame(
                    self._t, frame.rgba,
                    prompt=frame.prompt, seed=frame.seed,
                    module_id="vv", width=0, height=0)
            except Exception:
                # Some bridges may be raw HostBridge without frame path:
                # fall back to the generic push with a tagged dict.
                self.bridge.push(self._t, {
                    "frame": frame.rgba,
                    "prompt": frame.prompt,
                    "seed": float(frame.seed),
                    "model": frame.model,
                    "steps": float(frame.steps),
                    "vsa": float(frame.vsa),
                    "quant": frame.quant,
                })
        self._ring.append(frame)
        if len(self._ring) > self.max_ticks:
            self._ring.pop(0)
        self._t += 1
        return frame

    def run(self, ticks):
        for _ in range(int(ticks)):
            self.tick()
        return self._ring

    @property
    def consumed_prompts(self):
        return list(self._consumed_prompts)

    def latest(self):
        return self._ring[-1] if self._ring else None

    def frames_for_prompt(self, prompt):
        return [f for f in self._ring if f.prompt == prompt]

    def stats(self):
        return {
            "t": self._t,
            "consumed_prompts": len(self._consumed_prompts),
            "frames_in_ring": len(self._ring),
            "fasth3_calls": getattr(self.fasth3, "_calls", None),
            "fasth3_frames_emitted": getattr(self.fasth3, "_frames_emitted", None),
            "fasth3_last_prompt": getattr(self.fasth3, "_last_prompt", ""),
            "bridge_depth": self.bridge.depth() if self.bridge else 0,
        }

    def __repr__(self):
        return ("FastH3Session(t=%d consumed=%d ring=%d)"
                % (self._t, len(self._consumed_prompts), len(self._ring)))


# -- iter 36: InfiniteFastH3Loop (end-to-end pipeline) ---------------------

class InfiniteFastH3Loop:
    """End-to-end pipeline: FastH3 -> HostBridge -> viz_fasth3_video.

    Shape is identical to InfiniteVideoLoop but tagged with FastH3
    provenance so a QBF trace recorded against this loop replays
    identically against a different back-end (real ComfyUI / llama.cpp)
    when the FastH3 contract is honored.

    Each step() drives one FastH3 frame through the bridge, the
    bicameral pipeline's conscious engine renders it via viz_fasth3_video,
    and the trace observer records the FastH3Frame metadata so the
    full provenance (steps/vsa/quant/model) round-trips through QBF.
    """

    def __init__(self, fasth3, viewer, bridge_map=None, bridge_latency=1,
                 prompts=None, module_id="vfh", max_ticks=10_000,
                 trace=None):
        self.fasth3 = fasth3
        self.viewer = viewer
        self.bridge_map = list(bridge_map or [])
        self.bridge_latency = int(bridge_latency)
        self.module_id = str(module_id)
        self.max_ticks = int(max_ticks)
        self._t = 0
        self._frame_index = 0
        self.bridge = viewer.bridge
        self.session = FastH3Session(
            fasth3, prompts=prompts, frames_per_prompt=1,
            bridge=self.bridge, max_ticks=max_ticks,
            start_seed=0)
        self.trace = trace

    @property
    def t(self):
        return self._t

    @property
    def frame_index(self):
        return self._frame_index

    def _next_prompt(self):
        if (self.session.prompt_consensus is not None
                and self.session._consumed_prompts):
            return self.session.prompt_consensus(
                self.session._consumed_prompts)
        if not self.session.prompts:
            return "loop"
        idx = self._t % len(self.session.prompts)
        return self.session.prompts[idx]

    def step(self):
        if self._t >= self.max_ticks:
            return None
        t = self._t
        prompt = self._next_prompt()
        r = self.fasth3.generate(prompt, seed=self.session._seed,
                                 n_frames=1)
        self.session._seed += 1
        self.session._consumed_prompts.append(prompt)
        if len(self.session._consumed_prompts) > 64:
            self.session._consumed_prompts.pop(0)
        rgba = r["frames"][0]
        frame = FastH3Frame(
            rgba=rgba, prompt=prompt, seed=r.get("seed", 0),
            t=t, h3_latency_ms=r.get("h3_latency_ms", 0.0),
            steps=r.get("steps", FASTH3_DEFAULT_STEPS),
            vsa=r.get("vsa", FASTH3_DEFAULT_VSA),
            vsa_keep=r.get("vsa_keep", FASTH3_DEFAULT_VSA_KEEP),
            quant=r.get("quant", FASTH3_DEFAULT_QUANT),
            model=r.get("model", "FastH3"))
        # push via frame path so viz_fasth3_video can decode + render
        self.bridge.push_frame(
            t, frame.rgba,
            prompt=frame.prompt, seed=frame.seed,
            module_id=self.module_id, width=0, height=0)
        self.session._ring.append(frame)
        if len(self.session._ring) > self.max_ticks:
            self.session._ring.pop(0)
        frames_drained = self.bridge.pop_frame(t + self.bridge_latency)
        # Prefer the latest frame whose module_id matches ours; otherwise
        # the latest frame at all (the bridge mixes sub->con traffic).
        latest = None
        for cand in reversed(frames_drained):
            if cand.get("module_id", "") == self.module_id:
                latest = cand
                break
        if latest is None and frames_drained:
            latest = frames_drained[-1]
        if latest:
            rgba_bytes = latest.get("rgba", b"")
            if rgba_bytes:
                pipe = self.viewer.pipeline
                key = self.module_id + ".frame"
                pipe.con.bus.set(key, bytes(rgba_bytes))
        self.viewer.pipeline.tick()
        self.viewer._tick_count = self.viewer.pipeline._t
        self.viewer._depth_history.append(self.bridge.depth())
        if len(self.viewer._depth_history) > 512:
            self.viewer._depth_history.pop(0)
        if self.trace is not None:
            w_g = latest.get("_w", 0.0) if latest else 0.0
            x_g = latest.get("_x", 0.0) if latest else 0.0
            y_g = latest.get("_y", 0.0) if latest else 0.0
            z_g = latest.get("_z", 0.0) if latest else 0.0
            self.trace.record_video_frame(
                rgba=frame.rgba, t=t,
                prompt=frame.prompt, seed=frame.seed,
                h3_latency_ms=frame.h3_latency_ms,
                width=0, height=0,
                w_gate=w_g, x_gate=x_g, y_gate=y_g, z_gate=z_g)
            # Provenance metadata encoded in in_ports (the QBF
            # frame entry contract — record_node's in_ports dict
            # round-trips through .qbf archives):
            self.trace.record_node(
                module_id=self.module_id, module_type="viz_fasth3_video",
                in_ports={"frame_bytes": len(frame.rgba),
                          "model": frame.model,
                          "steps": frame.steps,
                          "vsa": int(frame.vsa),
                          "vsa_keep": frame.vsa_keep,
                          "quant": frame.quant},
                out_ports={"rgba_decoded": len(frame.rgba)},
                latency_us=int(frame.h3_latency_ms * 1000),
                t=t, tag="infinite_fasth3_loop")
        self._t += 1
        self._frame_index += 1
        return frame

    def run(self, ticks):
        out = []
        for _ in range(int(ticks)):
            f = self.step()
            if f is None:
                break
            out.append(f)
        return out

    def stats(self):
        return {
            "t": self._t,
            "frame_index": self._frame_index,
            "session_t": self.session._t,
            "frames_in_ring": len(self.session._ring),
            "consumed_prompts": len(self.session._consumed_prompts),
            "fasth3_calls": getattr(self.fasth3, "_calls", None),
            "bridge_depth": self.bridge.depth(),
            "bridge_pushed": self.bridge._pushed,
            "bridge_popped": self.bridge._popped,
            "trace_video_seq": self.trace.video_seq if self.trace is not None else None,
            "viewer_tick": self.viewer._tick_count,
        }

    def __repr__(self):
        return ("InfiniteFastH3Loop(t=%d frames=%d bridge_depth=%d)"
                % (self._t, self._frame_index, self.bridge.depth()))
