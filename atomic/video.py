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
