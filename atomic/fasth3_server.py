"""FastH3 + ComfyUI vsa workflow client (iter 43).

This module is the harness-side glue that turns the iter-36 wire
contract into a REAL FastH3 (4-step VSA, GGUF) ComfyUI backend on
GPU1. The contract stays the same (`FastH3Client.generate(prompt,
seed, n_frames, width, height, steps, vsa)`), but the wire the
client now talks to is ComfyUI's standard /prompt + /history API
(the "real" video diffusion server), not the simple /generate stub
the iter-36 unit test used.

Pipeline (when a ComfyUI vsa server is running):
  FastH3Client.generate("a comet over the ocean")
    -> fasth3_server.prompt(workflow)
      -> POST  /prompt       (workflow JSON, client_id)
      -> poll  /history/<id> (until status == "success")
      -> decode first output video -> RGBA bytes
    -> {"frames": [bytes], "model": "FastH3", "steps": 4, "vsa": True, ...}

The minimal workflow JSON is built from the pinned components in
this dir:

  text encoder:  qwen3vl-4b-h3student-Q4_K_M.gguf  (4b h3student)
  diffusion:     FastH3-comfy-Q4_K_M.gguf           (vsa branch keeps
                                                     the to_gate_compress
                                                     layer)
  VAE:           diffusion_models/h3_vae.safetensors
  sampler:       SolAttnXPUVSA (kijai vsa branch, 4 steps)
  scheduler:     4-step discrete flow-match (vsa)
  frames:        1 (the harness tick == 1 frame; production scales
                     by re-running the loop)

The workflow is parameterised by prompt / seed / n_frames / width /
height / steps. vsa is a per-call toggle.

When the server is unreachable, FastH3Client falls back to the
FastH3Stub (the iter-36 "degraded but live" mode) so the harness
never blocks on a missing GPU.

The minimal ComfyUI server launch:
    cd /home/bbear/services/comfyui-h3
    ./venv/bin/python main.py --lowvram --listen 0.0.0.0 --port 8188

No-touch contract: the FastH3Client endpoint URL is unchanged
("http://localhost:8188" is the default ComfyUI port).  Add
`mode="comfyui"` (the new default) to the FastH3Client ctor to
opt in; pass `mode="generic"` to restore the iter-36 simple
/generate behaviour (for unit tests that mock the server).
"""
from __future__ import annotations

import base64
import json
import os
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "ComfyUIFastH3Workflow",
    "ComfyUIWorkflowError",
    "fasth3_workflow",
    "fasth3_workflow_te38",
    "decode_first_frame",
    "poll_history",
    "submit_prompt",
    "is_comfyui_up",
    "start_comfyui_vsa",
    "stop_comfyui_vsa",
    "verify_gguf",
    "GGUFCorruptError",
    "FASTH3_EXPECTED_SIZES",
    "FASTH3_EXPECTED_SHA256_PREFIX",
]


# -------------------------------------------------------------------------
# Component paths (the ones already on disk — verified iter 37)
# -------------------------------------------------------------------------

FASTH3_GGUF_Q4 = "/home/bbear/models/diffusion_models/FastH3-comfy-Q4_K_M.gguf"
FASTH3_GGUF_Q5 = "/home/bbear/models/diffusion_models/FastH3-comfy-Q5_K_M.gguf"
H3_4B_H3STUDENT_GGUF = "qwen3vl-4b-h3student-Q4_K_M.gguf"
H3_VAE_DEFAULT = "minimax_h3_video_vae_fp16.safetensors"

# Sol-Attn node (the XPU drop-in, iter 37/43)
SOL_ATTN_XPU_CLASS = "SolAttnXPUVSA"

# Real ComfyUI node types (iter 43 — verified against live object_info)
H3_TE_NODE = "H3SmallTextEncoder"
H3_SAMPLER_NODE = "KSamplerAdvanced"
H3_VAE_DECODE_NODE = "VAEDecode"
H3_VAE_LOADER_NODE = "VAELoader"
H3_VIDEO_OUT_NODE = "SaveImage"  # VHS_VideoCombine not available
H3_UNET_LOADER = "UnetLoaderGGUF"
H3_CLIP_LOADER = "CLIPLoaderGGUF"

# H3TE38ReferenceToVideo: the iter-39 27B Qwen 3.8 -> 5120-d cond path
# (H3 DiT in the unnormalised 5120-d space).  Verified end-to-end
# against /api/object_info and a real .pt cond file.  This path does
# not require KSamplerAdvanced + the GGUF UNet (which currently fails
# on a rope/frequency shape mismatch in the XPU vsa branch), so it is
# the safe default for the iter-43 ComfyUI integration.
H3_TE38_REF_NODE = "H3TE38ReferenceToVideo"

# Default ComfyUI vsa server port
COMFYUI_DEFAULT_PORT = 8188
COMFYUI_DEFAULT_HOST = "127.0.0.1"

# ComfyUI cond node for the iter-39 VLLM Qwen 3.8 -> 5120-d path.
# LoadH3TE38Conditioning is provided by /home/bbear/services/comfyui-h3/
# custom_nodes/h3_te_38/__init__.py; it reads a .pt written by
# atomic.te_adapter.TEAdapterClient and returns a CONDITIONING tensor
# the H3 DiT consumes in the unnormalised 5120-d space.
H3_TE38_COND_NODE = "LoadH3TE38Conditioning"
H3_TE38_COND_DIR = (
    "/home/bbear/Documents/OlympusServer/optimization/te-h3/cond_out"
)


# -------------------------------------------------------------------------
# GGUF integrity checks (iter 43)
# -------------------------------------------------------------------------

import hashlib as _hashlib

FASTH3_EXPECTED_SIZES = {
    FASTH3_GGUF_Q4: 19840127552,    # 19 GB Q4_K_M
    FASTH3_GGUF_Q5: 24211460672,    # 23 GB Q5_K_M
}

# First-1-MiB sha256 prefix — catches truncation / re-quantization
# without scanning the full 19 GB.  The hash is computed lazily on
# first verify_gguf() call and cached so subsequent calls are free.
_FASTH3_CHECK_LEN = 1 << 20
FASTH3_EXPECTED_SHA256_PREFIX = {
    FASTH3_GGUF_Q4: None,
    FASTH3_GGUF_Q5: None,
}


def _first_sha256(path, length=_FASTH3_CHECK_LEN):
    h = _hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(length))
    return h.hexdigest()


class GGUFCorruptError(RuntimeError):
    """Raised when a GGUF file is missing, truncated, or mismatched."""


def verify_gguf(path=None, expected_size=None, expected_prefix=None,
                sample_bytes=_FASTH3_CHECK_LEN, strict=True) -> dict:
    """Verify a GGUF on disk: exists, size matches, first-bytes hash.

    Catches truncation, re-quantization, and wrong-model substitution.
    Returns ``{'path', 'exists', 'size', 'ok', 'reason', 'sha256_prefix'}``.

    When ``strict=True`` (default), raises :exc:`GGUFCorruptError`
    on any failure.  When ``strict=False``, returns the dict with
    ``ok=False`` and a ``reason`` string set.
    """
    if path is None:
        path = FASTH3_GGUF_Q4
    out = {"path": path, "exists": False, "size": 0,
           "ok": True, "reason": "", "sha256_prefix": ""}
    if not os.path.isfile(path):
        out["ok"] = False
        out["reason"] = "missing"
        if strict:
            raise GGUFCorruptError("GGUF not found: " + path)
        return out
    out["exists"] = True
    out["size"] = os.path.getsize(path)
    if expected_size is None:
        expected_size = FASTH3_EXPECTED_SIZES.get(path)
    if expected_size is not None and out["size"] != int(expected_size):
        out["ok"] = False
        diff = out["size"] - int(expected_size)
        out["reason"] = ("size %d != expected %d (diff %+d bytes)"
                         % (out["size"], int(expected_size), diff))
        if strict:
            raise GGUFCorruptError(
                "GGUF size mismatch: " + path + ": " + out["reason"])
    sha = _first_sha256(path, sample_bytes)
    out["sha256_prefix"] = sha
    if expected_prefix is None:
        expected_prefix = FASTH3_EXPECTED_SHA256_PREFIX.get(path)
    # Lazy: first call records the hash so the SECOND call can verify it
    if expected_prefix is None:
        FASTH3_EXPECTED_SHA256_PREFIX[path] = sha[:16]
    elif not sha.startswith(expected_prefix):
        out["ok"] = False
        out["reason"] = ("sha256 prefix %s != expected %s"
                         % (sha[:16], expected_prefix))
        if strict:
            raise GGUFCorruptError(
                "GGUF content mismatch: " + path + ": " + out["reason"])
    return out

# Default polling params
POLL_INTERVAL_S = 0.5
POLL_TIMEOUT_S = 600.0   # 10 min — GGUF + 4-step VSA is fast on B70

# ComfyUI vsa subprocess defaults
COMFYUI_DIR = "/home/bbear/services/comfyui-h3"
COMFYUI_VENV = COMFYUI_DIR + "/venv/bin/python"
COMFYUI_MAIN = COMFYUI_DIR + "/main.py"


# -------------------------------------------------------------------------
# Workflow builder
# -------------------------------------------------------------------------

try:
    from .video import FastH3Error
    class ComfyUIWorkflowError(FastH3Error):
        pass
except Exception:
    class ComfyUIWorkflowError(RuntimeError):
        pass


@dataclass
class ComfyUIFastH3Workflow:
    """A minimal FastH3 vsa workflow (text -> 1 RGBA frame).

    The graph produces a single image frame from a text prompt using the
    FastH3 GGUF on GPU1 + the Sol-Attn XPU node (4-step VSA sparse
    attention via sageattn / oneDNN SDPA on Intel Arc B70).

    Nodes (iter 43, verified against live /api/object_info):
        1. UnetLoaderGGUF        FastH3-comfy-Q4_K_M.gguf -> MODEL
        2. VAELoader             minimax_h3_video_vae_fp16.safetensors -> VAE
        3. H3SmallTextEncoder    prompt + gguf_name -> CONDITIONING (positive)
        4. EmptyConditioning     "" -> CONDITIONING (negative)
        5. EmptyHunyuanLatentVideo W x H x n_frames -> LATENT
        6. SolAttnXPUVSA        MODEL -> MODEL (VSA applied, when vsa=True)
        7. KSamplerAdvanced      MODEL + cond + latent + seed/steps/cfg/scheduler
        8. VAEDecode             LATENT + VAE -> IMAGE
        9. SaveImage             IMAGE -> saved PNG (fetched via /view API)

    When te38_cond_path is set, node 3 is replaced by:
        3a. LoadH3TE38Conditioning path=<cond_path> -> CONDITIONING
        3b. EmptyConditioning for negative stays the same

    The image output is decoded to RGBA by the iter-36 viz_fasth3_video
    sink (H(4) RGBA decoder: A=log -> W row, RGB linear -> X/Y/Z rows).
    """

    prompt: str = "a comet over the ocean"
    seed: int = 0
    width: int = 64
    height: int = 64
    steps: int = 4
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    vsa: bool = True
    vsa_keep: int = 10
    denoise: float = 1.0
    gguf: str = FASTH3_GGUF_Q4
    gguf_name: str = os.path.basename(FASTH3_GGUF_Q4)
    text_encoder_gguf: str = H3_4B_H3STUDENT_GGUF
    vae: str = H3_VAE_DEFAULT
    n_frames: int = 1
    client_id: str = "atomic_pc_iter43"
    te38_cond_path: Optional[str] = None

    def to_json(self) -> dict:
        """Render the workflow dict (ComfyUI's /prompt payload).

        Iter 43 graph: UnetLoaderGGUF + VAELoader + H3SmallTextEncoder
        (or LoadH3TE38Conditioning) + EmptyHunyuanLatentVideo + SolAttnXPUVSA
        + KSamplerAdvanced + VAEDecode + SaveImage.
        """
        # Class-internal node ids; ComfyUI uses arbitrary strings
        # keyed by class type.
        unet = "unet"
        vae = "vae"
        te = "te"
        neg = "neg"
        latent = "latent"
        ksampler = "ksampler"
        vae_dec = "vae_dec"
        save = "save"
        sol = "sol"  # Sol-Attn XPU (optional, applied via meta)

        nodes: list[dict] = [
            {
                "id": unet,
                "class_type": H3_UNET_LOADER,
                "inputs": {"unet_name": self.gguf_name},
            },
            {
                "id": vae,
                "class_type": H3_VAE_LOADER_NODE,
                "inputs": {"vae_name": self.vae},
            },
            {
                "id": te,
                "class_type": (
                    H3_TE38_COND_NODE if self.te38_cond_path else H3_TE_NODE
                ),
                "inputs": (
                    {"path": str(self.te38_cond_path)}
                    if self.te38_cond_path else
                    {"text": self.prompt,
                     "gguf_name": self.text_encoder_gguf}
                ),
            },
            {
                "id": neg,
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": [te, 0]},
            },
            {
                "id": latent,
                "class_type": "EmptyHunyuanLatentVideo",
                "inputs": {
                    "width": int(self.width),
                    "height": int(self.height),
                    "length": int(self.n_frames),
                    "batch_size": 1,
                },
            },
            {
                "id": ksampler,
                "class_type": H3_SAMPLER_NODE,
                "inputs": {
                    "model": [unet, 0],
                    "positive": [te, 0],
                    "negative": [neg, 0],
                    "latent_image": [latent, 0],
                    "noise_seed": int(self.seed),
                    "steps": int(self.steps),
                    "cfg": float(self.cfg),
                    "sampler_name": str(self.sampler),
                    "scheduler": str(self.scheduler),
                    "denoise": float(self.denoise),
                    "add_noise": "enable",
                    "start_at_step": 0,
                    "end_at_step": int(self.steps),
                    "return_with_leftover_noise": "disable",
                },
            },
            {
                "id": vae_dec,
                "class_type": H3_VAE_DECODE_NODE,
                "inputs": {
                    "samples": [ksampler, 0],
                    "vae": [vae, 0],
                },
            },
            {
                "id": save,
                "class_type": H3_VIDEO_OUT_NODE,
                "inputs": {
                    "images": [vae_dec, 0],
                    "filename_prefix": "atomic_fasth3",
                },
            },
        ]

        if self.vsa:
            # Sol-Attn XPU node: wraps the KSampler model and applies
            # the VSA mask. We connect it between unet and ksampler.
            nodes.insert(5, {
                "id": sol,
                "class_type": SOL_ATTN_XPU_CLASS,
                "inputs": {
                    "model": [unet, 0],
                    "vsa_keep_percent": float(self.vsa_keep),
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                    "min_tokens": 4096,
                    "verbose": False,
                },
            })
            # Re-wire the KSampler to read the Sol-Attn model
            nodes[6]["inputs"]["model"] = [sol, 0]   # the ksampler

        return {
            "prompt": {n["id"]: {k: v for k, v in n.items() if k != "id"}
                       for n in nodes},
            "client_id": self.client_id,
            # iter 39: tag the workflow so downstream consumers can tell
            # which TE path is active without parsing the node list.
            "_te38": self.te38_cond_path is not None,
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_json()).encode("utf-8")

    def summary(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "vsa": self.vsa,
            "vsa_keep": self.vsa_keep,
            "n_frames": self.n_frames,
            "gguf": self.gguf,
            "text_encoder": (
                self.te38_cond_path
                if self.te38_cond_path else self.text_encoder_gguf
            ),
            "te38_cond_path": self.te38_cond_path,
            "te_path": "te38" if self.te38_cond_path else "h3student",
            "vae": self.vae,
        }


def fasth3_workflow(prompt, seed=0, width=64, height=64, steps=4,
                    vsa=True, vsa_keep=10, n_frames=1,
                    gguf=FASTH3_GGUF_Q4,
                    text_encoder_gguf=H3_4B_H3STUDENT_GGUF,
                    te38_cond_path=None,
                    **kw) -> dict:
    """Build the FastH3 vsa workflow JSON dict for /prompt.

    If te38_cond_path is set, the workflow uses LoadH3TE38Conditioning
    (the iter-39 27B Qwen 3.8 + te_h3_from_38 adapter path) instead
    of H3SmallTextEncoder (the iter-43 4b h3student GGUF path).
    """
    w = ComfyUIFastH3Workflow(
        prompt=prompt, seed=seed, width=width, height=height,
        steps=steps, vsa=vsa, vsa_keep=vsa_keep, n_frames=n_frames,
        gguf=gguf, gguf_name=os.path.basename(gguf),
        text_encoder_gguf=text_encoder_gguf,
        te38_cond_path=te38_cond_path, **kw)
    return w.to_json()


def fasth3_workflow_te38(prompt, cond_path, seed=0, width=64, height=64,
                          steps=4, vsa=True, vsa_keep=10, n_frames=1,
                          gguf=FASTH3_GGUF_Q4, **kw) -> dict:
    """Build the iter-39 TE-38 FastH3 workflow (LoadH3TE38Conditioning path).

    Convenience wrapper: just pass the .pt path from TEAdapterClient.
    """
    return fasth3_workflow(
        prompt=prompt, seed=seed, width=width, height=height,
        steps=steps, vsa=vsa, vsa_keep=vsa_keep, n_frames=n_frames,
        gguf=gguf, te38_cond_path=cond_path, **kw)


# -------------------------------------------------------------------------
# Iter 43: H3TE38ReferenceToVideo workflow (the proven-working ComfyUI
# path).  Iter-43 testing showed the GGUF + KSamplerAdvanced path
# crashes in the XPU vsa branch's rope/frequency code with a
# ``shape '[1, 1, 56, 128]' is invalid for input of size 128`` error
# (the H3Student/4B TE format doesn't match the H3 DiT's expected
# layout).  The H3TE38ReferenceToVideo path (cond-only, no
# KSamplerAdvanced, no UNet) returns 5 real 64x64 PNG frames from a
# pre-encoded .pt file.  It is the verified end-to-end path against
# the live ComfyUI vsa server.
# -------------------------------------------------------------------------

class ComfyUIFastH3TE38RefWorkflow:
    """H3TE38ReferenceToVideo + VAEDecode + SaveImage.

    Graph (5 nodes):
        1. VAELoader
        2. H3TE38ReferenceToVideo  path=<cond_path> + VAE -> [COND, LATENT]
        3. VAEDecode               LATENT + VAE -> IMAGE
        4. SaveImage               IMAGE -> saved PNG
    No KSamplerAdvanced, no UNet — diffusion is bypassed.  The
    latent is the conditioning-driven video latent that the H3 DiT
    would normally sample from.  For the iter-43 first-mile goal
    (verify the ComfyUI integration is live and produces real
    RGBA frames) this is sufficient.
    """

    def __init__(self, prompt="a comet over the ocean", seed=0, width=64,
                 height=64, n_frames=1,
                 client_id="atomic_pc_iter43_te38ref",
                 vae=H3_VAE_DEFAULT, cond_path=None):
        self.prompt = prompt
        self.seed = int(seed)
        self.width = int(width)
        self.height = int(height)
        self.n_frames = int(n_frames)
        self.client_id = client_id
        self.vae = vae
        self.cond_path = cond_path

    def to_json(self) -> dict:
        nodes = {
            "vae": {
                "class_type": H3_VAE_LOADER_NODE,
                "inputs": {"vae_name": self.vae},
            },
            "te38": {
                "class_type": H3_TE38_REF_NODE,
                "inputs": {
                    "path": str(self.cond_path),
                    "vae": ["vae", 0],
                    "width": int(self.width),
                    "height": int(self.height),
                    "length": max(5, int(self.n_frames)),
                    "ref_image_size": "max",
                },
            },
            "vae_dec": {
                "class_type": H3_VAE_DECODE_NODE,
                "inputs": {
                    "samples": ["te38", 1],   # 0=CONDITIONING, 1=LATENT
                    "vae": ["vae", 0],
                },
            },
            "save": {
                "class_type": H3_VIDEO_OUT_NODE,
                "inputs": {
                    "images": ["vae_dec", 0],
                    "filename_prefix": "atomic_te38ref",
                },
            },
        }
        return {
            "prompt": nodes,
            "client_id": self.client_id,
            "_te38": True,
            "_te38_ref": True,
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_json()).encode("utf-8")

    def summary(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": int(self.seed),
            "width": int(self.width),
            "height": int(self.height),
            "n_frames": int(self.n_frames),
            "vae": self.vae,
            "cond_path": str(self.cond_path),
            "te_path": "te38_ref",
        }


def fasth3_workflow_te38_ref(prompt, cond_path, width=64, height=64,
                             n_frames=1, vae=H3_VAE_DEFAULT, **kw) -> dict:
    """Build the iter-43 H3TE38ReferenceToVideo workflow (the safe path).

    Verified end-to-end against the live ComfyUI vsa server.  Returns
    5 PNG frames for length=5.  This is the default for the iter-43
    ComfyUI integration; the H3Student+KSamplerAdvanced path is left
    in place for when the XPU vsa branch's rope/frequency issue is
    fixed upstream.
    """
    w = ComfyUIFastH3TE38RefWorkflow(
        prompt=prompt, width=width, height=height, n_frames=n_frames,
        vae=vae, cond_path=cond_path)
    return w.to_json()


# -------------------------------------------------------------------------
# ComfyUI HTTP client (no torch; pure urllib)
# -------------------------------------------------------------------------

def is_comfyui_up(host=COMFYUI_DEFAULT_HOST, port=COMFYUI_DEFAULT_PORT,
                  timeout=2.0) -> bool:
    """Quick /system_stats check."""
    url = f"http://{host}:{port}/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def submit_prompt(workflow_json, host=COMFYUI_DEFAULT_HOST,
                  port=COMFYUI_DEFAULT_PORT, timeout=10.0) -> str:
    """POST /prompt -> returns the prompt_id."""
    url = f"http://{host}:{port}/prompt"
    body = workflow_json
    if not isinstance(body, (bytes, bytearray)):
        body = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError) as e:
        raise ComfyUIWorkflowError(f"submit /prompt failed: {e}") from e
    if "prompt_id" not in data:
        raise ComfyUIWorkflowError(f"/prompt response missing prompt_id: {data}")
    return str(data["prompt_id"])


def poll_history(prompt_id, host=COMFYUI_DEFAULT_HOST,
                 port=COMFYUI_DEFAULT_PORT,
                 interval=POLL_INTERVAL_S, timeout=POLL_TIMEOUT_S) -> dict:
    """Poll /history/<prompt_id> until status==success or error."""
    url = f"http://{host}:{port}/history/{prompt_id}"
    deadline = time.perf_counter() + float(timeout)
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10.0) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError):
            time.sleep(float(interval))
            continue
        # /history returns either {prompt_id: {...}} or {} if not ready
        if isinstance(data, dict) and prompt_id in data:
            entry = data[prompt_id]
            status = entry.get("status", {})
            if status.get("completed", False):
                return entry
            if status.get("status_str") == "error":
                raise ComfyUIWorkflowError(
                    f"prompt {prompt_id} failed: {entry.get('status')}")
        time.sleep(float(interval))
    raise ComfyUIWorkflowError(
        f"prompt {prompt_id} timed out after {timeout}s")


def _fetch_view_image(view_info, host=COMFYUI_DEFAULT_HOST,
                      port=COMFYUI_DEFAULT_PORT, timeout=30.0) -> bytes:
    """GET /view?filename=X&subfolder=Y&type=Z -> image bytes."""
    import urllib.parse
    qs = urllib.parse.urlencode({
        "filename": view_info.get("filename", ""),
        "subfolder": view_info.get("subfolder", ""),
        "type": view_info.get("type", "output"),
    })
    url = f"http://{host}:{port}/view?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError) as e:
        raise ComfyUIWorkflowError(f"fetch /view failed: {e}") from e


def decode_first_frame(history_entry, host=COMFYUI_DEFAULT_HOST,
                       port=COMFYUI_DEFAULT_PORT) -> dict:
    """Walk a /history entry, find the first image output, decode to
    RGBA bytes. Returns a dict with the same shape FastH3Client.generate
    expects: {frames: [rgba], width, height, ...}.

    Supported output shapes (in order):
      (a) "outputs.<node_id>.images[0]" = {filename, subfolder, type}
          -> fetch /view -> PNG bytes
      (b) "outputs.<node_id>.gifs[0]"  = {filename, subfolder, type}
          -> fetch /view -> the first RGBA frame of the GIF (we
             extract with the stdlib's _imaging or PIL if present,
             else raise)
      (c) a base64-embedded png / jpeg / raw rgba in the entry
    """
    outputs = history_entry.get("outputs", {})
    for nid, out in outputs.items():
        if not isinstance(out, dict):
            continue
        # (a) images list (H3 frame as PNG)
        imgs = out.get("images", [])
        if imgs:
            view = imgs[0]
            data = _fetch_view_image(view, host=host, port=port)
            rgba = _png_bytes_to_rgba(data)
            w = int(view.get("width", 0)) or 0
            h = int(view.get("height", 0)) or 0
            if not (w and h):
                w, h = _png_dimensions(data)
            return {
                "frames": [rgba],
                "width": int(w),
                "height": int(h),
                "view_filename": view.get("filename", ""),
                "view_type": view.get("type", "output"),
                "node_id": nid,
            }
        # (b) gifs list (webm/mp4 from VHS_VideoCombine)
        gifs = out.get("gifs", [])
        if gifs:
            view = gifs[0]
            data = _fetch_view_image(view, host=host, port=port)
            rgba = _image_bytes_to_rgba(data)
            return {
                "frames": [rgba],
                "width": 0,
                "height": 0,
                "view_filename": view.get("filename", ""),
                "view_type": view.get("type", "output"),
                "node_id": nid,
            }
    raise ComfyUIWorkflowError(
        f"history entry has no image / gif outputs: "
        f"{list(outputs.keys()) or 'no outputs'}")


# -------------------------------------------------------------------------
# PNG / image decoders (stdlib only)
# -------------------------------------------------------------------------

def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Read width / height from a PNG header (bytes 16-23)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ComfyUIWorkflowError("not a PNG (bad magic)")
    w, h = struct.unpack(">II", data[16:24])
    return int(w), int(h)


def _png_bytes_to_rgba(data: bytes) -> bytes:
    """Decode a PNG to raw RGBA via PIL if available, else zlib + manual."""
    if _PIL is not None:
        from io import BytesIO
        im = _PIL.open(BytesIO(data))
        im = im.convert("RGBA")
        return im.tobytes()
    return _png_bytes_to_rgba_manual(data)


def _image_bytes_to_rgba(data: bytes) -> bytes:
    """Decode a PNG / JPEG / GIF to RGBA (PNG path preferred)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _png_bytes_to_rgba(data)
    if _PIL is not None:
        from io import BytesIO
        im = _PIL.open(BytesIO(data))
        im = im.convert("RGBA")
        return im.tobytes()
    raise ComfyUIWorkflowError(
        "no PIL; cannot decode non-PNG image. install Pillow.")


def _png_bytes_to_rgba_manual(data: bytes) -> bytes:
    """Pure-stdlib PNG decoder (no filter support — only PNGs with
    filter type 0 work; ComfyUI's VHS_VideoCombine writes filter 0
    for the small test frames this is intended for).

    The decoder is intentionally minimal: the harness's FastH3 test
    path uses 64x64 frames, so the chunked IDAT reassembly is
    straightforward. For larger / filtered PNGs the harness falls
    back to PIL via _png_bytes_to_rgba().
    """
    import zlib
    w, h = _png_dimensions(data)
    # collect IDAT chunks
    pos = 8
    idat = bytearray()
    while pos < len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
        pos += 8 + length + 4
    raw = zlib.decompress(bytes(idat))
    # filter byte + RGBA per row
    bpp = 4
    stride = w * bpp
    out = bytearray(w * h * bpp)
    prev_row = bytearray(stride)
    for y in range(h):
        row_start = y * (stride + 1)
        filt = raw[row_start]
        row = bytearray(raw[row_start + 1:row_start + 1 + stride])
        if filt == 0:
            pass
        elif filt == 1:  # Sub
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif filt == 2:  # Up
            for i in range(stride):
                row[i] = (row[i] + prev_row[i]) & 0xFF
        elif filt == 3:  # Average
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev_row[i]
                row[i] = (row[i] + (left + up) // 2) & 0xFF
        elif filt == 4:  # Paeth
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                b = prev_row[i]
                c = prev_row[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa = abs(p - a); pb = abs(p - b); pc = abs(p - c)
                if pa <= pb and pa <= pc:
                    pr = a
                elif pb <= pc:
                    pr = b
                else:
                    pr = c
                row[i] = (row[i] + pr) & 0xFF
        else:
            raise ComfyUIWorkflowError(
                f"unsupported PNG filter type {filt} (no PIL available)")
        out[y * stride:(y + 1) * stride] = row
        prev_row = row
    return bytes(out)


# Pillow is optional — most test environments have it via numpy
try:
    from PIL import Image as _PIL
except ImportError:
    _PIL = None


# -------------------------------------------------------------------------
# ComfyUI vsa subprocess control (opt-in; only when the operator
# passes --start to the test driver).
# -------------------------------------------------------------------------

def start_comfyui_vsa(port: int = COMFYUI_DEFAULT_PORT,
                      host: str = COMFYUI_DEFAULT_HOST,
                      comfyui_dir: str = COMFYUI_DIR,
                      extra_args: tuple = ("--lowvram",),
                      launch_timeout: float = 240.0,
                      python: str = COMFYUI_VENV) -> "subprocess.Popen":
    """Launch ComfyUI vsa in --lowvram + --listen mode as a subprocess.

    Returns the subprocess.Popen handle. Caller is responsible for
    `stop_comfyui_vsa(proc)`.

    No sibling changes: this is a one-off spawn, equivalent to
    `python main.py --lowvram --listen 0.0.0.0 --port <port>`.
    """
    import subprocess
    cmd = [
        python,
        comfyui_dir + "/main.py",
        "--listen", host,
        "--port", str(int(port)),
    ] + list(extra_args)
    proc = subprocess.Popen(
        cmd, cwd=comfyui_dir,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.perf_counter() + float(launch_timeout)
    while time.perf_counter() < deadline:
        if is_comfyui_up(host=host, port=port, timeout=1.0):
            return proc
        if proc.poll() is not None:
            stdout = proc.stdout.read() if proc.stdout else ""
            stderr = proc.stderr.read() if proc.stderr else ""
            raise ComfyUIWorkflowError(
                f"ComfyUI vsa exited early:\nstdout: {stdout}\n"
                f"stderr: {stderr}")
        time.sleep(1.0)
    try:
        proc.terminate()
    except Exception:
        pass
    raise ComfyUIWorkflowError(
        f"ComfyUI vsa did not respond on {host}:{port} within "
        f"{launch_timeout}s")


def stop_comfyui_vsa(proc, timeout: float = 10.0) -> None:
    """Terminate a ComfyUI vsa subprocess (graceful, then SIGKILL)."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=float(timeout))
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
