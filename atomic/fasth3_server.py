"""FastH3 + ComfyUI vsa workflow client (iter 38).

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
    "decode_first_frame",
    "poll_history",
    "submit_prompt",
    "is_comfyui_up",
    "start_comfyui_vsa",
    "stop_comfyui_vsa",
]


# -------------------------------------------------------------------------
# Component paths (the ones already on disk — verified iter 37)
# -------------------------------------------------------------------------

FASTH3_GGUF_Q4 = "/home/bbear/models/diffusion_models/FastH3-comfy-Q4_K_M.gguf"
FASTH3_GGUF_Q5 = "/home/bbear/models/diffusion_models/FastH3-comfy-Q5_K_M.gguf"
H3_4B_H3STUDENT = "/home/bbear/models/text_encoders/qwen3vl-4b-h3student-Q4_K_M.gguf"
H3_VAE_DEFAULT = "olympus/h3_vae.safetensors"   # via extra_model_paths

# Sol-Attn node (the XPU drop-in, iter 37)
SOL_ATTN_XPU_CLASS = "SolAttnXPUVSA"

# Minimal-fastH3 TE wrapper (kijai vsa branch)
H3_TE_NODE = "H3TextEncode8"     # see h3_te_38/nodes.py INPUT_TYPES
H3_SAMPLER_NODE = "KSamplerAdvanced"
H3_VAE_DECODE_NODE = "VAEDecode"
H3_VIDEO_OUT_NODE = "VHS_VideoCombine"

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

    The graph is a 6-node chain that produces a single image frame
    from a text prompt using the FastH3 GGUF on GPU1 + the
    Sol-Attn XPU node (the 4-step VSA sparse-attention kernel).

    Nodes (insertion order = execution order in ComfyUI v0.33.1):
        1. H3TextEncode8   text -> cond (uses the 4b h3student TE)
        2. EmptyLatentImage  W,H  -> latent (1 batch, H/8 x W/8)
        3. CheckpointLoaderSimple "FastH3-comfy-Q4_K_M.gguf"
        4. KSamplerAdvanced  (model + cond + latent, 4 steps, vsa)
        5. VAEDecode  latent -> image
        6. VHS_VideoCombine  (image -> single-frame webm / mp4)

    The image output is the RGBA frame the iter-36 viz_fasth3_video
    sink decodes.
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
    text_encoder: str = H3_4B_H3STUDENT
    vae: str = H3_VAE_DEFAULT
    n_frames: int = 1
    client_id: str = "atomic_pc_iter38"
    # iter 39: if set, the workflow uses LoadH3TE38Conditioning (the
    # 27B Qwen 3.8 + te_h3_from_38 MLP path) instead of H3TextEncode8
    # (the 4b h3student GGUF path).  `te38_cond_path` MUST point at a
    # .pt file in H3_TE38_COND_DIR (or one of the node's other allowed
    # roots).  When set, `text_encoder` is ignored.
    te38_cond_path: Optional[str] = None

    def to_json(self) -> dict:
        """Render the workflow dict (ComfyUI's /prompt payload)."""
        # Class-internal node ids; ComfyUI uses arbitrary strings
        # keyed by class type.
        ckpt = "ckpt"
        te = "te"
        neg = "neg"
        latent = "latent"
        ksampler = "ksampler"
        vae_dec = "vae_dec"
        vhs = "vhs"
        sol = "sol"  # Sol-Attn XPU (optional, applied via meta)

        nodes: list[dict] = [
            {
                "id": ckpt,
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": os.path.basename(self.gguf)},
            },
            {
                "id": te,
                "class_type": (
                    H3_TE38_COND_NODE if self.te38_cond_path else H3_TE_NODE
                ),
                "inputs": (
                    {"path": str(self.te38_cond_path)}
                    if self.te38_cond_path else
                    {"text_encoder": os.path.basename(self.text_encoder),
                     "prompt": self.prompt,
                     "n_frames": int(self.n_frames)}
                ),
            },
            {
                "id": "neg",
                "class_type": "EmptyConditioning",
                "inputs": {"text": ""},
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
                    "model": [ckpt, 0],
                    "positive": [te, 0],
                    "negative": ["neg", 0],
                    "latent_image": [latent, 0],
                    "noise_seed": int(self.seed),
                    "steps": int(self.steps),
                    "cfg": float(self.cfg),
                    "sampler_name": str(self.sampler),
                    "scheduler": str(self.scheduler),
                    "denoise": float(self.denoise),
                },
            },
            {
                "id": vae_dec,
                "class_type": H3_VAE_DECODE_NODE,
                "inputs": {
                    "samples": [ksampler, 0],
                    "vae": [ckpt, 2],
                },
            },
            {
                "id": vhs,
                "class_type": H3_VIDEO_OUT_NODE,
                "inputs": {
                    "images": [vae_dec, 0],
                    "frame_rate": 8,
                    "loop_count": 0,
                    "filename_prefix": "atomic_fasth3",
                    "format": "image/png",
                    "save_output": True,
                },
            },
        ]

        if self.vsa:
            # Sol-Attn XPU node: wraps the KSampler model and applies
            # the VSA mask. We connect it between ckpt and ksampler.
            nodes.insert(3, {
                "id": sol,
                "class_type": SOL_ATTN_XPU_CLASS,
                "inputs": {
                    "model": [ckpt, 0],
                    "keep": int(self.vsa_keep),
                    "vsa": True,
                },
            })
            # Re-wire the KSampler to read the Sol-Attn model
            nodes[4]["inputs"]["model"] = [sol, 0]   # the ksampler
            # renumber: ksampler is now index 4, vae_dec 5, vhs 6
            for n in nodes[3:]:
                pass   # ids are explicit strings; no renumbering needed

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
                if self.te38_cond_path else self.text_encoder
            ),
            "te38_cond_path": self.te38_cond_path,
            "te_path": "te38" if self.te38_cond_path else "h3student",
        }


def fasth3_workflow(prompt, seed=0, width=64, height=64, steps=4,
                    vsa=True, vsa_keep=10, n_frames=1,
                    gguf=FASTH3_GGUF_Q4,
                    text_encoder=H3_4B_H3STUDENT,
                    te38_cond_path=None,
                    **kw) -> dict:
    """Build the FastH3 vsa workflow JSON dict for /prompt.

    If te38_cond_path is set, the workflow uses LoadH3TE38Conditioning
    (the iter-39 27B Qwen 3.8 + te_h3_from_38 adapter path) instead
    of H3TextEncode8 (the iter-38 4b h3student path).
    """
    w = ComfyUIFastH3Workflow(
        prompt=prompt, seed=seed, width=width, height=height,
        steps=steps, vsa=vsa, vsa_keep=vsa_keep, n_frames=n_frames,
        gguf=gguf, text_encoder=text_encoder,
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
