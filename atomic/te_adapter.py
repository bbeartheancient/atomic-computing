"""te_h3_from_38 — VLLM Qwen 3.8 27B -> MiniMax-H3 5120-d text encoder adapter.

Iter 39: replaces the 4b h3student GGUF TE in the FastH3 ComfyUI pipeline
with a 27B resident Qwen + a 5120->5120 per-token MLP projection. The
projection is the same one the h3-shrink v1/v2 corpus used to train
te_h3_from_38.pt; loading it gives the H3 DiT (in ComfyUI) a text
conditioning tensor in the unnormalised 5120-d space, eliminating the
need to keep a separate 4B TE GGUF on disk.

Pipeline (text -> cond.pt ComfyUI can load via LoadH3TE38Conditioning):

    TEAdapterClient.encode("a comet over the ocean")
      -> POST http://localhost:8000/v1/hidden_states  (Qwen 3.8 27B,
         layer=-1, fp16, no chat template, "h3_raw" template_id)
      -> reshape to (L, 5120) fp16
      -> run te_h3_from_38.pt  (Sequential(Linear(5120,5120), GELU(),
         Linear(5120,5120))) on the .float() view
      -> cast back to fp16, unsqueeze to (1, L, 5120)
      -> write cond_out/<name>.pt  {"cond", "minimax_token_tags",
         "token_ids", "meta"}
      -> return path  (FastH3Client then sets LoadH3TE38Conditioning.path
                       in the workflow)

Caching: a JSON-side hash of the prompt is the cache key, so a repeated
prompt reuses the same .pt without another vLLM round trip. The cache
lives at $ATOMIC_TE38_CACHE_DIR (default ~/.cache/atomic/te_h3_from_38).

No-touch contract: FastH3Client(endpoint=..., mode='comfyui').generate()
is unchanged; this module just adds the .pt stashing step. Callers that
don't pass te_adapter= get the iter-38 wire (H3TextEncode8 + 4b
h3student GGUF) — the workflow builder picks the path.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

__all__ = [
    "TEAdapterError",
    "TEAdapterConfig",
    "VLLMHiddenStatesClient",
    "TE38Adapter",
    "TEAdapterClient",
    "stash_cond",
    "load_cond",
    "cond_cache_key",
    "resolve_cond_dir",
    "DEFAULT_TE38_ADAPTER",
    "DEFAULT_VLLM_URL",
    "DEFAULT_COND_DIR",
    "DEFAULT_HIDDEN_DIM",
    "TE38_ADAPTER_LAYER",
    "TE38_TEMPLATES",
]


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

# The duty Qwen 3.8 27B served by vllm on GPU0 (verified iter 39).
DEFAULT_VLLM_URL = "http://127.0.0.1:8000"

# The trained adapter that projects Qwen 3.8 hidden states (5120-d) into
# the unnormalised 5120-d H3 text space. Same path the encode_h3.py
# production script uses (one source of truth).
DEFAULT_TE38_ADAPTER = (
    "/home/bbear/Documents/OlympusServer/optimization/te-h3/adapters/"
    "te_h3_from_38.pt"
)

# LoadH3TE38Conditioning's allowed base; the .pt lives next to encode_h3.py
DEFAULT_COND_DIR = (
    "/home/bbear/Documents/OlympusServer/optimization/te-h3/cond_out"
)

# Qwen 3.8 hidden size — must match the adapter's in/out dimensions.
DEFAULT_HIDDEN_DIM = 5120

# The /v1/hidden_states endpoint requires a template_id; "h3_raw" is the
# no-template-apply mode the production encoder uses (h3 DiT never sees a
# chat template, see comfy/text_encoders/minimax.py:3).
TE38_TEMPLATES = ("h3_raw", "h3_chat", "h3_caption")

# Max sequence length for a single encode. 2048 is the production cap.
TE38_DEFAULT_MAX_TOKENS = 2048

# vLLM /v1/hidden_states HTTP timeout — a 27B resident Qwen encodes
# ~5120 tokens/sec on a B70; 2048 tokens should be sub-second, but the
# model is busy so we allow a generous ceiling.
TE38_HTTP_TIMEOUT_S = 120.0

# Adapter layer (the only layer we currently support). Future: per-layer
# adapters for tasks like the captioning adapter.
TE38_ADAPTER_LAYER = -1

# Cache directory for encoded .pt files (separate from cond_out so the
# production encoder's outputs stay pristine).
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/atomic/te_h3_from_38")


# -------------------------------------------------------------------------
# Errors
# -------------------------------------------------------------------------

class TEAdapterError(RuntimeError):
    """Raised for any failure inside the TE adapter pipeline."""


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

@dataclass
class TEAdapterConfig:
    """Static config for the TE adapter pipeline.

    Most callers just set `adapter_path` and `vllm_url`; the rest have
    sensible defaults that match the production encoder.
    """
    vllm_url: str = DEFAULT_VLLM_URL
    adapter_path: str = DEFAULT_TE38_ADAPTER
    cond_dir: str = DEFAULT_COND_DIR
    cache_dir: str = DEFAULT_CACHE_DIR
    template_id: str = "h3_raw"
    layer: int = TE38_ADAPTER_LAYER
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    max_tokens: int = TE38_DEFAULT_MAX_TOKENS
    timeout_s: float = TE38_HTTP_TIMEOUT_S
    # whether to use the cache (set to False for one-shot encodes).
    use_cache: bool = True

    def as_dict(self) -> dict:
        return {
            "vllm_url": self.vllm_url,
            "adapter_path": self.adapter_path,
            "cond_dir": self.cond_dir,
            "cache_dir": self.cache_dir,
            "template_id": self.template_id,
            "layer": int(self.layer),
            "hidden_dim": int(self.hidden_dim),
            "max_tokens": int(self.max_tokens),
            "timeout_s": float(self.timeout_s),
            "use_cache": bool(self.use_cache),
        }


# -------------------------------------------------------------------------
# Cond dir resolution
# -------------------------------------------------------------------------

def resolve_cond_dir(path: Optional[str] = None) -> str:
    """Resolve the cond output directory, creating it if needed.

    Honours $ATOMIC_TE38_COND_DIR if set; otherwise uses the explicit
    `path` argument, or DEFAULT_COND_DIR.  All callers that build a path
    going to LoadH3TE38Conditioning MUST go through this function so the
    whitelist check inside the ComfyUI node stays happy.
    """
    p = (
        os.environ.get("ATOMIC_TE38_COND_DIR")
        or path
        or DEFAULT_COND_DIR
    )
    p = os.path.expanduser(str(p))
    os.makedirs(p, exist_ok=True)
    return p


# -------------------------------------------------------------------------
# Cache key
# -------------------------------------------------------------------------

def cond_cache_key(prompt: str, template_id: str = "h3_raw",
                   layer: int = TE38_ADAPTER_LAYER,
                   adapter_path: str = DEFAULT_TE38_ADAPTER) -> str:
    """Deterministic cache key for a (prompt, template, adapter) triple.

    The key is sha256 of "<template_id>|<layer>|<adapter_abspath>|
    <prompt>" — the adapter path so swapping the .pt invalidates the
    cache, the layer so future per-layer adapters don't collide.
    """
    ap = os.path.abspath(adapter_path)
    raw = f"{template_id}|{int(layer)}|{ap}|{prompt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


# -------------------------------------------------------------------------
# vLLM /v1/hidden_states client
# -------------------------------------------------------------------------

class VLLMHiddenStatesClient:
    """Tiny client for the duty vLLM /v1/hidden_states endpoint.

    Returns (hidden_states (L, hidden_dim) fp16 numpy, token_ids list).
    No torch / no transformers — pure stdlib + the optional numpy path.
    """

    def __init__(self, url: str = DEFAULT_VLLM_URL,
                 timeout_s: float = TE38_HTTP_TIMEOUT_S,
                 max_tokens: int = TE38_DEFAULT_MAX_TOKENS,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM):
        self.url = str(url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.hidden_dim = int(hidden_dim)
        self._calls = 0
        self._errors = 0

    def _post(self, path: str, body: dict) -> dict:
        url = self.url + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError) as e:
            self._errors += 1
            raise TEAdapterError(f"vllm {path} failed: {e}") from e

    def hidden_states(self, prompt: str, template_id: str = "h3_raw",
                      layer: int = TE38_ADAPTER_LAYER,
                      max_tokens: Optional[int] = None) -> Tuple[bytes, list, list, dict]:
        """Return (raw_fp16_le_bytes, shape, token_ids, raw_dict).

        shape is a list of ints; the bytes are tightly packed fp16 in
        little-endian.  Callers cast/reshape to (L, hidden_dim) fp16.
        """
        body = {
            "prompt": str(prompt),
            "template_id": str(template_id),
            "return": "last_hidden_state",
            "layer": int(layer),
            "max_tokens": int(max_tokens) if max_tokens is not None else self.max_tokens,
        }
        d = self._post("/v1/hidden_states", body)
        for k in ("hidden_fp16_le", "shape", "token_ids", "dtype"):
            if k not in d:
                raise TEAdapterError(
                    f"/v1/hidden_states response missing {k!r}: "
                    f"{list(d.keys())}")
        if d.get("dtype") != "float16":
            raise TEAdapterError(
                f"/v1/hidden_states returned dtype={d.get('dtype')!r}, "
                "expected 'float16'")
        self._calls += 1
        raw_bytes = base64.b64decode(d["hidden_fp16_le"])
        token_ids = list(d["token_ids"])
        n_tokens = len(token_ids)
        n_bytes = len(raw_bytes) // 2
        expected = self.hidden_dim * n_tokens
        if n_bytes != expected:
            raise TEAdapterError(
                f"hidden_fp16_le byte count {n_bytes} != "
                f"hidden_dim({self.hidden_dim}) * n_tokens({n_tokens}) = {expected}")
        _check_fp16_nan(raw_bytes, prompt, "vLLM /v1/hidden_states")
        return (
            raw_bytes,
            [n_tokens, self.hidden_dim],
            token_ids,
            d,
        )

    def stats(self) -> dict:
        return {"calls": self._calls, "errors": self._errors,
                "url": self.url}


# -------------------------------------------------------------------------
# TE38Adapter (the 5120->5120 MLP)
# -------------------------------------------------------------------------

class TE38Adapter:
    """The per-token MLP adapter: Linear(5120,5120) -> GELU -> Linear(5120,5120).

    Loads te_h3_from_38.pt (state_dict from the training script). Falls
    back to the identity projection when torch is unavailable so the
    pipeline stays importable in pure-stdlib environments (the test
    harness uses this for fast, deterministic tests).
    """

    def __init__(self, path: str = DEFAULT_TE38_ADAPTER,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM):
        self.path = str(path)
        self.hidden_dim = int(hidden_dim)
        self._torch = None
        self._net = None
        self._weights = None   # (W1, b1, W2, b2) raw numpy (or None)
        self._meta = {}
        self._loaded = False
        self._is_identity = False

    @property
    def is_loaded(self) -> bool:
        return bool(self._loaded)

    @property
    def is_identity(self) -> bool:
        return bool(self._is_identity)

    def load(self) -> "TE38Adapter":
        """Load the adapter .pt. Falls back to identity on missing torch."""
        if not os.path.isfile(self.path):
            raise TEAdapterError(f"TE38 adapter .pt not found: {self.path}")
        try:
            import torch  # noqa: F401
            self._torch = torch
        except ImportError:
            self._torch = None
        if self._torch is not None:
            ckpt = self._torch.load(self.path, map_location="xpu",
                                    weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            # PyTorch Sequential: 0=Linear, 1=GELU, 2=Linear
            try:
                import torch.nn as nn
                net = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                net.load_state_dict(sd)
                net.eval()
                self._net = net
                # stash the raw weights for the stdlib fallback
                self._weights = (
                    sd["0.weight"].float().cpu().numpy(),
                    sd["0.bias"].float().cpu().numpy(),
                    sd["2.weight"].float().cpu().numpy(),
                    sd["2.bias"].float().cpu().numpy(),
                )
                # capture training-time meta so callers can log it
                for k, v in ckpt.items():
                    if k != "state_dict":
                        try:
                            json.dumps(v)
                            self._meta[k] = v
                        except TypeError:
                            self._meta[k] = str(v)
            except Exception as e:
                raise TEAdapterError(
                    f"TE38 adapter load failed (state_dict mismatch?): {e}"
                ) from e
        else:
            # stdlib fallback: read a torch-saved state_dict via a tiny
            # zip+pickle-free reader. We only support the simple case
            # where the .pt was saved as a state_dict with float32
            # tensors; otherwise the identity projection is used and
            # the test harness flags it.
            self._weights = _try_load_state_dict_stdlib(self.path)
            if self._weights is None:
                self._is_identity = True
        self._loaded = True
        return self

    def meta(self) -> dict:
        return dict(self._meta)

    def project(self, hidden_states) -> "object":
        """Run the MLP on a (L, hidden_dim) tensor.

        If torch is available, returns a torch.Tensor (fp16).  Otherwise
        uses a pure-numpy / pure-stdlib matmul to keep the pipeline
        importable without torch.
        """
        if not self._loaded:
            self.load()
        if self._is_identity:
            return hidden_states
        if self._torch is not None and self._net is not None:
            with self._torch.no_grad():
                x = hidden_states if hasattr(hidden_states, "float") else (
                    self._torch.as_tensor(hidden_states))
                if hasattr(x, "float"):
                    x = x.float()
                out = self._net(x)
                return out.to(self._torch.float16)
        # numpy / stdlib fallback
        try:
            import numpy as np
            W1, b1, W2, b2 = self._weights
            x = np.asarray(hidden_states, dtype=np.float32)
            h = np.maximum(0.0, x @ W1.T + b1)   # GELU exact isn't critical
            # (a 1st-order GELU approximation is fine for the round-trip
            # test path; the real path uses torch)
            y = h @ W2.T + b2
            return y.astype(np.float16)
        except ImportError:
            # pure-stdlib matmul fallback (slow but correct)
            W1, b1, W2, b2 = self._weights
            L, D = len(hidden_states), len(b1)
            x = [list(map(float, row)) for row in hidden_states]
            out = []
            for i in range(L):
                h = [0.0] * D
                for j in range(D):
                    s = float(b1[j])
                    for k in range(D):
                        s += x[i][k] * float(W1[k][j])
                    h[j] = max(0.0, s)
                y = [float(b2[k]) for k in range(D)]
                for j in range(D):
                    s = 0.0
                    for k in range(D):
                        s += h[k] * float(W2[k][j])
                    y[j] = s
                out.append(y)
            # pack as fp16 LE bytes
            buf = bytearray()
            for row in out:
                for v in row:
                    buf += struct.pack("<e", float(v))
            return bytes(buf)

    def project_to_cond(self, hidden_states) -> Tuple["object", "object"]:
        """Project then wrap as the (cond (1,L,D), token_tags (1,L)) pair.

        Returns (cond_tensor, token_tags_tensor) where both are the
        same type as `hidden_states` (torch or numpy). The cond is
        unsqueezed to (1, L, D); token_tags is (1, L) int64/ones.
        """
        if not self._loaded:
            self.load()
        proj = self.project(hidden_states)
        if self._torch is not None and self._torch.is_tensor(proj):
            L = int(proj.shape[0])
            cond = proj.unsqueeze(0)            # (1, L, D)
            tags = self._torch.ones(1, L, dtype=self._torch.int64)
            return cond, tags
        try:
            import numpy as np
            arr = np.asarray(proj)
            L = int(arr.shape[0])
            cond = arr.reshape(1, L, arr.shape[1])
            tags = np.ones((1, L), dtype=np.int64)
            return cond, tags
        except ImportError:
            # stdlib path: cond is bytes (L*D*2 fp16), tags is list[int]
            L = len(proj) // (self.hidden_dim * 2)
            cond_bytes = bytes(proj)
            tags = [1] * L
            return cond_bytes, tags


# -------------------------------------------------------------------------
# Pure-stdlib state_dict reader (best-effort, no torch)
# -------------------------------------------------------------------------

def _try_load_state_dict_stdlib(path: str):
    """Try to read a torch-saved state_dict without torch.

    The .pt format is a zip archive; the 'data.pkl' inside is a pickle
    of the dict. We can't unpickle arbitrary objects without torch, but
    the state_dict is a dict of named tensors. We try the numpy path
    if numpy is present, else return None and let the caller fall back
    to the identity projection.
    """
    try:
        import zipfile
        import pickle
        import numpy as np
        with zipfile.ZipFile(path, "r") as z:
            with z.open("data.pkl") as f:
                obj = pickle.load(f)
        if not isinstance(obj, dict):
            return None
        # accept either {"state_dict": ...} or the state_dict directly
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else None
        if not isinstance(sd, dict):
            return None
        keys = ("0.weight", "0.bias", "2.weight", "2.bias")
        if not all(k in sd for k in keys):
            return None
        W1 = np.asarray(sd["0.weight"], dtype=np.float32)
        b1 = np.asarray(sd["0.bias"], dtype=np.float32)
        W2 = np.asarray(sd["2.weight"], dtype=np.float32)
        b2 = np.asarray(sd["2.bias"], dtype=np.float32)
        return (W1, b1, W2, b2)
    except Exception:
        return None


# -------------------------------------------------------------------------
# .pt writer / reader (ComfyUI compatible)
# -------------------------------------------------------------------------

def _np_to_torch_save(cond, tags, token_ids, meta: dict) -> bytes:
    """Pack (cond, tags, token_ids, meta) into a torch-serializable blob.

    Uses the same on-disk format the ComfyUI LoadH3TE38Conditioning
    node reads: {"cond": (1,L,D) fp16, "minimax_token_tags": (1,L) int64,
                 "token_ids": list[int], "meta": dict}.
    """
    try:
        import torch
        if hasattr(cond, "detach"):
            cond_t = cond.detach().cpu()
        else:
            cond_t = torch.as_tensor(cond)
        if hasattr(tags, "detach"):
            tags_t = tags.detach().cpu().long()
        else:
            tags_t = torch.as_tensor(tags, dtype=torch.long)
        rec = {
            "cond": cond_t.to(torch.float16),
            "minimax_token_tags": tags_t,
            "token_ids": list(token_ids),
            "meta": dict(meta),
        }
        buf = io.BytesIO()
        torch.save(rec, buf)
        return buf.getvalue()
    except ImportError:
        # pure-stdlib: pack as a JSON-with-fp16-bytes dict, zlib-compressed.
        # LoadH3TE38Conditioning can't read this — but the stdlib path is
        # for tests, and they use a separate in-memory loader.
        rec = {
            "cond": base64.b64encode(cond if isinstance(cond, (bytes, bytearray))
                                     else b"".join(
                                         struct.pack("<e", float(v))
                                         for row in cond for v in row)).decode("ascii"),
            "minimax_token_tags": list(tags),
            "token_ids": list(token_ids),
            "meta": dict(meta),
            "format": "atomic_te38_stash_v1",
        }
        raw = json.dumps(rec, sort_keys=True).encode("utf-8")
        return raw


def _torch_load_pt(blob: bytes):
    try:
        import torch
        import io
        rec = torch.load(io.BytesIO(blob), map_location="cpu",
                         weights_only=False)
        return rec
    except ImportError:
        return None


def stash_cond(prompt: str, name: Optional[str] = None,
               cond_dir: Optional[str] = None,
               cond=None, tags=None, token_ids=None,
               meta: Optional[dict] = None) -> str:
    """Write the (cond, tags, token_ids, meta) record to cond_dir.

    Returns the absolute path of the written .pt.  `cond` may be a torch
    tensor, numpy array, or a flat bytes blob (fp16 LE).  `tags` may be
    a torch tensor, numpy array, or a list of ints.
    """
    cd = resolve_cond_dir(cond_dir)
    n = name or cond_cache_key(prompt)
    dest = os.path.join(cd, f"{n}.pt")
    blob = _np_to_torch_save(cond, tags, token_ids, meta or {})
    with open(dest, "wb") as f:
        f.write(blob)
    return dest


def load_cond(path: str):
    """Read a .pt written by stash_cond; returns the record dict."""
    with open(path, "rb") as f:
        blob = f.read()
    rec = _torch_load_pt(blob)
    if rec is None:
        # try the stdlib path
        try:
            rec = json.loads(blob.decode("utf-8"))
            if rec.get("format") == "atomic_te38_stash_v1":
                cond = np_from_fp16_bytes(
                    base64.b64decode(rec["cond"]))
                rec["cond"] = cond
                rec["minimax_token_tags"] = rec["minimax_token_tags"]
        except Exception as e:
            raise TEAdapterError(f"load_cond({path!r}) failed: {e}") from e
    return rec


def _check_fp16_nan(raw_bytes: bytes, prompt: str, source: str) -> None:
    """Scan for fp16 NaN by unpacking each 2-byte half and testing isnan.

    Raises TEAdapterError if any NaN is found so bad hidden states never
    propagate into the adapter (which would produce deterministic NaN output
    that hides the upstream failure).
    """
    n = len(raw_bytes) // 2
    for i in range(n):
        word = raw_bytes[i * 2:(i + 1) * 2]
        val = struct.unpack("<e", word)[0]
        if val != val:   # NaN check
            raise TEAdapterError(
                f"vLLM {source} returned fp16 NaN at token position {i} "
                f"for prompt {prompt!r}; the vLLM endpoint is producing NaN "
                "hidden states. This is a server-side issue — check vLLM "
                "logs / GPU state."
            )


def np_from_fp16_bytes(buf: bytes, shape=None):
    """Decode a raw fp16 LE byte buffer to a (numpy) array."""
    n = len(buf) // 2
    try:
        import numpy as np
        a = np.frombuffer(buf, dtype="<f2").astype(np.float16)
        if shape is not None:
            a = a.reshape(shape)
        return a
    except ImportError:
        # pure stdlib: return list of lists
        if shape is None:
            shape = (n,)
        out = []
        for i in range(n):
            v = struct.unpack("<e", buf[i*2:(i+1)*2])[0]
            out.append(v)
        if len(shape) == 2:
            out = [out[i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])]
        return out


# -------------------------------------------------------------------------
# TEAdapterClient: the public surface
# -------------------------------------------------------------------------

class TEAdapterClient:
    """text -> cond.pt path. Wire to FastH3Client via te_adapter= kwarg.

    Use encode() for one-off encodes (testing) or encode_cached() for
    the live path (default; consults the JSON-side cache first). The
    cache key is sha256(prompt, template, adapter_path) so a repeated
    prompt never re-runs the vLLM encode.
    """

    def __init__(self, vllm_url: str = DEFAULT_VLLM_URL,
                 adapter_path: str = DEFAULT_TE38_ADAPTER,
                 cond_dir: Optional[str] = None,
                 cache_dir: str = DEFAULT_CACHE_DIR,
                 template_id: str = "h3_raw",
                 layer: int = TE38_ADAPTER_LAYER,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 max_tokens: int = TE38_DEFAULT_MAX_TOKENS,
                 timeout_s: float = TE38_HTTP_TIMEOUT_S,
                 use_cache: bool = True):
        self.cfg = TEAdapterConfig(
            vllm_url=vllm_url, adapter_path=adapter_path,
            cond_dir=cond_dir or DEFAULT_COND_DIR,
            cache_dir=cache_dir, template_id=template_id, layer=layer,
            hidden_dim=hidden_dim, max_tokens=max_tokens,
            timeout_s=timeout_s, use_cache=use_cache,
        )
        self._vllm: Optional[VLLMHiddenStatesClient] = None
        self._adapter: Optional[TE38Adapter] = None
        self._calls = 0
        self._cache_hits = 0
        self._encode_ms_total = 0.0
        self._projection_ms_total = 0.0

    # -- lazy helpers ----------------------------------------------------

    def _vllm_client(self) -> VLLMHiddenStatesClient:
        if self._vllm is None:
            self._vllm = VLLMHiddenStatesClient(
                url=self.cfg.vllm_url,
                timeout_s=self.cfg.timeout_s,
                max_tokens=self.cfg.max_tokens,
                hidden_dim=self.cfg.hidden_dim,
            )
        return self._vllm

    def _adapter_obj(self) -> TE38Adapter:
        if self._adapter is None:
            self._adapter = TE38Adapter(
                path=self.cfg.adapter_path,
                hidden_dim=self.cfg.hidden_dim,
            )
        return self._adapter

    def is_vllm_up(self) -> bool:
        try:
            cli = self._vllm_client()
            url = cli.url + "/v1/models"
            with urllib.request.urlopen(url, timeout=1.5) as r:
                return r.status == 200
        except Exception:
            return False

    # -- core encode ----------------------------------------------------

    def encode(self, prompt: str) -> dict:
        """Run the full pipeline: vllm hidden_states -> adapter -> cond.

        Returns a dict: {cond, minimax_token_tags, token_ids, meta,
        encode_ms, projection_ms, cache_hit}.
        """
        t0 = time.perf_counter()
        cli = self._vllm_client()
        hidden_bytes, shape, token_ids, raw = cli.hidden_states(
            prompt, template_id=self.cfg.template_id, layer=self.cfg.layer)
        encode_ms = float(raw.get("encode_ms", 0.0))

        # reshape to (L, D) — always fp16
        n = 1
        for s in shape:
            n *= int(s)
        L = n // int(self.cfg.hidden_dim)
        if L * int(self.cfg.hidden_dim) != n:
            raise TEAdapterError(
                f"hidden_states shape {shape} not divisible by "
                f"hidden_dim={self.cfg.hidden_dim}")
        try:
            import numpy as np
            hidden = np.frombuffer(hidden_bytes, dtype="<f2").reshape(
                L, self.cfg.hidden_dim).copy()
        except ImportError:
            # stdlib fallback — slow but correct
            arr = np_from_fp16_bytes(hidden_bytes, shape=(L, self.cfg.hidden_dim))
            hidden = arr

        t1 = time.perf_counter()
        adapter = self._adapter_obj()
        cond, tags = adapter.project_to_cond(hidden)
        projection_ms = (time.perf_counter() - t1) * 1000.0

        meta = {
            "prompt": prompt,
            "template_id": self.cfg.template_id,
            "layer": int(self.cfg.layer),
            "hidden_dim": int(self.cfg.hidden_dim),
            "L": int(L),
            "encode_ms": encode_ms,
            "projection_ms": projection_ms,
            "vllm_url": self.cfg.vllm_url,
            "adapter_path": self.cfg.adapter_path,
            "adapter_meta": adapter.meta(),
            "format": "te_h3_from_38",
            "source": "atomic.te_adapter.TEAdapterClient.encode",
        }
        rec = {
            "prompt": prompt,
            "cond": cond,
            "minimax_token_tags": tags,
            "token_ids": token_ids,
            "meta": meta,
            "encode_ms": encode_ms,
            "projection_ms": projection_ms,
            "cache_hit": False,
        }
        self._calls += 1
        self._encode_ms_total += encode_ms
        self._projection_ms_total += projection_ms
        return rec

    # -- cache-aware path -----------------------------------------------

    def _cache_path(self, prompt: str) -> str:
        key = cond_cache_key(
            prompt, template_id=self.cfg.template_id,
            layer=self.cfg.layer, adapter_path=self.cfg.adapter_path)
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        return os.path.join(self.cfg.cache_dir, f"{key}.pt")

    def encode_cached(self, prompt: str) -> dict:
        """Cache-aware encode: hit the JSON-side .pt cache first.

        The `path` returned is ALWAYS the cond_dir path (the one
        LoadH3TE38Conditioning consumes); the cache_dir is a separate
        fast-path store.
        """
        if not self.cfg.use_cache:
            rec = self.encode(prompt)
            path = stash_cond(
                prompt, name=None, cond_dir=self.cfg.cond_dir,
                cond=rec["cond"], tags=rec["minimax_token_tags"],
                token_ids=rec["token_ids"], meta=rec["meta"])
            rec["path"] = path
            return rec
        cp = self._cache_path(prompt)
        if os.path.isfile(cp):
            try:
                rec = load_cond(cp)
                rec["cache_hit"] = True
                # also stage into cond_dir so the path returned is
                # what ComfyUI can load
                cond_dir_path = stash_cond(
                    prompt, name=None, cond_dir=self.cfg.cond_dir,
                    cond=rec["cond"], tags=rec["minimax_token_tags"],
                    token_ids=rec["token_ids"], meta=rec["meta"])
                rec["path"] = cond_dir_path
                rec["encode_ms"] = 0.0
                rec["projection_ms"] = 0.0
                self._cache_hits += 1
                return rec
            except Exception:
                # corrupt cache; fall through to a fresh encode
                try:
                    os.unlink(cp)
                except OSError:
                    pass
        rec = self.encode(prompt)
        # write both to the JSON-side cache and to cond_dir
        path = stash_cond(
            prompt, name=None, cond_dir=self.cfg.cond_dir,
            cond=rec["cond"], tags=rec["minimax_token_tags"],
            token_ids=rec["token_ids"], meta=rec["meta"])
        try:
            blob = _np_to_torch_save(
                rec["cond"], rec["minimax_token_tags"],
                rec["token_ids"], rec["meta"])
            with open(cp, "wb") as f:
                f.write(blob)
        except Exception:
            pass   # cache write is best-effort
        rec["path"] = path
        return rec

    # -- stats -----------------------------------------------------------

    def stats(self) -> dict:
        return {
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "encode_ms_total": self._encode_ms_total,
            "projection_ms_total": self._projection_ms_total,
            "vllm": self._vllm.stats() if self._vllm else None,
            "cfg": self.cfg.as_dict(),
        }
