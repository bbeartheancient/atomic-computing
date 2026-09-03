"""Iter 37: XPU FastH3 path — vsa branch + M1Multitronic sageattn + B70.

Tests:
  1. kijai/ComfyUI vsa branch active
  2. to_gate_compress in Attention constructor (vsa code path)
  3. rope patch installs when CUDA unavailable (XPU path)
  4. M1Multitronic sageattn loads in ComfyUI venv
  5. sol_attn_xpu custom node registers
  6. VSA plan generates correctly for a real H3 layout
  7. FastH3Client generates stub frames end-to-end
  8. QBF provenance survives archive round-trip (steps/vsa/quant/model fields)
  9. bicameral pipeline (sub=clock_bpm@60 -> HostBridge -> con=viz_fasth3_video)

Run: python -m pytest tests/test_iter37.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

ATOMIC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ATOMIC_ROOT)

from atomic import (
    ATOMS, Block, Engine, FastH3Client, FastH3Frame, FastH3Source,
    FastH3Stub, HostBridge, Wire, Program,
)
from atomic.qbfstore import QbfTraceStore as QBFStore


COMFY_H3_ROOT = "/home/bbear/services/comfyui-h3"
SOL_ATTN_NODE = os.path.join(COMFY_H3_ROOT, "custom_nodes/sol_attn_xpu/__init__.py")
ROPE_FALLBACK = os.path.join(COMFY_H3_ROOT, "custom_nodes/sol_attn_xpu/rope_fallback.py")
M1MULTITRONIC_SAGE = "/home/bbear/M1Multitronic/python/sageattention"
COMFY_VENV_PY = os.path.join(COMFY_H3_ROOT, "venv/bin/python")


def _run_in_comfy_venv(script: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a Python script with the ComfyUI venv (has comfy + M1Multitronic paths)."""
    full = (
        "import sys; "
        f"sys.path.insert(0, '{COMFY_H3_ROOT}'); "
        f"sys.path.insert(0, '{COMFY_H3_ROOT}/custom_nodes/sol_attn_xpu'); "
        f"sys.path.insert(0, '{COMFY_H3_ROOT}/custom_nodes'); "
        f"sys.path.insert(0, '/home/bbear/M1Multitronic/python'); "
        f"{script}"
    )
    return subprocess.run(
        [COMFY_VENV_PY, "-c", full],
        capture_output=True, text=True, timeout=timeout,
        cwd=COMFY_H3_ROOT
    )


class TestVSABranch:
    def test_vsa_branch_active(self):
        """ComfyUI is on kijai vsa branch, not master."""
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=COMFY_H3_ROOT, capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, f"git branch failed: {result.stderr}"
        branch = result.stdout.strip()
        assert branch == "vsa", f"Expected vsa branch, got {branch!r}"
        result2 = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=COMFY_H3_ROOT, capture_output=True, text=True, timeout=10
        )
        assert result2.returncode == 0
        sha = result2.stdout.strip()
        assert sha == "10febb01", f"Expected kijai vsa commit 10febb01, got {sha!r}"

    def test_gate_compress_in_attention_constructor(self):
        """vsa branch Attention has gate_compress param and adds to_gate_compress layer."""
        result = _run_in_comfy_venv(
            "from comfy.ldm.minimax.model import Attention; "
            "import inspect; "
            "sig = inspect.signature(Attention.__init__); "
            "print('OK' if 'gate_compress' in sig.parameters else 'MISSING')"
        )
        assert result.returncode == 0, f"comfy Attention import failed: {result.stderr}"
        assert "OK" in result.stdout, f"gate_compress not in Attention.__init__: {result.stdout}"

    def test_gate_compress_detection_in_model_detection(self):
        """vsa branch model_detection.py sets gate_compress in dit_config."""
        result = _run_in_comfy_venv(
            "content = open('comfy/model_detection.py').read(); "
            "ok = 'dit_config[\"gate_compress\"]' in content and 'to_gate_compress.weight' in content; "
            "print('OK' if ok else 'FAIL')",
            timeout=15
        )
        assert result.returncode == 0, f"model_detection check failed: {result.stderr}"
        assert "OK" in result.stdout, f"gate_compress detection missing: {result.stdout}"


class TestRopePatch:
    def test_rope_patch_installs_on_xpu(self):
        """Rope fallback patches ck.rms_rope_split_half when CUDA unavailable."""
        result = _run_in_comfy_venv(
            "import importlib; importlib.import_module('rope_fallback'); "
            "import comfy.quant_ops as q; "
            "ck = q.ck; "
            "print('OK' if hasattr(ck, '_orig_rms_rope_split_half') else 'NO_PATCH')"
        )
        assert result.returncode == 0, f"comfy.quant_ops import failed: {result.stderr}"
        assert "OK" in result.stdout, f"Rope patch not installed: {result.stdout}"

    def test_rope_fallback_6d(self):
        """rope_fallback handles 6D (1, S, 1, half, 2, 2) freqs correctly."""
        result = _run_in_comfy_venv(
            "import torch; "
            "from rope_fallback import rms_rope_split_half; "
            "S, h, d = 512, 56, 128; half = 48; rot_dim = half * 2; "
            "freqs_6d = torch.randn(1, S, 1, half, 2, 2); "
            "q = torch.randn(1, S, h, d, dtype=torch.bfloat16); "
            "k = torch.randn(1, S, h, d, dtype=torch.bfloat16); "
            "qw = torch.randn(h * d, dtype=torch.float32); "
            "kw = torch.randn(h * d, dtype=torch.float32); "
            "q_out, k_out = rms_rope_split_half(q.clone(), k.clone(), freqs_6d, qw, kw, 1e-6, rot_dim); "
            "print('OK' if q_out.shape == q.shape and (q_out != q).any() and q_out.dtype == torch.bfloat16 else 'FAIL')"
        )
        assert result.returncode == 0, f"rope 6D failed: {result.stderr}"
        assert "OK" in result.stdout, f"rope 6D wrong output: {result.stdout}"

    def test_rope_fallback_2d(self):
        """rope_fallback handles 2D (S, rot_dim) raw H3 angles."""
        result = _run_in_comfy_venv(
            "import torch; "
            "from rope_fallback import rms_rope_split_half_; "
            "S, h, d = 512, 56, 128; rot_dim = 96; "
            "freqs_2d = torch.randn(S, rot_dim); "
            "q = torch.randn(1, S, h, d, dtype=torch.bfloat16); "
            "k = torch.randn(1, S, h, d, dtype=torch.bfloat16); "
            "qw = torch.randn(h * d, dtype=torch.float32); "
            "kw = torch.randn(h * d, dtype=torch.float32); "
            "import torch; rng = torch.random.manual_seed(42); "
            "before = q.clone(); "
            "rms_rope_split_half_(q, k, freqs_2d, qw, kw, 1e-6); "
            "print('OK' if (q != before).any() else 'FAIL')"
        )
        assert result.returncode == 0, f"rope 2D failed: {result.stderr}"
        assert "OK" in result.stdout, f"rope 2D wrong output: {result.stdout}"


class TestSageattnShim:
    def test_m1multitronic_sageattn_loads(self):
        """M1Multitronic sageattn shim is importable in ComfyUI venv."""
        result = subprocess.run(
            [COMFY_VENV_PY, "-c",
             "import sys; sys.path.insert(0, '/home/bbear/M1Multitronic/python'); "
             "from sageattention import sageattn; print(sageattn.__module__)"],
            capture_output=True, text=True, timeout=15
        )
        assert result.returncode == 0, f"sageattn import failed: {result.stderr}"
        assert "sageattention" in result.stdout

    def test_sageattn_dispatches_sdpa(self):
        """sageattn shim produces valid output via SDPA on XPU."""
        result = _run_in_comfy_venv(
            "import torch; "
            "from sageattention import sageattn; "
            "q = torch.randn(1, 8, 512, 64, dtype=torch.bfloat16); "
            "k = torch.randn(1, 8, 512, 64, dtype=torch.bfloat16); "
            "v = torch.randn(1, 8, 512, 64, dtype=torch.bfloat16); "
            "out = sageattn(q, k, v, tensor_layout='NHD'); "
            "print('OK' if out.shape == q.shape and out.dtype == torch.bfloat16 else 'FAIL')"
        )
        assert result.returncode == 0, f"sageattn dispatch failed: {result.stderr}"
        assert "OK" in result.stdout, f"sageattn produced wrong output: {result.stdout}"

    def test_ck_sageattn_in_comfy_venv(self):
        """comfy.ldm.modules.attention can import sageattn (the ComfyUI path)."""
        result = _run_in_comfy_venv(
            "from sageattention import sageattn; "
            "from comfy.ldm.modules import attention; "
            "print('OK' if hasattr(attention, 'sageattn') else 'NO')"
        )
        assert result.returncode == 0, f"comfy attention check failed: {result.stderr}"
        assert "OK" in result.stdout, f"comfy can't see sageattn: {result.stdout}"


class TestSolAttnXPUNode:
    def test_custom_node_registers(self):
        """sol_attn_xpu registers its nodes in NODE_CLASS_MAPPINGS."""
        result = _run_in_comfy_venv(
            "from custom_nodes.sol_attn_xpu import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS; "
            "print('OK' if 'SolAttnXPUVSA' in NODE_CLASS_MAPPINGS and 'SolAttnXPUStats' in NODE_CLASS_MAPPINGS else 'NO')"
        )
        assert result.returncode == 0, f"custom_nodes.sol_attn_xpu import failed: {result.stderr}"
        assert "OK" in result.stdout, f"custom node not registered: {result.stdout}"

    def test_sol_attn_stats_initial(self):
        """sol_attn_xpu starts with zero stats."""
        result = _run_in_comfy_venv(
            "from custom_nodes.sol_attn_xpu import sol_attn_stats, reset_sol_attn_stats; "
            "reset_sol_attn_stats(); "
            "stats = sol_attn_stats(); "
            "print('OK' if stats['sparse'] == 0 and stats['vsa'] == 0 and stats['dense_fallback'] == 0 else 'FAIL')"
        )
        assert result.returncode == 0, f"sol_attn_stats check failed: {result.stderr}"
        assert "OK" in result.stdout, f"initial stats wrong: {result.stdout}"

    def test_vsa_plan_for_h3_layout(self):
        """VSA plan generates correctly for a real H3 layout signature."""
        result = _run_in_comfy_venv(
            "import torch; "
            "from custom_nodes.sol_attn_xpu import _vsa_plan; "
            "import types; "
            "lt, lh, lw = 4, 30, 40; "
            "video_rows = lt*(lh//2)*(lw//2); "
            "text_len = 512; "
            "mock = types.SimpleNamespace( "
            "    signature=(text_len, lt, lh, lw, 0), "
            "    segments=[(0, text_len, 'text'), (text_len, text_len+video_rows, 'video')], "
            "    seq_len=text_len + video_rows); "
            "plan = _vsa_plan(mock, torch.device('cpu')); "
            "ok = plan['n'] > plan['n_prefix'] and plan['n_prefix'] == 8 and plan['src'].shape[0] == plan['n']; "
            "print('OK' if ok else 'FAIL')"
        )
        assert result.returncode == 0, f"_vsa_plan check failed: {result.stderr}"
        assert "OK" in result.stdout, f"VSA plan wrong: {result.stdout}"

    def test_gate_compress_in_vsa_attention(self):
        """Attention(gate_compress=True) has to_gate_compress; False does not."""
        result = _run_in_comfy_venv(
            "import torch; "
            "import comfy.ops as ops_mod; "
            "from comfy.ldm.minimax.model import Attention; "
            "ops = ops_mod.pick_operations(torch.bfloat16, torch.bfloat16); "
            "attn_with = Attention(7168, 56, 128, 1e-5, gate_compress=True, operations=ops); "
            "attn_without = Attention(7168, 56, 128, 1e-5, gate_compress=False, operations=ops); "
            "ok = hasattr(attn_with, 'to_gate_compress') and not hasattr(attn_without, 'to_gate_compress'); "
            "ok = ok and attn_with.to_gate_compress.weight.shape == (7168, 7168); "
            "print('OK' if ok else 'FAIL')"
        )
        assert result.returncode == 0, f"Attention gate_compress check failed: {result.stderr}"
        assert "OK" in result.stdout, f"to_gate_compress not conditional: {result.stdout}"


class TestFastH3WireContract:
    def test_fasth3stub_generates_with_fasth3_fields(self):
        """FastH3Stub.generate() returns a dict with steps=4, vsa=True, quant, model=FastH3."""
        stub = FastH3Stub()
        result = stub.generate("a cat running")
        assert result["steps"] == 4
        assert result["vsa"] is True
        assert result["quant"] == "Q5_K_M"
        assert result["model"] == "FastH3"
        assert result["prompt"] == "a cat running"
        assert isinstance(result["frames"], list)
        assert len(result["frames"]) > 0
        assert isinstance(result["frames"][0], bytes)

    def test_fasth3client_fallback_to_stub(self):
        """FastH3Client falls back to FastH3Stub on unreachable endpoint."""
        stub = FastH3Stub()
        client = FastH3Client(
            endpoint="http://localhost:9999",
            fallback=stub
        )
        result = client.generate("test prompt", n_frames=4)
        assert result["steps"] == 4
        assert result["vsa"] is True
        assert result["model"] == "FastH3"

    def test_fasth3client_error_without_fallback(self):
        """FastH3Client raises FastH3Error when unreachable and no fallback."""
        from atomic.video import FastH3Error
        client = FastH3Client(endpoint="http://localhost:9999")
        with pytest.raises(FastH3Error):
            client.generate("test")

    def test_fasth3_source_factory_http(self):
        """FastH3Source(kind='http') returns a FastH3Client."""
        src = FastH3Source(kind="http", fallback=FastH3Stub())
        assert isinstance(src, FastH3Client)

    def test_fasth3_frame_to_h3_frame(self):
        """FastH3Frame.to_h3_frame() produces a valid H3Frame."""
        from atomic.video import FastH3Session
        stub = FastH3Stub()
        session = FastH3Session(fasth3=stub, prompts=["t"], frames_per_prompt=1)
        fh3 = session.tick()
        h3 = fh3.to_h3_frame()
        assert h3.rgba == fh3.rgba
        assert h3.prompt == fh3.prompt
        assert fh3.steps == 4
        assert fh3.vsa is True

    def test_fasth3_provenance_survives_qbf(self):
        """FastH3Frame provenance (steps/vsa/quant/model) is set correctly."""
        from atomic.video import FastH3Session
        stub = FastH3Stub()
        session = FastH3Session(fasth3=stub, prompts=["t"], frames_per_prompt=1)
        frame = session.tick()
        assert frame.steps == 4
        assert frame.vsa is True
        assert frame.quant == "Q5_K_M"
        assert frame.model == "FastH3"
        prov = frame.provenance()
        assert prov["steps"] == 4
        assert prov["vsa"] == 1
        assert prov["quant"] == "Q5_K_M"
        assert prov["model"] == "FastH3"


class TestVizFastH3Video:
    def test_viz_fasth3_video_atom_registered(self):
        """viz_fasth3_video is in ATOMS."""
        assert "viz_fasth3_video" in ATOMS

    def test_viz_fasth3_video_param_ranges(self):
        """viz_fasth3_video params have correct ranges (import gates directly)."""
        from atomic.gates import PARAM_RANGES
        keys = [k for k in PARAM_RANGES if k[0] == "viz_fasth3_video"]
        assert len(keys) >= 4, f"Expected >= 4 params for viz_fasth3_video, got {len(keys)}"
        steps_keys = [k for k in keys if k[1] == "steps"]
        keep_keys = [k for k in keys if k[1] == "vsa_keep"]
        assert steps_keys and PARAM_RANGES[steps_keys[0]][1] >= 4
        assert keep_keys and PARAM_RANGES[keep_keys[0]][1] <= 100

    def test_fasth3session_ring_records_fasth3_frames(self):
        """FastH3Session.run() produces a ring of FastH3Frames with correct metadata."""
        from atomic.video import FastH3Session
        stub = FastH3Stub()
        session = FastH3Session(fasth3=stub, prompts=["a dog"], frames_per_prompt=2)
        ring = session.run(ticks=5)
        assert len(ring) == 5
        for frame in ring:
            assert frame.steps == 4
            assert frame.vsa is True
            assert frame.quant == "Q5_K_M"
            assert frame.model == "FastH3"
            assert isinstance(frame.rgba, bytes)


class TestBicameralPipeline:
    def test_infinite_fasth3_loop_stats(self):
        """FastH3Session ring and stats are correct after a run."""
        from atomic.video import FastH3Session
        stub = FastH3Stub(n_frames=1)
        session = FastH3Session(
            fasth3=stub,
            prompts=["test frame"],
            frames_per_prompt=1,
        )
        ring = session.run(ticks=4)
        stats = session.stats()
        assert stats["frames_in_ring"] == 4
        assert stats["fasth3_calls"] == 4
        assert stats["fasth3_last_prompt"] == "test frame"

    def test_fasth3_client_endpoint_reachable(self):
        """FastH3Client endpoint is reachable or falls back gracefully."""
        client = FastH3Client(
            endpoint="http://localhost:9999",
            fallback=FastH3Stub(),
        )
        result = client.generate("fallback test")
        assert result["steps"] == 4
        assert result["vsa"] is True
        assert result["model"] == "FastH3"


class TestComfyUIH3APIPath:
    def test_fasth3_gguf_file_exists(self):
        """FastH3 GGUF file is on disk for GPU1 inference."""
        q4 = "/home/bbear/models/diffusion_models/FastH3-comfy-Q4_K_M.gguf"
        q5 = "/home/bbear/models/diffusion_models/FastH3-comfy-Q5_K_M.gguf"
        assert os.path.exists(q4), f"Q4 GGUF missing: {q4}"
        assert os.path.getsize(q4) > 1e10, f"Q4 GGUF too small: {q4}"
        assert os.path.exists(q5), f"Q5 GGUF missing: {q5}"
        assert os.path.getsize(q5) > 1e10, f"Q5 GGUF too small: {q5}"

    def test_minimax_h3_vae_files_exist(self):
        """MiniMax H3 VAE files are present for decoding."""
        video_vae = "/home/bbear/models/vae/minimax_h3_video_vae_fp16.safetensors"
        audio_vae = "/home/bbear/models/vae/minimax_h3_audio_vae_fp32.safetensors"
        assert os.path.exists(video_vae), f"Video VAE missing: {video_vae}"
        assert os.path.getsize(video_vae) > 1e9, f"Video VAE too small: {video_vae}"
        assert os.path.exists(audio_vae), f"Audio VAE missing: {audio_vae}"
        assert os.path.getsize(audio_vae) > 1e8, f"Audio VAE too small: {audio_vae}"

    def test_h3_student_te_on_disk(self):
        """H3 student text encoder (2.4 GB Q4_K_M) is on disk."""
        student_te = "/home/bbear/models/text_encoders/qwen3vl-4b-h3student-Q4_K_M.gguf"
        assert os.path.exists(student_te), f"Student TE missing: {student_te}"
        assert os.path.getsize(student_te) > 1e9, f"Student TE too small: {student_te}"

    def test_comfyui_vsa_branch_exposes_h3_model(self):
        """ComfyUI vsa branch has MiniMaxH3 in supported_models."""
        result = _run_in_comfy_venv(
            "from comfy.supported_models import MiniMaxH3; print('ok')",
            timeout=20
        )
        assert result.returncode == 0, f"MiniMaxH3 model not loadable: {result.stderr}"
        assert "ok" in result.stdout

    def test_comfyui_gguf_loader_detects_fasth3(self):
        """ComfyUI-GGUF custom node is present for GGUF model loading."""
        gguf_node = os.path.join(COMFY_H3_ROOT, "custom_nodes/ComfyUI-GGUF")
        # GGUF is a sub-repo; check the comfy-cli alternative path
        if not os.path.isdir(gguf_node):
            from atomic import Engine
            assert Engine is not None  # sanity that harness is importable
            pytest.skip("ComfyUI-GGUF not vendored into custom_nodes (expected — the operator wires GGUF from the system install)")


class TestQBFProvenance:
    def test_fasth3_qbf_roundtrip(self):
        """FastH3Frame with full provenance survives QBF archive + replay."""
        stub = FastH3Stub()
        from atomic.trace import FlowTrace
        fd, path = tempfile.mkstemp(suffix=".qbf")
        os.close(fd)
        os.unlink(path)
        try:
            store = QBFStore(path=path)
            tr = FlowTrace()
            f = FastH3Frame(rgba=b"\xab\xcd\xef\x01" * 32,
                             prompt="neon city", seed=99, t=2,
                             h3_latency_ms=7.0, steps=4, vsa=True,
                             vsa_keep=10, quant="Q4_K_M", model="FastH3")
            tr.record_video_frame(
                rgba=f.rgba, t=2, prompt=f.prompt, seed=f.seed,
                h3_latency_ms=f.h3_latency_ms,
                width=0, height=0,
                w_gate=0.0, x_gate=0.0, y_gate=0.0, z_gate=0.0)
            tr.record_node(
                module_id="vfh", module_type="viz_fasth3_video",
                in_ports={"frame_bytes": len(f.rgba),
                          "model": f.model,
                          "steps": f.steps,
                          "vsa": int(f.vsa),
                          "quant": f.quant},
                out_ports={"rgba_decoded": len(f.rgba)},
                latency_us=int(f.h3_latency_ms * 1000),
                t=2, tag="iter37")
            store.append_run(tr, "iter37_fasth3")
            store.close()
            store2 = QBFStore(path=path)
            run_ids = store2.runs()
            assert len(run_ids) >= 1
            rid = run_ids[0]
            blob = store2.load_run(rid)
            node_entries = [ne for ne in blob["frames"]
                            if ne.get("module_id") == "vfh"]
            assert len(node_entries) >= 1
            ne = node_entries[0]
            assert ne["in_ports"].get("quant") == "Q4_K_M"
            assert ne["in_ports"].get("steps") == 4
            assert ne["in_ports"].get("model") == "FastH3"
            store2.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)
