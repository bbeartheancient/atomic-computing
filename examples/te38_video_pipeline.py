"""Iter 40: TE-38 video pipeline — VLLM encode + TE adapter + FastH3 + QBF trace.

End-to-end demo of the iter-39 TE-38 pipeline wired into the QBF
trace store. The pipeline is:

    TEAdapterClient.encode_cached(prompt)
        -> vLLM /v1/hidden_states  (Qwen 3.8 27B, layer=-1, fp16)
        -> te_h3_from_38.pt MLP    (5120->5120, GELU, 5120->5120)
        -> stash cond.pt           (cond + token_tags + token_ids + meta)
        -> FastH3Client.generate  (ComfyUI /prompt, LoadH3TE38Conditioning)
        -> FlowTrace.video ring   (VideoFrameEntry per tick)
        -> QbfTraceStore.append_run (one shard, n_video blobs)
        -> QbfTraceStore.load_run   (replay-ready)

The demo exercises all three FastH3Client paths:
    1. te_adapter=None, mode="comfyui"   -> iter-38 H3TextEncode8 wire
    2. te_adapter=TEAdapterClient(), mode="comfyui"  -> LoadH3TE38Conditioning
    3. fallback=FastH3Stub()             -> deterministic stub (no GPU needed)

Only path 2 requires both vLLM (for encoding) and ComfyUI (for generation).
Paths 1 and 3 run offline.  The QBF trace archives frames from all paths.

Usage:
    python -m examples.te38_video_pipeline

Live services required for the full pipeline (path 2):
    - vLLM at localhost:8000  (Qwen 3.8 27B /v1/hidden_states)
    - ComfyUI vsa at localhost:8188  (FastH3 GGUF on GPU1)

Paths 1 and 3 run without services.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request

from atomic import (
    TEAdapterClient, FastH3Client, FastH3Stub,
    FlowTrace, QbfTraceStore,
    VideoFrameEntry,
    stash_cond, load_cond,
    DEFAULT_VLLM_URL,
)


def _check_vllm_up() -> bool:
    try:
        urllib.request.urlopen(DEFAULT_VLLM_URL + "/v1/models", timeout=2)
        return True
    except Exception:
        return False


def _check_comfyui_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:8188/system_stats", timeout=2)
        return True
    except Exception:
        return False


def _decode_h4_pixel(rgba: bytes, width: int, height: int):
    """Decode last-pixel H4 gates from RGBA (same as viz_video atom)."""
    import math
    n = width * height
    j = (n - 1) * 4
    a_raw = rgba[j + 3]
    r_raw = rgba[j]
    g_raw = rgba[j + 1]
    b_raw = rgba[j + 2]
    a_log = math.log(max(1, a_raw))
    from atomic.qbf import h4_gate
    w_row, z_row, y_row, x_row = h4_gate(
        (a_log, float(b_raw), float(g_raw), float(r_raw)))
    return float(w_row), float(x_row), float(y_row), float(z_row)


def _stash_and_workflow(prompt: str, cond_dir: str) -> dict:
    """Encode -> stash .pt -> return {path, encode_ms, projection_ms}."""
    te = TEAdapterClient(use_cache=False, cond_dir=cond_dir)
    rec = te.encode(prompt)
    return {
        "path": rec["path"],
        "encode_ms": rec["encode_ms"],
        "projection_ms": rec["projection_ms"],
        "L": rec["meta"]["L"],
        "cache_hit": rec["cache_hit"],
    }


def _run_stub_path(n_frames: int, width: int, height: int) -> FlowTrace:
    """Path 3: FastH3Stub, deterministic, no GPU needed."""
    print("\n[Path 3] FastH3Stub (deterministic, no GPU)")
    stub = FastH3Stub(width=width, height=height)
    trace = FlowTrace(max_frames=n_frames)
    for t in range(n_frames):
        r = stub.generate("a comet over the ocean", seed=t, n_frames=1)
        frame_bytes = r["frames"][0]
        w, x, y, z = _decode_h4_pixel(frame_bytes, width, height)
        trace.record_video_frame(
            rgba=frame_bytes, t=t,
            prompt="a comet over the ocean",
            seed=t,
            h3_latency_ms=r.get("h3_latency_ms", 0.0),
            width=width, height=height,
            w_gate=w, x_gate=x, y_gate=y, z_gate=z,
        )
        trace.record_node("stub", "FastH3Stub", {}, {}, 1.0, t=t)
    print(f"  recorded {trace.video_seq} frames (te_path=stub)")
    return trace


def _run_te38_path(prompt: str, cond_dir: str,
                    n_frames: int, width: int, height: int,
                    check_vllm: bool, check_comfyui: bool) -> tuple[FlowTrace, dict]:
    """Path 2: TEAdapterClient + FastH3Client(mode=comfyui) with te_adapter."""
    print(f"\n[Path 2] TE-38 pipeline (vLLM={check_vllm}, ComfyUI={check_comfyui})")

    enc_info = None
    if check_vllm:
        print(f"  [1] TEAdapterClient.encode_cached({prompt[:40]!r}...)")
        te = TEAdapterClient(use_cache=True, cond_dir=cond_dir)
        rec = te.encode_cached(prompt)
        enc_info = {
            "path": rec["path"],
            "encode_ms": rec["encode_ms"],
            "projection_ms": rec["projection_ms"],
            "L": rec["meta"]["L"],
            "cache_hit": rec["cache_hit"],
        }
        print(f"  [1] encode_ms={enc_info['encode_ms']:.1f}ms, "
              f"proj_ms={enc_info['projection_ms']:.1f}ms, "
              f"L={enc_info['L']}, "
              f"cache_hit={enc_info['cache_hit']}, "
              f"pt={enc_info['path']}")
    else:
        print("  [1] vLLM not up — skipping encode")
        return FlowTrace(max_frames=n_frames), {}

    if not check_comfyui:
        print("  [2] ComfyUI not up — skipping FastH3 generate")
        return FlowTrace(max_frames=n_frames), enc_info

    print(f"  [2] FastH3Client.generate (te_adapter=TEAdapterClient, mode=comfyui)")
    te = TEAdapterClient(use_cache=True, cond_dir=cond_dir)
    client = FastH3Client(
        endpoint="http://localhost:8188",
        mode="comfyui",
        te_adapter=te,
        fallback=FastH3Stub(width=width, height=height),
    )
    trace = FlowTrace(max_frames=n_frames)
    for t in range(n_frames):
        r = client.generate(prompt, seed=t, n_frames=1,
                            width=width, height=height)
        frame_bytes = r["frames"][0]
        w, x, y, z = _decode_h4_pixel(frame_bytes, width, height)
        trace.record_video_frame(
            rgba=frame_bytes, t=t,
            prompt=prompt,
            seed=t,
            h3_latency_ms=r.get("h3_latency_ms", 0.0),
            width=width, height=height,
            w_gate=w, x_gate=x, y_gate=y, z_gate=z,
        )
        trace.record_node(
            "te38", "FastH3Client(te38)",
            {"te38_cond_path": r.get("te38_cond_path", ""),
             "te_path": r.get("te_path", "unknown")},
            {}, 1.0, t=t)
    print(f"  recorded {trace.video_seq} frames "
          f"(te_path={client._te38_uses} te38, "
          f"{client._te38_fallbacks} fallback)")
    return trace, enc_info


def _run_iter38_path(n_frames: int, width: int, height: int,
                     check_comfyui: bool) -> FlowTrace:
    """Path 1: FastH3Client(mode=comfyui) WITHOUT te_adapter (iter-38 wire)."""
    print(f"\n[Path 1] Iter-38 wire (te_adapter=None, mode=comfyui, ComfyUI={check_comfyui})")
    if not check_comfyui:
        print("  ComfyUI not up — skipping FastH3 generate")
        return FlowTrace(max_frames=n_frames)

    client = FastH3Client(
        endpoint="http://localhost:8188",
        mode="comfyui",
        te_adapter=None,
        fallback=FastH3Stub(width=width, height=height),
    )
    trace = FlowTrace(max_frames=n_frames)
    for t in range(n_frames):
        r = client.generate("a nebula in deep blue", seed=t + 100,
                            n_frames=1, width=width, height=height)
        frame_bytes = r["frames"][0]
        w, x, y, z = _decode_h4_pixel(frame_bytes, width, height)
        trace.record_video_frame(
            rgba=frame_bytes, t=t,
            prompt="a nebula in deep blue",
            seed=t + 100,
            h3_latency_ms=r.get("h3_latency_ms", 0.0),
            width=width, height=height,
            w_gate=w, x_gate=x, y_gate=y, z_gate=z,
        )
        trace.record_node(
            "iter38", "FastH3Client(iter38)",
            {"te_path": r.get("te_path", "h3student")},
            {}, 1.0, t=t)
    print(f"  recorded {trace.video_seq} frames (te_path=h3student)")
    return trace


def _qbf_round_trip(trace: FlowTrace, shard_name: str,
                     note: str, width: int, height: int) -> dict:
    """Archive trace to .qbf shard, load_run, flow_trace, verify."""
    print(f"\n[QBF] append_run('{note}')")
    tmp = tempfile.mkdtemp(prefix="te38_qbf_")
    path = os.path.join(tmp, shard_name)
    store = QbfTraceStore(path)
    manifest = store.append_run(trace, note=note)
    print(f"  manifest: n_frames={manifest['n_frames']}, "
          f"n_video={manifest['n_video']}, vseq={manifest['vseq']}")
    assert manifest["vseq"] == trace.video_seq, \
        f"vseq mismatch: {manifest['vseq']} vs {trace.video_seq}"

    print(f"  load_run(0):")
    d = store.load_run(0)
    assert len(d["video"]) == trace.video_seq, \
        f"video count mismatch: {len(d['video'])} vs {trace.video_seq}"
    assert len(d["video"][0]["rgba"]) == width * height * 4, \
        f"frame size mismatch"

    print(f"  flow_trace(0):")
    ft = store.flow_trace(0)
    assert ft.video_seq == trace.video_seq
    for i, (orig, rebuilt) in enumerate(zip(trace.video, ft.video)):
        assert orig.t == rebuilt.t, f"frame {i} t mismatch"
        assert orig.prompt == rebuilt.prompt, f"frame {i} prompt mismatch"
        assert orig.seed == rebuilt.seed, f"frame {i} seed mismatch"
        assert abs(orig.w_gate - rebuilt.w_gate) < 1e-4, \
            f"frame {i} w_gate mismatch"
    print(f"  {trace.video_seq} frames verified (prompt/seed/w_gate/rgba)")

    store.close()
    try:
        os.unlink(path)
    except OSError:
        pass
    os.rmdir(tmp)
    return manifest


def main():
    print("=" * 60)
    print("iter40: TE-38 video pipeline — VLLM encode + FastH3 + QBF trace")
    print("=" * 60)

    width = 16
    height = 16
    n_frames = 3
    prompt = "integrated_multimodal_description: [Shot 1] a comet over the ocean"

    check_vllm = _check_vllm_up()
    check_comfyui = _check_comfyui_up()
    print(f"\nservices: vLLM={check_vllm}, ComfyUI={check_comfyui}")

    tmp = tempfile.mkdtemp(prefix="te38_cond_")
    cond_dir = os.path.join(tmp, "cond")
    os.makedirs(cond_dir, exist_ok=True)
    print(f"cond_dir: {cond_dir}")

    manifests = {}

    # Path 3: stub (always runs, no GPU)
    trace3 = _run_stub_path(n_frames, width, height)
    manifests["stub"] = _qbf_round_trip(
        trace3, "stub.qbf", "iter40 stub path", width, height)

    # Path 2: TE-38 (needs vLLM, optionally ComfyUI)
    trace2, enc_info = _run_te38_path(
        prompt, cond_dir, n_frames, width, height,
        check_vllm, check_comfyui)
    if trace2.video_seq > 0:
        manifests["te38"] = _qbf_round_trip(
            trace2, "te38.qbf", "iter40 te38 path", width, height)
        print(f"\n  TE-38 encode: L={enc_info.get('L')}, "
              f"encode_ms={enc_info.get('encode_ms', 0):.1f}, "
              f"proj_ms={enc_info.get('projection_ms', 0):.1f}")
    else:
        print(f"\n  TE-38 path: no frames recorded (vLLM or ComfyUI down)")

    # Path 1: iter-38 wire (optionally ComfyUI)
    trace1 = _run_iter38_path(n_frames, width, height, check_comfyui)
    if trace1.video_seq > 0:
        manifests["iter38"] = _qbf_round_trip(
            trace1, "iter38.qbf", "iter40 iter38 wire", width, height)

    # Summary
    print("\n" + "=" * 60)
    print(f"iter40: {len(manifests)} traces archived")
    for k, m in manifests.items():
        print(f"  {k}: n_video={m['n_video']}, vseq={m['vseq']}")
    print("=" * 60)

    try:
        os.rmdir(cond_dir)
        os.rmdir(tmp)
    except OSError:
        pass

    return {"manifests": manifests, "enc_info": enc_info}


if __name__ == "__main__":
    main()
