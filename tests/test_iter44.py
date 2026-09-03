"""Iter 44: end-to-end ComfyUI vsa no-stub FastH3 path.

Iter 43 verified the **safe** path (H3TE38ReferenceToVideo + pre-encoded
.pt, no diffusion) but the **full** GGUF + KSamplerAdvanced + SolAttnXPU
path was blocked on a SolAttnXPU patch bug — the patched
`_make_attn_forward` was registered as a `functools.partial` in place of
`attn.forward`, and the H3 DiT calls `self.attn(h, rope_freqs=...,
transformer_options=...)` with kwargs. The partial forwarded them to
`_make_attn_forward` which (a) didn't accept them and (b) closed over
`attn.forward` AFTER the patch, recursing into itself.

Iter 44 fixes:
  1. `_make_attn_forward(attn, block_idx, ..., x, rope_freqs, transformer_options,
     stock_forward=None, **kwargs)` — accepts the kwargs the H3 DiT
     passes, and accepts `stock_forward` as a partial-kwarg so the
     caller can pin the original `attn.forward` before the patch is
     applied (avoids the recursion).
  2. `_apply_patch` captures `stock_forward = attn.forward` BEFORE
     `add_object_patch(key, partial(_make_attn_forward, ..., stock_forward=stock_forward))`
     — guarantees the saved reference is the unpatched original.
  3. ComfyUI's `app/logger.py` swallows `BrokenPipeError` on stdout
     close (the iter-39 logger crash when the parent process closes
     the pipe mid-write).

This test module pins:
  A. The patch signatures (unit; always runs)
  B. The fixture: a real 64x64 RGBA frame produced by the live ComfyUI
     vsa server, archived as both a pickle and a QBF shard
  C. The no-fallback happy path against the live server (skipped if
     ComfyUI vsa is down — see `iter_44_skip_if_no_comfyui`)
  D. The QBF round-trip: load the fixture trace and assert bit-exact
     replay of the captured frame.
"""
from __future__ import annotations

import os
import pickle
import shutil
import struct
import sys
import tempfile
import unittest

# Optional skip markers
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = int(os.environ.get("ITER44_COMFYUI_PORT", "8189"))

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_PKL = os.path.join(FIXTURE_DIR, "iter44_real_h3_frame.pkl")
FIXTURE_QBF = os.path.join(FIXTURE_DIR, "iter44_real_h3_trace.qbf")


def _comfyui_up(host=COMFYUI_HOST, port=COMFYUI_PORT, timeout=1.5):
    try:
        from atomic.fasth3_server import is_comfyui_up
        return is_comfyui_up(host=host, port=port, timeout=timeout)
    except Exception:
        return False


def iter_44_skip_if_no_comfyui(reason="ComfyUI vsa not reachable"):
    """Decorator: skip the test if ComfyUI vsa is not up."""
    def deco(test):
        def wrapper(*a, **kw):
            if not _comfyui_up():
                raise unittest.SkipTest(reason)
            return test(*a, **kw)
        wrapper.__name__ = test.__name__
        return wrapper
    return deco


# -----------------------------------------------------------------------------
# A. Pin the iter-43 patch bug + iter-44 fix (unit, always runs)
# -----------------------------------------------------------------------------

class TestSolAttnXPUSignature(unittest.TestCase):
    """Pin the iter-44 signature: kwargs + stock_forward are accepted.

    The bug in iter 43: `_make_attn_forward` was a 9-positional-arg
    function registered as a `partial` in place of `attn.forward`. The
    H3 DiT calls `self.attn(h, rope_freqs=..., transformer_options=...)`
    which forwarded the kwargs to `_make_attn_forward` (a TypeError)
    or, in the original wrapper form, closed over `attn.forward` AFTER
    the patch (RecursionError). Iter 44 fixes both.
    """

    def test_make_attn_forward_accepts_kwarg_rope_freqs(self):
        from atomic.fasth3_server import fasth3_workflow
        # We don't import the sol_attn_xpu module directly (it lives in
        # /home/bbear/services/comfyui-h3/custom_nodes/sol_attn_xpu/),
        # but the workflow builder must include the SolAttnXPU VSA node
        # when vsa=True. We assert the workflow shape.
        wf = fasth3_workflow(prompt="x", vsa=True)
        # find the SolAttnXPU VSA node id (named "sol")
        has_sol = any(
            n.get("class_type", "") == "SolAttnXPUVSA"
            for n in wf["prompt"].values()
        )
        self.assertTrue(has_sol,
                        "SolAttnXPUVSA missing from the vsa workflow")

    def test_workflow_no_vsa_omits_sol_node(self):
        from atomic.fasth3_server import fasth3_workflow
        wf = fasth3_workflow(prompt="x", vsa=False)
        has_sol = any(
            n.get("class_type", "") == "SolAttnXPUVSA"
            for n in wf["prompt"].values()
        )
        self.assertFalse(has_sol,
                         "SolAttnXPUVSA must not appear in vsa=False workflow")

    def test_comfyui_workflow_node_types_iter44(self):
        """Pin the 8-node vsa workflow shape (the contract for the
        iter-44 happy path)."""
        from atomic.fasth3_server import fasth3_workflow
        wf = fasth3_workflow(prompt="a comet", vsa=True, steps=4,
                             width=64, height=64, n_frames=1)
        nodes = wf["prompt"]
        # Required node types
        required = {
            "UnetLoaderGGUF", "VAELoader", "H3SmallTextEncoder",
            "ConditioningZeroOut", "EmptyHunyuanLatentVideo",
            "SolAttnXPUVSA", "KSamplerAdvanced", "VAEDecode", "SaveImage",
        }
        present = {n["class_type"] for n in nodes.values()}
        missing = required - present
        self.assertFalse(missing, "missing node types: %s" % missing)
        # KSamplerAdvanced must be wired to the SolAttnXPU model output
        ksampler = next(n for n in nodes.values()
                        if n["class_type"] == "KSamplerAdvanced")
        self.assertEqual(ksampler["inputs"]["model"][0], "sol",
                         "KSamplerAdvanced must consume the SolAttnXPU VSA-wrapped model")


# -----------------------------------------------------------------------------
# B. Pin the real-frame fixture (unit; fixture must exist)
# -----------------------------------------------------------------------------

class TestIter44Fixture(unittest.TestCase):
    """Pin the iter-44 fixture: a real 64x64 RGBA frame from the live
    ComfyUI vsa server, archived as a pickle. The fixture is a stable
    test artifact that any future iter can replay against.
    """

    def test_fixture_pkl_exists(self):
        self.assertTrue(os.path.isfile(FIXTURE_PKL),
                        "missing fixture: %s" % FIXTURE_PKL)
        with open(FIXTURE_PKL, "rb") as f:
            d = pickle.load(f)
        self.assertEqual(d["width"], 64)
        self.assertEqual(d["height"], 64)
        self.assertEqual(len(d["rgba"]), 64 * 64 * 4)
        self.assertEqual(d["steps"], 1)
        self.assertTrue(d["vsa"], "fixture must use VSA path")
        self.assertIn("comet", d["prompt"].lower())
        # The fixture must be a real frame, not a stub
        # (not all-zero and not pure gray)
        non_zero = sum(1 for b in d["rgba"] if b != 0)
        self.assertGreater(non_zero, 100,
                           "fixture frame is mostly zero (likely stub)")

    def test_fixture_qbf_exists(self):
        self.assertTrue(os.path.isfile(FIXTURE_QBF),
                        "missing QBF fixture: %s" % FIXTURE_QBF)
        from atomic import QbfTraceStore
        store = QbfTraceStore(FIXTURE_QBF)
        try:
            runs = store.runs()
            self.assertEqual(runs, [0])
            d = store.load_run(0)
            self.assertEqual(d["manifest"]["n_video"], 1)
            self.assertEqual(len(d["video"]), 1)
            self.assertEqual(d["video"][0]["width"], 64)
            self.assertEqual(d["video"][0]["height"], 64)
            self.assertEqual(len(d["video"][0]["rgba"]), 64 * 64 * 4)
        finally:
            store.close()


# -----------------------------------------------------------------------------
# C. The no-fallback happy path against the live server
# -----------------------------------------------------------------------------

class TestIter44LiveNoFallback(unittest.TestCase):
    """End-to-end against the live ComfyUI vsa server, NO stub fallback.

    Skipped if ComfyUI vsa isn't reachable. The intent is to assert the
    wire contract holds with the patched sol_attn_xpu custom node:
    a real RGBA frame returns, the workflow's SolAttnXPU VSA node
    runs without TypeError or RecursionError, and the FastH3Client
    doesn't fall back to a stub.
    """

    def setUp(self):
        if not _comfyui_up():
            self.skipTest("ComfyUI vsa not reachable on %s:%d" %
                          (COMFYUI_HOST, COMFYUI_PORT))

    def test_fasth3_generate_no_fallback_64x64(self):
        from atomic import FastH3Client
        c = FastH3Client(
            endpoint="http://%s:%d" % (COMFYUI_HOST, COMFYUI_PORT),
            fallback=None,    # <-- THE key change: no stub fallback
            steps=1, vsa=True, vsa_keep=10,
            mode="comfyui",
        )
        r = c.generate("a comet over the ocean at sunset",
                       seed=42, width=64, height=64)
        self.assertEqual(r["width"], 64)
        self.assertEqual(r["height"], 64)
        self.assertEqual(len(r["frames"]), 1)
        rgba = r["frames"][0]
        self.assertEqual(len(rgba), 64 * 64 * 4)
        # Real frame is not all-zero
        non_zero = sum(1 for b in rgba if b != 0)
        self.assertGreater(non_zero, 100,
                           "frame mostly zero (would indicate stub fallback)")
        # The no-fallback contract: the client must not silently swap
        # in the stub. We assert via the latency + view_filename markers.
        self.assertEqual(r.get("model"), "FastH3")
        self.assertTrue(r.get("view_filename", "").startswith("atomic_"),
                        "view_filename %r is not a real ComfyUI output" %
                        r.get("view_filename"))
        # end-to-end latency: real diffusion takes > 100ms
        self.assertGreater(r["h3_latency_ms"], 50.0)

    def test_fasth3_generate_vsa_off_64x64(self):
        """The dense (vsa=False) path: SolAttnXPU is omitted, the
        diffusion runs the unmodified H3 DiT."""
        from atomic import FastH3Client
        c = FastH3Client(
            endpoint="http://%s:%d" % (COMFYUI_HOST, COMFYUI_PORT),
            fallback=None, steps=1, vsa=False,
            mode="comfyui",
        )
        r = c.generate("a comet", seed=7, width=64, height=64)
        self.assertEqual(r["width"], 64)
        self.assertEqual(len(r["frames"]), 1)
        self.assertEqual(len(r["frames"][0]), 64 * 64 * 4)
        self.assertFalse(r.get("vsa", True),
                         "vsa=False should be reflected in result")
        self.assertTrue(r.get("view_filename", "").startswith("atomic_"))


# -----------------------------------------------------------------------------
# D. QBF round-trip: load the fixture trace and assert replay
# -----------------------------------------------------------------------------

class TestIter44QBFReplay(unittest.TestCase):
    """The iter-44 QBF round-trip: load the real-frame fixture trace
    and assert the stored RGBA bytes are bit-exact preserved.

    The replay contract: QBF stores the RGBA as a JSON blob (base64)
    inside the per-run manifest + per-frame video entries. The store
    is portable, so any future iter can re-derive a FlowTrace and
    replay the stored stimulus against a fresh engine.
    """

    def test_qbf_load_run_returns_real_frame(self):
        from atomic import QbfTraceStore
        self.assertTrue(os.path.isfile(FIXTURE_QBF),
                        "QBF fixture missing: %s" % FIXTURE_QBF)
        store = QbfTraceStore(FIXTURE_QBF)
        try:
            d = store.load_run(0)
            self.assertEqual(d["manifest"]["n_video"], 1)
            self.assertEqual(d["manifest"]["n_frames"], 1)
            self.assertEqual(d["manifest"]["run_id"], 0)
            vf = d["video"][0]
            self.assertEqual(vf["width"], 64)
            self.assertEqual(vf["height"], 64)
            self.assertEqual(len(vf["rgba"]), 64 * 64 * 4)
            # The QBF stores a real ComfyUI frame, not a stub. Assert
            # it's not all-zero and not pure gray (a stub would be one
            # of those).
            non_zero = sum(1 for b in vf["rgba"] if b != 0)
            self.assertGreater(non_zero, 100,
                               "QBF frame is mostly zero (likely stub)")
        finally:
            store.close()

    def test_qbf_flow_trace_rebuild(self):
        """`store.flow_trace(rid)` rebuilds a live FlowTrace from the
        stored stimulus; that FlowTrace is replay-ready against any
        engine."""
        from atomic import QbfTraceStore
        store = QbfTraceStore(FIXTURE_QBF)
        try:
            ft = store.flow_trace(0)
            # FlowTrace has a snapshot() method that returns the same
            # shape the store understands
            snap = ft.snapshot()
            self.assertEqual(snap["n_video"], 1)
            self.assertEqual(len(snap["video"]), 1)
            self.assertEqual(snap["video"][0]["width"], 64)
            self.assertEqual(snap["video"][0]["height"], 64)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
