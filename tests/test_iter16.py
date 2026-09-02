"""test_iter16 — zvec-grep retrieval (iter 16).

Pinned against zg CLI + index built for ~/ATOMIC-PC (local/potion-code-16m-v2).
When zg is absent the suite skips with a clear message; harness never vendors zg.
"""
import os, sys
import shutil
import subprocess
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic.context import (
    is_available, zg_version, query, query_rg, assert_retrieval,
    CANONICAL_QUERIES, status,
)

pytestmark = pytest.mark.skipif(False, reason="placeholder")

def _skip_if_no_zg():
    if not is_available():
        pytest.skip("zg not found on PATH (~/.local/bin/zg) — run `npm install -g @zvec/zvec-grep --prefix ~/.local` and `zg index ~/ATOMIC-PC --embedding local/potion-code-16m-v2`")

def test_zg_available_and_version():
    _skip_if_no_zg()
    v = zg_version()
    assert v is not None and v != "", v
    # version should look like 0.2.1
    assert v[0].isdigit()

def test_zg_status_ready():
    _skip_if_no_zg()
    s = status()
    # when index exists, status contains ready or coverage
    assert "ready" in s.lower() or "coverage" in s.lower() or "ATOMIC-PC" in s, s

def test_h4_retrieval_fidelity():
    _skip_if_no_zg()
    assert_retrieval("H4 gate row layout and W dominance", "ATOMIC-PC-CORE.md")

def test_tiles_retrieval_fidelity():
    _skip_if_no_zg()
    assert_retrieval("tile heatmap animation from trace replay", "tiles.py")

def test_qbf_retrieval_fidelity():
    _skip_if_no_zg()
    # qbf appears in qbf.py / qbfstore.py / trace.py etc; substring "qbf" suffices
    assert_retrieval("QBF portable store ring wrap", "qbf")

def test_rg_exact():
    _skip_if_no_zg()
    res = query_rg("h4_gate", extra_args=["-n"])
    assert res["available"] is True
    blob = res["raw"]
    # rg should find h4_gate in qbf.py / bridge.py
    assert "h4_gate" in blob, blob[:500]

def test_canonical_queries_all():
    _skip_if_no_zg()
    for q, expect in CANONICAL_QUERIES:
        assert_retrieval(q, expect, limit=5)

def test_query_wrapper_returns_structured():
    _skip_if_no_zg()
    res = query("H4 gate", limit=2)
    assert res["available"] is True
    assert isinstance(res["hits"], list)
    assert len(res["hits"]) >= 1
    assert "raw" in res
