"""context.py — zvec-grep retrieval wrapper (iter 16).

Thin stdlib wrapper around the external `zg` CLI (npm @zvec/zvec-grep).
The harness stays pure-Python; zg is optional — when absent the helper
reports unavailable and tests skip with a clear message. No vendoring;
zg is invoked via subprocess / MCP only.

Usage:
  from atomic.context import is_available, query, query_rg, zg_version
  hits = query("H4 gate row layout and W dominance")
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# zg binary discovery
# ---------------------------------------------------------------------------

def _find_zg() -> Optional[str]:
    # 1) PATH
    p = shutil.which("zg")
    if p:
        return p
    # 2) common npm global prefix ~/.local/bin
    cand = Path.home() / ".local" / "bin" / "zg"
    if cand.is_file():
        return str(cand)
    cand2 = Path("/usr/local/bin/zg")
    if cand2.is_file():
        return str(cand2)
    return None

ZG_BIN = _find_zg()
DEFAULT_ROOT = str(Path.home() / "ATOMIC-PC")

# ---------------------------------------------------------------------------
# availability / version
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """True if `zg` is on this machine."""
    return _find_zg() is not None

def zg_version(timeout: float = 5.0) -> Optional[str]:
    """Return `zg --version` or None if unavailable."""
    zg = _find_zg()
    if not zg:
        return None
    try:
        r = subprocess.run([zg, "--version"], capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None

def status(root: Optional[str] = None, timeout: float = 10.0) -> str:
    """Return `zg status <root>` text (or empty if zg missing)."""
    zg = _find_zg()
    if not zg:
        return ""
    cmd = [zg, "status"]
    if root:
        cmd.append(root)
    else:
        cmd.append(DEFAULT_ROOT)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# core query (hybrid FTS+vector)
# ---------------------------------------------------------------------------

_HIT_RE = re.compile(r"#\d+\s+\S+\s+matchedBy=\S+", re.MULTILINE)

def _parse_hits(raw: str) -> List[Dict[str, Any]]:
    """Parse zg human output into hit dicts (best-effort)."""
    hits: List[Dict[str, Any]] = []
    # files list via File: headers — more reliable than hit ranges
    files = re.findall(r"File:\s+(\S+)", raw)
    # each hit line contains score + range
    # Range: may appear on same line or following line
    cur_file: Optional[str] = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("File:"):
            # File: atomic/tiles.py  -> cur_file
            parts = stripped.split()
            if len(parts) >= 2:
                cur_file = parts[1]
            continue
        # hit line: #1 indexed_entity matchedBy=...
        if re.match(r"#\d+\s+", stripped) and "matchedBy=" in stripped:
            # score
            sm = re.search(r"score=([0-9.]+)", stripped)
            score = float(sm.group(1)) if sm else None
            # Range: <a>-<b> may be on same hit header or next line "Range: X-Y"
            rm = re.search(r"Range:\s*(\d+)-(\d+)", line)
            if rm:
                a, b = int(rm.group(1)), int(rm.group(2))
            else:
                # look ahead not available here; store none
                a, b = None, None
            hits.append({"file": cur_file or "", "range": (a, b) if a is not None else None, "score": score, "raw": stripped})
            continue
        # separate Range: line after hit without being the hit line itself
        if stripped.startswith("Range:") and hits and hits[-1].get("range") is None:
            rm = re.search(r"Range:\s*(\d+)-(\d+)", stripped)
            if rm:
                hits[-1]["range"] = (int(rm.group(1)), int(rm.group(2)))
    # if hits have no file (no File: preceding), fall back to files list ordering
    if hits and all(not h["file"] for h in hits) and files:
        for i, h in enumerate(hits):
            if i < len(files):
                h["file"] = files[i]
    # also synthesize hits from File: headers even if hit lines not parsed (e.g. rg mode)
    if not hits and files:
        for f in files:
            hits.append({"file": f, "range": None, "score": None, "raw": f})
    return hits

def query(
    text: str,
    limit: int = 7,
    preview: str = "full",
    root: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Run `zg query <text>` (hybrid FTS+vector) and return structured result.

    Returns dict with keys: query, available, raw, hits, files.
    When zg is absent, available=False and hits=[].
    """
    zg = _find_zg()
    if not zg:
        return {"query": text, "available": False, "raw": "", "hits": [], "files": [], "reason": "zg not found on PATH (~/.local/bin/zg)"}
    # Build command: positional query is hybrid; pass as single arg
    # Use --human to get stable parsable output (agent markdown also works).
    cmd = [zg, "query", text, "--limit", str(limit), "--human"]
    # preview flag if requested and zg version supports it
    if preview in ("full", "short", "none"):
        cmd += ["--preview", preview]
    # scope to workspace root via cwd — zg discovers nearest .zvec-grep upward
    cwd = root or DEFAULT_ROOT
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        raw = (r.stdout or "") + (r.stderr or "")
        hits = _parse_hits(raw)
        files = []
        for h in hits:
            if h["file"] not in files:
                files.append(h["file"])
        return {"query": text, "available": True, "raw": raw, "hits": hits, "files": files, "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"query": text, "available": True, "raw": "", "hits": [], "files": [], "error": "timeout"}
    except Exception as e:
        return {"query": text, "available": True, "raw": "", "hits": [], "files": [], "error": str(e)}

def query_rg(
    pattern: str,
    path: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Run `zg query --rg <pattern> [path]` (managed ripgrep).

    Returns dict with raw output and lines.
    """
    zg = _find_zg()
    if not zg:
        return {"pattern": pattern, "available": False, "raw": "", "lines": [], "reason": "zg not found"}
    cmd = [zg, "query", "--rg", pattern]
    if extra_args:
        cmd += extra_args
    if path:
        cmd.append(path)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=DEFAULT_ROOT)
        raw = (r.stdout or "") + (r.stderr or "")
        lines = [l for l in raw.splitlines() if l.strip()]
        return {"pattern": pattern, "available": True, "raw": raw, "lines": lines, "returncode": r.returncode}
    except Exception as e:
        return {"pattern": pattern, "available": True, "raw": "", "lines": [], "error": str(e)}

# ---------------------------------------------------------------------------
# fidelity helpers — pin that harness queries resolve to expected files
# ---------------------------------------------------------------------------

def assert_retrieval(
    query_text: str,
    expect_file_substr: str,
    limit: int = 7,
) -> Dict[str, Any]:
    """Query and assert at least one hit file contains expect_file_substr.

    Raises AssertionError if not found (and zg is available).
    If zg is unavailable, returns with available=False (caller should skip).
    """
    res = query(query_text, limit=limit)
    if not res.get("available"):
        return res
    files = res.get("files", [])
    # also consider raw for file names not in hits parse (case-insensitive)
    blob = res.get("raw", "")
    ef = expect_file_substr.lower()
    if not any(ef in f.lower() for f in files) and ef not in blob.lower():
        raise AssertionError(f"retrieval fidelity: query {query_text!r} expected file containing {expect_file_substr!r} but got files={files}")
    return res

# convenience: the three canonical harness queries pinned in selftest/tests
CANONICAL_QUERIES: List[Tuple[str, str]] = [
    ("H4 gate row layout and W dominance", "ATOMIC-PC-CORE.md"),
    ("tile heatmap animation from trace replay", "tiles.py"),
    ("QBF portable store ring wrap", "qbf"),
]
