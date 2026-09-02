"""qbf_records — persist recorded WS frames to a portable .qbf shard.

iter 4 record/replay: the UI client captures live WS snapshots and POSTs
them here; we write them as `frames` blobs in a per-program .qbf shard
(sibling to `atomic.qbfstore.QbfTraceStore` but much simpler — one shard
per program, N runs, each run a single flat `frames` JSON blob).

This is intentionally NOT a FlowTrace replay (which is per-tick
reconstruction).  The UI's record is "what the user saw on screen",
which is exactly the bus+series+views+t tuple that the renderer
consumes.  Replay sends those frames back to the client; the client
just paints them.  This keeps the loop trivial: record -> save ->
load -> redraw.

Storage:
  $ATOMIC_QBF_DIR/ui_records/{program}.qbf
  per program: a single .qbf file holding N run blobs
  run blob = JSON {t0, ts, program, frames: [...]}
"""
from __future__ import annotations

import os
import time
import struct
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


_QBF_HEADER = b"QBF1"
_QBF_OFFSET = 64


def get_qbf_dir() -> Path:
    base = os.environ.get("ATOMIC_QBF_DIR",
                          os.path.join(os.path.expanduser("~"), ".runtime",
                                       "atomic_qbf"))
    p = Path(base) / "ui_records"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _shard_path(program: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in program)
    return get_qbf_dir() / f"{safe}.qbf"


def _read_shard(path: Path) -> List[Dict[str, Any]]:
    """Read a (possibly empty/missing) .qbf shard and return its run list."""
    if not path.exists() or path.stat().st_size < _QBF_OFFSET:
        return []
    data = path.read_bytes()
    runs = []
    i = _QBF_OFFSET
    while i + 8 <= len(data):
        (size,) = struct.unpack(">Q", data[i:i+8])
        i += 8
        if i + size > len(data):
            break
        blob = data[i:i+size]
        i += size
        try:
            runs.append(json.loads(blob.decode("utf-8")))
        except Exception:
            continue
    return runs


def _write_shard(path: Path, runs: List[Dict[str, Any]]):
    """Rewrite the entire shard with the given runs."""
    with open(path, "wb") as f:
        f.write(_QBF_HEADER)
        f.write(b"\x00" * (_QBF_OFFSET - 4))
        for run in runs:
            blob = json.dumps(run).encode("utf-8")
            f.write(struct.pack(">Q", len(blob)))
            f.write(blob)


def save_record(program: str, frames: List[Dict[str, Any]]) -> str:
    """Append a new run to the program's shard; return its run id."""
    path = _shard_path(program)
    runs = _read_shard(path)
    next_idx = len(runs)
    run_id = f"r{next_idx:04d}"
    runs.append({
        "run_id": run_id,
        "program": program,
        "t0": frames[0].get("t", 0) if frames else 0,
        "tN": frames[-1].get("t", 0) if frames else 0,
        "n_frames": len(frames),
        "ts": int(time.time()),
        "frames": frames,
    })
    _write_shard(path, runs)
    return run_id


def list_records(program: str) -> List[Dict[str, Any]]:
    """Return metadata-only list of runs (no frames)."""
    path = _shard_path(program)
    runs = _read_shard(path)
    return [{
        "rid": r["run_id"],
        "t0": r.get("t0", 0),
        "tN": r.get("tN", 0),
        "n_frames": r.get("n_frames", 0),
        "ts": r.get("ts", 0),
    } for r in runs]


def load_record(program: str, run_id: str) -> Optional[List[Dict[str, Any]]]:
    """Return frames for a given run, or None if not found."""
    path = _shard_path(program)
    for r in _read_shard(path):
        if r["run_id"] == run_id:
            return r.get("frames") or []
    return None
