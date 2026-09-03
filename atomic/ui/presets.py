"""Presets: server-side persistent preset storage.

Iter 46: replaces localStorage-only presets with server-side storage backed
by a JSON file in the QBF records directory. Presets are independent of
the record/replay shard format so they remain editable and portable.

Storage: $ATOMIC_QBF_DIR/ui_records/ui_presets.json

A preset captures the visual state of a program at a given moment:
  name, program, description, groups, tile_names, tileColors, tileVizOverride,
  zoom, accentColor, params (module.key -> float), ts

The program dropdown selects a program; loading a preset applies the visual
state. Params are stageable but the engine must be running to apply them.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

__all__ = [
    "get_presets_dir",
    "list_presets",
    "get_preset",
    "save_preset",
    "delete_preset",
]


def get_presets_dir() -> Path:
    base = os.environ.get(
        "ATOMIC_QBF_DIR",
        os.path.join(os.path.expanduser("~"), ".runtime", "atomic_qbf"),
    )
    p = Path(base) / "ui_records"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _presets_path() -> Path:
    return get_presets_dir() / "ui_presets.json"


def _load_all() -> dict[str, Any]:
    path = _presets_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: dict[str, Any]):
    path = _presets_path()
    path.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")


def list_presets(program: str | None = None) -> list[dict[str, Any]]:
    """Return all preset metadata (no frames/bus data).

    If program is given, filter to that program only.
    """
    all_data = _load_all()
    out = []
    for name, pdata in all_data.items():
        if program and pdata.get("program") != program:
            continue
        out.append({
            "name": name,
            "program": pdata.get("program", ""),
            "description": pdata.get("description", ""),
            "ts": pdata.get("ts", 0),
        })
    out.sort(key=lambda x: x["ts"], reverse=True)
    return out


def get_preset(name: str) -> dict[str, Any] | None:
    """Return full preset data (including groups, tile_names, params, etc.)."""
    all_data = _load_all()
    return all_data.get(name)


def save_preset(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Save a preset; overwrites if name already exists."""
    if not name or not isinstance(name, str):
        raise ValueError("preset name must be a non-empty string")
    all_data = _load_all()
    preset = {
        "name": name,
        "program": str(data.get("program", "")),
        "description": str(data.get("description", "")),
        "groups": dict(data.get("groups") or {}),
        "tile_names": dict(data.get("tile_names") or {}),
        "tileColors": dict(data.get("tileColors") or {}),
        "tileVizOverride": dict(data.get("tileVizOverride") or {}),
        "zoom": float(data.get("zoom", 1.0)),
        "accentColor": data.get("accentColor"),
        "params": dict(data.get("params") or {}),
        "ts": int(time.time()),
    }
    all_data[name] = preset
    _save_all(all_data)
    return preset


def delete_preset(name: str) -> bool:
    """Delete a preset by name. Returns True if it existed."""
    all_data = _load_all()
    if name in all_data:
        del all_data[name]
        _save_all(all_data)
        return True
    return False
