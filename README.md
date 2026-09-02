# ATOMIC-PC — atomizing data processes into keyword gates

A live data stream routed through small typed gate blocks (logic gates, CV modulators, H(4) spatial gates) and rendered onto a tiled display. An LLM assembles a working app by matching function paths through the gates — not by writing lines of a conventional language.

This dir is the project root and concept home: `ATOMIC-PC-CORE.md` is the single source of truth for the gate list, H(4) row layout, and display-tile matrix; `ATOMIC-PC-STATE.md` is the persistent plan + evidence journal.

## Harness

Pure-Python twin of the fabric `jsfx/evaluatePatch` runtime, 22 sections green.

```
atomic/
  bus.py        rock-solid signal store + per-node host bridge
  engine.py     Python twin of evaluatePatch (batch run() + live tick() with per-tick feeds)
  gates.py      unified atom catalog (CV + logic + quantum + alogic + h4_slide + sinks)
  program.py    AtomicProgram IR (strict validation + 5 compile targets + WGSL)
  oracle.py     node driver MODE 1 batch + MODE 2 resident live (per-tick stdio hooks)
  jsnum.py dsp.py  JS-fidelity numerics + MDCT
  tiles.py      display model (control frame + 3x3/4x4 agnostic tile wall, heatmap_animation, validate_wgsl)
  trace.py      flow observer (per-tick stimulus + per-node in/out/latency rings, snapshot/export, replay, latency_histogram_from_trace)
  qbf.py qbfstore.py  Quantum Blob Format + portable trace archive (no 50 MB cap, H4 optional)
  swarm.py decompose.py teach.py evolve.py bridge.py demo.py context.py
  ui/           FastAPI tile wall (port 18094, HTML5 canvas, WebSocket streaming, zoom, accent picker, presets, record/replay, split view)
  ui/static/index.html  4x4 canvas tile wall + heatmap/series/xy/wxyz3d renderers
tests/          pytest suite mirroring the gauntlet (185 tests)
atomic/selftest.py  gauntlet (22 sections, ~180 checks, ~5s)
examples/       5 end-to-end live demos (QBF, H4, clock counter, swarm evolve teach, heatmap)
```

## Verify

Run from repo root (vendored `fabric/` + `hoa64/` — no external sibling required):

- ATOMIC-PC harness (185 tests): `python -m pytest tests -q`
  (oracle MODE 1+2 spawn `node`)
- ATOMIC-PC gauntlet (22 sections, ~180 checks, ~5s): `python -m atomic.selftest`
- Live demos (5): `python -m examples.qbf_persistence_round_trip`,
  `python -m examples.hadamard_wxyz_scope`, `python -m examples.gated_clock_counter`,
  `python -m examples.swarm_evolve_teach_demo`, `python -m examples.heatmap_animation`
- UI tile wall: `uvicorn atomic.ui:app --port 18094 --host 0.0.0.0` then
  `http://localhost:18094/run/hadamard_wxyz` (any of 7 demo programs)
- Retrieval (zvec-grep): `zg --version` (0.2.1), `zg status .` (ready, local/potion-code-16m-v2, 42/42),
  `zg query "H4 gate row layout" --human` hybrid FTS+vector; MCP via `zg install --target opencode --yes` + `zvec_grep_search`

Verify the vendored harness:
- EEL2/JSFX runtime + fabric suite: `python -m pytest fabric/tests -q`
- JSFX syntax: `node --check fabric/web/jsfx.js`
- spatial math (H4/HOA): `python -m hoa64.cli hadamard --selftest`
- .mv2 store (external, optional): `cargo test` in `memvid`

## Key contracts

- **Tick discipline**: 1 tick = host CPU/GPU step, modules tick in insertion order, wire latch = 1-tick latency, bus key `ui.tap` hardcoded, views window 512.
- **Node rule**: function/control ≤1 input port, N outputs, fan-out free, cables into one input SUM; sinks may stack.
- **H4**: Row0=W `[+ + + +]` (amplitude/consensus ~61%), Row1=Z `[+ - + -]`, Row2=Y `[+ + - -]`, Row3=X `[+ - - +]` (CORE canonical; audio rotates Z/Y/X 90°).
- **Trace is the bridge**: pure observer, touched runs stay bit-identical, replay from stored stimulus is bit-identical, 10k-frame ring + QBF portable archive.
- **WGSL**: `Program.to_wgsl()` emits `@compute @workgroup_size(64) @group(0) bus/params/state/bridge` with per-block `fn tick_<id>` and host-RAM bridge comment; validated by `Display.validate_wgsl()` (structural + `naga` if present).
- **Tiles**: 3×3 or 4×4 wall, `Display.heatmap_from_trace()` / `heatmap_from_swarm()` / `heatmap_animation()` render replay onto tiles normalized 0..1.
- **Retrieval**: `zg` 0.2.1 + `local/potion-code-16m-v2` index (`.zvec-grep/index.zvec`, 40 files / 500 entities) via `atomic/context.py` (`query`/`query_rg`/`assert_retrieval`) — stays green when zg absent, MCP `zvec_grep_search`/`_rg` same store.
- **UI**: `atomic.ui:app` FastAPI tile wall (port 18094, lifespan auto-registers 7 demo programs) + `ProgramViewer` (batch + per-tick `tick_once` + WebSocket queue broadcast + live tap/param feed apply). REST: `/api/programs`, `/api/control/<p>`, `/api/views/<p>`, `/api/snapshot/<p>`, `/api/tap/<p>`, `/api/feed/<p>`, `/api/batch/<p>`, `/api/stream/<p>`, `/api/record/<p>`, `/api/replay/<p>`; WS: `/ws/<p>` (per-tick snapshot stream + tap/param/feed/batch client msgs + ping/pong RTT + `_lat_eng`/`_lat_ws`). UI features: tile rename (dblclick), preset save/load (localStorage), record/replay (.qbf shard under `~/.runtime/atomic_qbf/ui_records/`), split view (pane 2 with own WS), signed heatmap (intensity + sign), Ctrl+scroll/+/−/0 zoom, accent color picker (overrides theme `--accent`).

## Sibling imports

Vendored: `fabric/` (2.0 MB, jsfx + microfx) and `hoa64/` (8.8 MB, Hadamard) are in-repo. Third-party siblings (`Rack`, `WebPd`, `circuitjs1`, `memvid`) remain external. Interpreters: `python` 3.11+; `node` v26.

## Iteration 15 — scale, viz, audit, release

- Decompose audit (async/walrus/f-string/comp) — fidelity pinned vs engine.
- Swarm 16 on 4×4 wall — parallel == serial, H4 consensus & row_cos.
- Trace → QBF → Replay at 10k frames — ring wrap, latency histogram, H4 1e-5.
- Tiles live viz — WGSL naga validation + per-tile heatmap_animation.
- Docs release + 4 sibling obligations verified green (only `~/ATOMIC-PC` edited).

## Iteration 16 — zvec-grep retrieval

- `zg` 0.2.1 installed (`npm install -g @zvec/zvec-grep --prefix ~/.local`, PATH `~/.local/bin`, `zg --version` 0.2.1).
- MCP `zg install --target opencode --yes` → `~/.config/opencode/opencode.json` stdio `zg server --stdio` + `AGENTS.md` managed guidance (zvec_grep_search / _rg).
- Index `zg index ~/ATOMIC-PC --embedding local/potion-code-16m-v2` → `ATOMIC-PC/.zvec-grep/index.zvec` (40/40 files, 500 entities, 256-dim cosine, no remote).
- Queries pinned via CLI + MCP and wrapper `atomic/context.py`: H4→CORE, tiles→tiles.py, QBF→qbf (limit 7, hybrid FTS+vector), `zg query --rg h4_gate` exact; harness skips cleanly when zg absent.
- Gauntlet + pytest green (only `~/ATOMIC-PC` edited).

## Iterations 17–22 — UI tile wall, live demos, polish

- Iter 17: planning/hardening — scope review, retrieval re-verification, git plan (closed).
- Iter 18: 5 live demos under `examples/` (QBF round-trip, H4 scope, gated clock counter, swarm→evolve→teach, heatmap animation). Selftest grew to 18 sections.
- Iter 19: `atomic/ui/` FastAPI tile wall (port 18094) + 7 demo programs + 4x4 canvas HTML5 renderer + WebSocket streaming. `tests/test_ui.py` (15 tests). Selftest 17 sections.
- Iter 20 (UI iter 4): keyboard shortcuts (space/r/+/-/g), tile rename, presets, record/replay (.qbf shards), split view, signed heatmap, RTT ping/pong, latency overlay. Selftest 19 sections, 175 checks.
- Iter 21 (UI iter 7): Ctrl+scroll/+/−/0 zoom + accent color picker (theme override via `--accent-override`).
- Iter 22: release prep — git history polished, UI/index.html (2543 lines) committed, harness still 185/185 + 22/22 selftest green.

Final state: 185 tests, 22-section gauntlet, vendored fabric/ + hoa64/, in-repo UI tile wall, live demos, retrieval layer.
