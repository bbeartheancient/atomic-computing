# ATOMIC-PC — atomizing data processes into keyword gates

A live data stream routed through small typed gate blocks (logic gates, CV modulators, H(4) spatial gates) and rendered onto a tiled display. An LLM assembles a working app by matching function paths through the gates — not by writing lines of a conventional language.

This dir is the project root and concept home: `ATOMIC-PC-CORE.md` is the single source of truth for the gate list, H(4) row layout, and display-tile matrix; `ATOMIC-PC-STATE.md` is the persistent plan + evidence journal.

## Harness

Pure-Python twin of the fabric `jsfx/evaluatePatch` runtime, 16 sections green.

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
tests/          pytest suite mirroring the gauntlet (170 tests)
atomic/selftest.py  gauntlet (16 sections, ~120 checks, ~5s)
```

## Verify

Run from `$HOME` so `hoa64` imports (it is cwd-imported — the one `hoa64` test skips otherwise):

- ATOMIC-PC harness (170 tests): `cd ~/ && ~/runtime/.venv/bin/python -m pytest ATOMIC-PC/tests -q`
  (oracle MODE 1+2 spawn `node`; run from `$HOME` for the full 170)
- ATOMIC-PC gauntlet (16 sections, ~120 checks, ~5s): `cd ~/ATOMIC-PC && ~/runtime/.venv/bin/python -m atomic.selftest`
  (selftest fixes `sys.path` for `hoa64`/`fabric`)
- Retrieval (zvec-grep): `zg --version` (0.2.1, `~/.local/bin/zg`), `zg status ~/ATOMIC-PC` (ready, local/potion-code-16m-v2, 40/40),
  `zg query "H4 gate row layout" --human` hybrid FTS+vector; MCP via `zg install --target opencode --yes` + `zvec_grep_search` (Agent context `~/ATOMIC-PC/AGENTS.md`)

Verify against sibling trees (run from there):

- EEL2/JSFX runtime + fabric suite: `~/M1Multitronic/.venv-fabric/bin/python -m pytest fabric/tests -q`
- JSFX syntax: `node --check ~/M1Multitronic/fabric/web/jsfx.js`
- host numpy / afi package: `~/runtime/.venv/bin/python -m pytest ~/M1Multitronic/python/tests -q`
- spatial math (H4/HOA): `cd ~/ && ~/hoa64-venv/bin/python -m hoa64.cli hadamard --selftest`
- .mv2 store: `cd ~/memvid && cargo test` (dormant since iter 8 — QBF is the portable store)

## Key contracts

- **Tick discipline**: 1 tick = host CPU/GPU step, modules tick in insertion order, wire latch = 1-tick latency, bus key `ui.tap` hardcoded, views window 512.
- **Node rule**: function/control ≤1 input port, N outputs, fan-out free, cables into one input SUM; sinks may stack.
- **H4**: Row0=W `[+ + + +]` (amplitude/consensus ~61%), Row1=Z `[+ - + -]`, Row2=Y `[+ + - -]`, Row3=X `[+ - - +]` (CORE canonical; audio rotates Z/Y/X 90°).
- **Trace is the bridge**: pure observer, touched runs stay bit-identical, replay from stored stimulus is bit-identical, 10k-frame ring + QBF portable archive.
- **WGSL**: `Program.to_wgsl()` emits `@compute @workgroup_size(64) @group(0) bus/params/state/bridge` with per-block `fn tick_<id>` and host-RAM bridge comment; validated by `Display.validate_wgsl()` (structural + `naga` if present).
- **Tiles**: 3×3 or 4×4 wall, `Display.heatmap_from_trace()` / `heatmap_from_swarm()` / `heatmap_animation()` render replay onto tiles normalized 0..1.
- **Retrieval**: `zg` 0.2.1 + `local/potion-code-16m-v2` index (`.zvec-grep/index.zvec`, 40 files / 500 entities) via `atomic/context.py` (`query`/`query_rg`/`assert_retrieval`) — stays green when zg absent, MCP `zvec_grep_search`/`_rg` same store.

## Sibling imports (never vendored)

`~/M1Multitronic`, `~/hoa64`, `~/memvid`, `~/Rack`, `~/WebPd`, `~/circuitjs1`.
Interpreters: `~/runtime/.venv/bin/python`, `~/M1Multitronic/.venv-fabric/bin/python`, `~/hoa64-venv/bin/python`; node v26.

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
