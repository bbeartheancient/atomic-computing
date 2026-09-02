# AGENTS.md — ATOMIC-PC (atomic-computing)

## Mission
Atomize data processes into keyword gates: an app is a live data stream routed
through small typed gate blocks (logic gates, CV modulators, H(4) spatial
gates) and rendered onto a tiled display. The point is that an LLM assembles
a working app by matching function paths through the gates — not by writing
lines of a conventional language.

## Objective
This directory is the project root and concept home: the spec
(`ATOMIC-PC-CORE.md`), the persistent plan+evidence journal
(`ATOMIC-PC-STATE.md`), and — since iter 3 — a LIVE Python harness
(`atomic/` + `tests/`): bus, the engine twin of jsfx `evaluatePatch`,
the full gate catalog, the program IR with 5 compile targets, the
node oracle in BOTH modes (MODE 1 batch, MODE 2 live/resident with
per-tick feed hooks), and — since iter 8 — the QBF portable trace
store (goal 6; the operator's "middle" format: working blobs, no
50 MB cap, H(4) gates optional). "Done" = the spec stays coherent
AND the harness stays green (170 tests, 16 sections, `ATOMIC-PC-STATE.md` tracks
per-iteration state; iter 17 re-verified 16/16 + 170 passed + 359 fabric + hoa selftest).
Still: no local git yet; the canonical remote
`github.com/bbeartheancient/atomic-computing` exists but is EMPTY.
Zero sibling edits: the harness IMPORTS the siblings (jsfx.js,
microfx, hoa64) and never vendors or modifies them.

PRIME DIRECTIVE: IF A BETTER WAY WOULD WORK, SUGGEST IT!

NOTE:  Project is created one iteration at a time: start with large concepts, make them work with smaller details later
IMPORTANT: When making edits, make the edits first. If additional questions persist, add them in a separate edit. Do not try to figure everyething out at once.
            -- iteration budgets are capped at 128k; you must make your edits and end turn for next iteration at 128k token usage.

Verify the harness (this dir; run from `$HOME` so `hoa64` imports —
it is cwd-imported — the one `hoa64` test skips otherwise):
- ATOMIC-PC harness (170 tests): `cd ~/ && ~/runtime/.venv/bin/python -m pytest ATOMIC-PC/tests -q`
  (oracle MODE 1+2 spawn `node`; run from `$HOME` for the full 170).
- ATOMIC-PC gauntlet (16 sections, ~119 checks, ~5s): `cd ~/ATOMIC-PC && ~/runtime/.venv/bin/python -m atomic.selftest`
  (run from `~/ATOMIC-PC`; selftest fixes `sys.path` for `hoa64`/`fabric`; `pytest` still needs `cd ~/` for the `hoa64` cwd-import).

Verify against the sibling trees (run from there):
- EEL2/JSFX runtime + fabric suite: `~/M1Multitronic/.venv-fabric/bin/python -m pytest fabric/tests -q`
- JSFX syntax: `node --check ~/M1Multitronic/fabric/web/jsfx.js`
- host numpy / afi package: `~/runtime/.venv/bin/python -m pytest ~/M1Multitronic/python/tests -q`
- spatial math (H4/HOA): `cd ~/ && ~/hoa64-venv/bin/python -m hoa64.cli hadamard --selftest`
- .mv2 store: `cd ~/memvid && cargo test`

## Map
One signal bus, read bottom-up.

1. **Currency = live streams.** Every source is a stream; a node transforms
   an input stream into output stream(s). Node rule (Rack v2): a function node
   has <=1 input port, any number of outputs; fan-out is free; cables into one
   input SUM. Two paths: SIGNAL (in -> function -> out) and CONTROL (modulates
   the signal). Hierarchy: source -> function -> control -> visualizer (sinks
   may stack many inputs; the one-input rule does not bind them).
2. **Gates (the atoms).** (a) Logic macros: AND OR NOT NAND NOR XOR XNOR plus
   IMPLY/NIMPLY and the electronic set, plus a quantum set (Identity, Pauli
   X/Y/Z, CNOT, CU, Phase, Hadamard, Swap, Toffoli) — Toffoli+Hadamard is the
   universal quaternion->linear gate. (b) **H(4) gate — the keystone**: a
   4-row Sylvester-Hadamard spatial gate splitting a related stream group into
   4 orthogonal streams. Canonical layout (CORE): Row0=W `[+ + + +]` =
   amplitude/consensus (dominant, ~61% energy); Row1=Z `[+ - + -]`;
   Row2=Y `[+ + - -]`; Row3=X `[+ - - +]`. LLM mapping: W = full-attention /
   consensus, Z/Y/X = the three linear layers (SSM-progression / routing /
   residual). For audio the Z/Y/X axis labels rotate 90 deg. FOA is a
   normalized H4 (hoa64). (c) CV modulator modules (const, clock_bpm, gain,
   smooth, threshold, moving_avg, mdct_flux, sensor, tap, toggle, accum, ...):
   the full catalog is in `microfx.py`.
3. **Assembly + display.** An AtomicProgram is a graph of AtomicBlocks (typed
   ports, EEL2 body) + Wires + an entry point, compiled multi-target (microfx
   / eel2 / python / wgsl / flowchart); `from_description()` turns natural
   language into a program. `dma_trace` taps the live flow — "the trace is the
   bridge" (GPU1 captures / subconscious, GPU0 decides / conscious). Output
   renders onto the display's own tiled matrix: a control frame (top-level
   i/o) + a 3x3 or 4x4 grid of universal, agnostic framebuffer tiles whose
   resolution derives from the full display; tiles link into larger
   sub-matrices (a tiled video wall).

Roadmap (the 10 goals) is in `~/M1Multitronic/docs/BICAMERAL_FRAMEWORK.md`
S10: learn from direct DMA streams (1) -> assemble blocks into visual
assembly (2) -> user edits blocks to run apps (3) -> learn function paths from
teacher examples (4) -> swappable domain vocab/n-gram sets routed by signal
path (5) -> stored in the portable QBF store (6; iter-8 directive:
the operator's "middle format" — working blobs, no 50 MB cap,
optional H4 gates — superseded .mv2 and set .tqbf aside) ->
self-improvement loops (7) ->
decompose any language/compiled code into atomic functions (8) -> bicameral
two-GPU split (9) -> cheap active params => parallel agent swarms (10).

Module roles in the verified harness (`~/M1Multitronic/fabric/`):
`web/jsfx.js` = EEL2/JSFX runtime + host bridge; `microfx.py` = module
catalog + ngram verbs; `atomic_program.py` = block/wire IR + compiler;
`dma_trace.py` = flow observer; `main.py` = FastAPI (:18093).

Module roles in the LOCAL harness (this dir, `~/ATOMIC-PC/atomic/`,
imported — never vendored — from the siblings): `bus.py` = the rock-
solid signal store + per-node host bridge; `engine.py` = the Python
twin of `evaluatePatch` (the pinned per-tick loop, batch `run()` +
live `tick()` with per-tick feeds); `gates.py` = the unified atom
catalog (CV + logic + quantum + alogic + h4_slide + sinks);
`program.py` = the AtomicProgram IR (strict validation + 5 compile
targets); `oracle.py` = the node driver, MODE 1 batch + MODE 2
resident live (per-tick stdio hooks for live taps/params);
`jsnum.py`/`dsp.py` = the JS-fidelity numerics + MDCT; `tiles.py` = the
display model (control frame + 3x3/4x4 agnostic tile wall,
resolution derived from the full display, linked tile groups =
larger sub-matrices); `trace.py` = the flow observer (per-tick
stimulus + per-node in/out/latency rings, snapshot/export,
`from_snapshot()` rebuild, `replay()` re-drives a fresh engine
from the recorded stimulus); `qbf.py` = the Quantum Blob Format
(goal 6, iter 8): a self-describing named-blob container — 64-byte
header, u64 offsets/sizes (no 50 MB tier, no search index to
poison), per-blob sha256, and the H(4) gate as an OPTIONAL per-
blob codec flag (the CORE keystone); `qbfstore.py` = the portable
trace archive on top of it (one .qbf shard per store: index +
per-run manifest/ticks/one blob per node frame; append_run/
load_run/flow_trace/replay_run/export_run; bit-identical replay;
dir ~/.runtime/atomic_qbf, env ATOMIC_QBF_DIR).
The engine takes an optional `trace=` observer: pure read-only
(touched runs stay bit-identical; off by default).

## Anchors
- This dir is the project root and concept home: `ATOMIC-PC-CORE.md` is the
  single source of truth for the gate list, H(4) row layout, and display-tile
  matrix. Change semantics THERE first, then in code.
- Git remote `https://github.com/bbeartheancient/atomic-computing` (description
  "atomizing data processes into keyword gates") exists but is EMPTY. This
  local dir is NOT a git repo yet.
- Sibling trees to import (never vendor): `~/M1Multitronic`, `~/hoa64`,
  `~/memvid`, `~/Rack`, `~/WebPd`, `~/circuitjs1`.
- Interpreters (verified present): `~/runtime/.venv/bin/python` (torch +
  transformers + memvid_sdk), `~/M1Multitronic/.venv-fabric/bin/python`,
  `~/hoa64-venv/bin/python`. CAUTION: the `~/miniforge3/envs/sage-dev` that
  `hoa64/AGENTS.md` names is NOT on this box — use `~/hoa64-venv`.
- Ports: vLLM :8000, fabric :18093, hoa64 serve :8765, hoa64 webapp :8770.
  PLE state: `~/.runtime/ngram_shards/`. node v26 for JSFX.

## Decision frames
- Import sibling trees; never vendor them.
- Live streams are the only currency — never design a batch API. A node is a
  stream->stream transform.
- Do not break the node rule (<=1 input on function nodes, N outputs, cables
  sum; sinks are exempt).
- H(4) Row0 is always W / amplitude. Keep CORE's row order; only the Z/Y/X axis
  labels rotate for audio.
- EEL2/JSFX idiom: undeclared double vars, case-insensitive, namespaced `a.b`;
  sections `@init/@tick/@slider/@gfx`; `mem[]`/`gmem[]`; `input(name)` /
  `output(name,v)`; `loop(count,code)` / `while(cond)(code)`.
- Keep atoms multi-target (an EEL2 body that also compiles to python/wgsl);
  prefer the language-agnostic form.
- Do not write a SYCL kernel for a cold path (the I64 lesson); keep
  allocations small on the live card (vLLM owns GPU0 VRAM).

## Obligations
- Touched EEL2/JSFX or fabric: `node --check .../jsfx.js` clean AND
  `...pytest fabric/tests -q` green.
- Touched host numpy/afi: `~/runtime/.venv/bin/python -m pytest
  ~/M1Multitronic/python/tests -q` green.
- Touched spatial math: hoa64 `hadamard --selftest` passes.
- Touched .mv2 persistence: `cd ~/memvid && cargo test` green AND you honored
  the .mv2 landmines (below). Iter-8 note: goal 6's store is now the
  local pure-Python .qbf — .mv2 is untouched, so this obligation is
  dormant unless you actually edit ~/memvid.
- Never restart production services (vllm/fabric) unless the task explicitly
  requires it.
- Do not commit unless asked. If you ARE asked: this dir needs `git init` +
  the remote set first (it is not a repo yet); then one commit per turn.

## Evidence
- Operational landmines + live journal: `~/M1Multitronic/AGENTS.md` (vLLM
  wedges, EngineCore core dump, PLE length gate, echo fix). 11-pass selftest:
  `~/M1Multitronic/FIRSTRUN.md`. Historical numbers:
  `~/M1Multitronic/docs/AGENTS-archive-*.md`.
- .mv2 traps: the 50 MB tier limit (MV001) is native — the PLE store rotates
  on it; always `enable_vec=False`; bookkeeping goes in tags, not metadata;
  `close_all()` after ~10 create/close cycles. Spec: `~/memvid/MV2_SPEC.md`.
- H(4) measured numbers (energy split, bandwidth tradeoff):
  `~/M1Multitronic/docs/h4-bridge-codec.md` + `h4-layer-gating.md`.
- EEL2 runtime traps: `compile()` must wrap parse output in a block node;
  `Proxy.has` must return true for all globals or patches break.
