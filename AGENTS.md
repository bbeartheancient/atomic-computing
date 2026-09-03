# AGENTS.md — ATOMIC-PC (atomic-computing)

## Mission
Atomize data processes into keyword gates: an app is a live data stream routed
through small typed gate blocks (logic gates, CV modulators, H(4) spatial
gates) and rendered onto a tiled display. The point is that an LLM assembles
a working app by matching function paths through the gates — not by writing
lines of a conventional language.


UPDATE 9/2/2026:  Source repos added to local drive:
    - WebPd: /home/bbear/WebPd
    - Open Stage Control: /home/bbear/open-stage-control_1.31.1_node
    - Pure Data: /home/bbear/pure-data
    - Pure Data JS: /home/bbear/pdjs
    - Infinite Livestream: /home/bbear/infinite-livestream
    
    
UPDATE 2 9/2/2026:  Local Duty LLM needs trained LoRa to fully enable atomic computing functions
    - LoRa similar to Backlot scriptwriting LoRa but for atomic computing functions

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
AND the harness stays green (185 tests, 22 sections, `ATOMIC-PC-STATE.md` tracks
per-iteration state; iter 25 verified 24/24 + 196 passed + 359 fabric + hoa selftest + naga WGSL validation).
Still: git remote `github.com/bbeartheancient/atomic-computing` is live on `main` (first push e9cf4f1, 43 files); vendored `fabric/` (2.0 MB, jsfx + microfx + fixture) and `hoa64/` (8.8 MB, Hadamard spatial lib) are now in-repo so a fresh clone is standalone. Third-party siblings (`Rack`, `WebPd`, `circuitjs1`, `memvid`) remain external and are not vendored.

PRIME DIRECTIVE: IF A BETTER WAY WOULD WORK, SUGGEST IT!

NOTE:  Project is created one iteration at a time: start with large concepts, make them work with smaller details later
IMPORTANT: When making edits, make the edits first. If additional questions persist, add them in a separate edit. Do not try to figure everyething out at once.
            -- iteration budgets are capped at 128k; you must make your edits and end turn for next iteration at 128k token usage.

Verify the harness (unified, vendored `fabric/` + `hoa64/`):
- ATOMIC-PC harness (425 tests): `python -m pytest tests -q`
  (oracle MODE 1+2 spawn `node`; vendored `fabric/web/jsfx.js` + `hoa64` — no external sibling required).
- ATOMIC-PC gauntlet (34 sections, ~210 checks, ~5s): `python -m atomic.selftest`
- Live demos (eight end-to-end scripts under `examples/`):
  - `python -m examples.qbf_persistence_round_trip` — record+archive+load+replay a trace via `.qbf` shard
  - `python -m examples.hadamard_wxyz_scope`        — h4_slide keystone + 4x4 Display + viz_series
  - `python -m examples.gated_clock_counter`        — from_description() -> compile -> engine run
  - `python -m examples.swarm_evolve_teach_demo`     — swarm -> evolve -> teach -> QBF round-trip
  - `python -m examples.heatmap_animation`          — Display.heatmap_animation from a live trace
  - `python -m examples.bicameral_pipeline`         — iter 25: sub=clock_bpm@60 -> HostBridge(bridge_latency=1) -> con=accum->smooth->viz_series (two engines, 1-tick host-RAM)
  - `python -m examples.swarm_video_h3_consensus`    — iter 33: 4-agent Swarm H4 W-channel consensus -> H3 prompt routing
  - `python -m examples.qbf_video_frame_trace`       — iter 33: H3 frames -> FlowTrace -> QBF shard -> load_run -> flow_trace (bit-exact video round-trip)
  - `python -m examples.infinite_slop_loop`          — iter 41: Swarm H4 pick -> H3Stub -> FlowTrace -> fitness -> SlopEvolver (bank evolve) -> QBF archive
- ATOMIC-PC UI tile wall (iter 17, port 18094, FastAPI + HTML5 canvas):
  - boot: `uvicorn atomic.ui:app --port 18094 --host 0.0.0.0`
  - browse: `http://localhost:18094/run/hadamard_wxyz` (any of 7 demo programs)
    or `http://localhost:18094/run/bicameral_clock` (iter 25: sub/con engines via HostBridge)
  - REST: `/api/programs` (includes `bicameral` key), `/api/control/<p>`, `/api/views/<p>`,
          `/api/snapshot/<p>`, `/api/tap/<p>`, `/api/feed/<p>`, `/api/batch/<p>`, `/api/stream/<p>`
  - REST (bicameral): `/api/bicameral`, `/api/bicameral/<p>/snapshot`, `/api/bicameral/<p>/batch`,
          `/api/bicameral/<p>/bridge`
  - REST (feed_video, iter 33): `/api/feed_video/<p>/start|stop|push_frame|batch|status`
          — server-push H3 frames into the engine via viz_video; WebSocket push via `/ws/feed_video/<p>`
  - WS: `/ws/<p>` — per-tick snapshot stream; `/ws/bicameral/<p>` (iter 25: bicameral pipeline,
    sub/con/bridge snapshot with bridge depth history); `/ws/feed_video/<p>` (iter 33: server-push tick loop)

Verify the vendored harness:
- EEL2/JSFX runtime + fabric suite: `python -m pytest fabric/tests -q`
- JSFX syntax: `node --check fabric/web/jsfx.js`
- spatial math (H4/HOA): `python -m hoa64.cli hadamard --selftest`
- .mv2 store (external, optional): `cargo test` in `memvid`

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

Roadmap (the 10 goals) is in `ATOMIC-PC-STATE.md` (BICAMERAL_FRAMEWORK archived in `fabric/docs/`)
S10: learn from direct DMA streams (1) -> assemble blocks into visual
assembly (2) -> user edits blocks to run apps (3) -> learn function paths from
teacher examples (4) -> swappable domain vocab/n-gram sets routed by signal
path (5) -> stored in the portable QBF store (6; iter-8 directive:
the operator's "middle format" — working blobs, no 50 MB cap,
optional H4 gates — superseded .mv2 and set .tqbf aside) ->
self-improvement loops (7) ->
decompose any language/compiled code into atomic functions (8) -> bicameral
two-GPU split (9) -> cheap active params => parallel agent swarms (10).

Module roles in the vendored harness (`fabric/`):
`web/jsfx.js` = EEL2/JSFX runtime + host bridge; `microfx.py` = module
catalog + ngram verbs; `atomic_program.py` = block/wire IR + compiler;
`dma_trace.py` = flow observer; `main.py` = FastAPI (:18093).

Module roles in the LOCAL harness (`atomic/`): `bus.py` = the rock-
solid signal store + per-node host bridge; `engine.py` = the Python
twin of `evaluatePatch` (the pinned per-tick loop, batch `run()` +
live `tick()` with per-tick feeds); `gates.py` = the unified atom
catalog (CV + logic + quantum + alogic + h4_slide + sinks);
`program.py` = the AtomicProgram IR (strict validation + 5 compile
targets, auto-views for `viz_*` sinks); `oracle.py` = the node
driver, MODE 1 batch + MODE 2 resident live (per-tick stdio hooks
for live taps/params); `jsnum.py`/`dsp.py` = the JS-fidelity
numerics + MDCT; `tiles.py` = the display model (control frame +
3x3/4x4 agnostic tile wall, resolution derived from the full
display, linked tile groups = larger sub-matrices);
`trace.py` = the flow observer (per-tick stimulus + per-node
in/out/latency rings, snapshot/export, `from_snapshot()` rebuild,
`replay()` re-drives a fresh engine from the recorded stimulus);
`qbf.py` = the Quantum Blob Format (goal 6, iter 8): a self-
describing named-blob container — 64-byte header, u64 offsets/sizes
(no 50 MB tier, no search index to poison), per-blob sha256, and
the H(4) gate as an OPTIONAL per-blob codec flag (the CORE
keystone); `qbfstore.py` = the portable trace archive on top of
it (one .qbf shard per store: index + per-run manifest/ticks/one
blob per node frame; append_run/load_run/flow_trace/replay_run/
export_run; bit-identical replay; dir ~/.runtime/atomic_qbf,
env ATOMIC_QBF_DIR); `ui/` (iter 17) = the FastAPI tile wall
(`atomic.ui.server:app`, port 18094, lifespan auto-registers
7 demo programs) + `viewer.py` (ProgramViewer: batch + per-tick
`tick_once` + WebSocket queue broadcast + live tap/param feed
apply, all snapshotted into `{t, running, bus, series, views}`),
`programs.py` (built-in demo program registry: clock_counter,
gated_clock_counter, sine_lfo_scope, hadamard_wxyz, xy_pad,
wxyz3d_demo, heatmap_demo), `static/index.html` (4x4 canvas tile
wall + control frame + WebSocket live stream + heatmap/series/xy/
wxyz3d renderers, falls back to REST fetch when WS unavailable).
The engine takes an optional `trace=` observer: pure read-only
(touched runs stay bit-identical; off by default).

## Anchors
- This dir is the project root and concept home: `ATOMIC-PC-CORE.md` is the
  single source of truth for the gate list, H(4) row layout, and display-tile
  matrix. Change semantics THERE first, then in code.
- Git remote `https://github.com/bbeartheancient/atomic-computing` (description
  "atomizing data processes into keyword gates") is live on `main` (first push e9cf4f1); vendored `fabric/` and `hoa64/` make a fresh clone standalone.
- Sibling trees: vendored `fabric/` (2.0 MB) and `hoa64/` (8.8 MB) are in-repo; third-party siblings (`Rack`, `WebPd`, `circuitjs1`, `memvid`) remain external and are not vendored.
- Interpreters: `python` 3.11+ (`torch` + `transformers` for local harness; `node` v26 for JSFX oracle; `numpy` for `hoa64`).
- Ports (optional local services): `fabric` :18093, `hoa64` :8765/:8770, `atomic.ui` :18094. Node v26 for JSFX.

## Decision frames
- Import sibling trees; vendored `fabric/` and `hoa64/` are the exception for standalone (others never vendored).
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
- Touched host numpy/afi: `python -m pytest python/tests -q` green (external `afi` repo, not vendored).
- Touched spatial math: hoa64 `hadamard --selftest` passes.
- Touched .mv2 persistence: `cargo test` in `memvid` green AND you honored
  the .mv2 landmines (below). Iter-8 note: goal 6's store is now the
  local pure-Python .qbf — .mv2 is untouched, so this obligation is
  dormant unless you actually edit `memvid`.
- Never restart production services (vllm/fabric) unless the task explicitly
  requires it.
- Do not commit unless asked. If you ARE asked: this dir needs `git init` +
  the remote set first (it is not a repo yet); then one commit per turn.

## Evidence
- Operational landmines + live journal: `fabric/AGENTS.md` (vLLM wedges, EngineCore core dump). Historical numbers: `fabric/docs/AGENTS-archive-*.md`.
- .mv2 traps: the 50 MB tier limit (MV001) is native — the PLE store rotates
  on it; always `enable_vec=False`; bookkeeping goes in tags, not metadata;
  `close_all()` after ~10 create/close cycles. Spec: `memvid/MV2_SPEC.md` (external).
- H(4) measured numbers (energy split, bandwidth tradeoff):
  `fabric/docs/h4-bridge-codec.md` + `h4-layer-gating.md`.
- EEL2 runtime traps: `compile()` must wrap parse output in a block node;
  `Proxy.has` must return true for all globals or patches break.

## Development Tasks (operator-supplied, queued)

### ADD FUNCTION: video generation
1. Duty VLLM already has trained text encoder abilitied for MiniMax H3 (H3
   weights should still be downloaded).
2. H3 FastVideo cuts generation times down to minimum:
   https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo
3. Video generation requests can load the H3 T2VA pipeline on **GPU1** while
   leaving the duty VLLM (Qwen 27B) loaded on **GPU0** — the bicameral split
   becomes a real two-model pipeline, not just two Engines over host RAM
   (`atomic.bridge.HostBridge` stays the wire contract; the GPU pair becomes
   the actual GPU0/GPU1 topology).
5. Video generation pipeline is set up like **"infinite slop"** or
   **"interdimensional cable"** — continuous generation, never-ending feed:
   - https://github.com/reactor-team/infinite-livestream
   - https://haoailab.com/blogs/fasth3-preview/
   The atomic-computing analog: a `viz_wxyz3d` / `viz_video` sink consumes
   the H3 frames as a live stream (per-tick = per-frame), renders onto the
   tile wall, and the trace is captured via `dma_trace` so every frame is
   portable through QBF.
7. Videos created by infinite-livestream are sent to a **Jellyfin** server
   as a randomized live channel — surfaced via **HDHomeRun** or **M3U Tuner**
   so any LAN client can tune in. The atomic-computing analog: a
   `live_export` sink tails the active program, ffmpeg-muxes into an
   HLS/DASH segment under the Jellyfin `Live TV` ingest, and a small
   scheduler rotates the channel so a fleet of `infinite-livestream`
   instances becomes a randomized multichannel universe.

Notes for the implementation turn (do NOT pre-build; the task is queued):
- H3 + FastVideo live in ComfyUI space; the harness treats them as an
  **external GPU0/1 service** (subprocess + a small HTTP/gRPC bridge, not
  vendored). The atomic side only sees `frames: List[bytes]` per tick.
- The "infinite slop" loop is the SAME shape as our `Swarm.run(parallel=True)`
  + `Evolver.run_swarm`: cheap active params => cheap active prompts =>
  parallel video streams. The teacher registry's domain vocab (iter 13)
  becomes the prompt bank; H4 consensus picks the next prompt.
- Jellyfin + HDHomeRun is an OS-level concern (jellyfin package,
  hdhr-stream.c, M3U at `/etc/jellyfin/livetv/*.m3u`); the harness
  documents the topology, does not vendor the stack.
