ATOMIC COMPUTING CONCEPTS

- atomic computing builds applications from DMA functions and simple gates
- allows LLMs to build autonomously by functions instead of through code languages

** all data sources are considered live streams 

H4 gate is for spatializing related data streams (like tensors in an LLM):
    Row0 = Full Attention Layer, W vector [+, +, +, +] (amplitude of spatial field)
    Row1 = Linear Attention Layer 1, Z Vector [+, -, +, -] (linear front/back)
    Row2 = Linear Attention Layer 2, Y Vector [+, +, -, -] (linear up/down)
    Row3 = Linear Attention Layer 3, X Vector [+, -, -, +] (linear left/right)
    
    NOTE: When using for audio applications, matrix vectors are shifted by 90 degrees:
        W = amplitude
        Z = linear up/down
        Y = linear left/right
        X = linear front/back
               
    -  see FOA/HOA formulas for higher dimensional calculations

    
USING rh.py AS VALIDATOR

LOGIC:

    the rh.py script validates the zeroes of the H4 gate:

        "If the true zeros lie on the critical line (sigma_n = 1/2, tau_n = t_n) then
        xi_n(1-xi_n) = rho_n(1-rho_n) and every term vanishes => bound = 0 exactly,
        confirming consistency with RH."

    - the Hadamard library `hoa64` (vendored) has passed `rh.py` validation

    - so for any data passing through the H4 gate, the same critical line should apply        
    
LOGIC GATES FOR GATE MACROS:
    - AND: Outputs 1 only if all inputs are 1.
    - OR: Outputs 1 if at least one input is 1.
    - NOT: Takes a single input and inverts it (0 becomes 1, and 1 becomes 0).
    - NAND: The opposite of an AND gate; outputs 0 only if all inputs are 1.
    - NOR: The opposite of an OR gate; outputs 1 only if all inputs are 0.
    - XOR (Exclusive OR): Outputs 1 if the inputs are different.
    - XNOR (Exclusive NOR): Outputs 1 if the inputs are the same.
    
    Extra Gate Functions added from
        - https://en.wikipedia.org/wiki/Logic_gate - include IMPLY, NIMPLY, and all Electronic Gates
        - https://en.wikipedia.org/wiki/Quantum_logic_gate - include Identity, Pauli X/Y/Z, CNOT, CU, Phase Shift, Hadamard, Swap, and Toffoli gates
        - Toffoli + Hadamard Gate is a universal 3D exponential (quaternion) to linear function gate


DISPLAY TILES:
    - Atomic Computing apps use the same pre-defined matrix of display tiles as the display output
        - Display has control frame with top-level i/o controls
        - GUI display uses 3x3 or 4x4 tiled matrix of framebuffer outputs
        - framebuffer outputs are agnostic and universal
        - full display resolution determines matrix tile resolution 
        - works like a tiled video wall
        - multiple tiles can be linked together for larger submatrix displays
        - Iter 14: tiles wire to bridge metrics and swarm: HostBridge latency
          histogram -> per-tile counts (to_tiles), BicameralPipeline metrics()
          expose histogram + H4 sample, TileGroup/Display.heatmap_from_trace()
          and heatmap_from_swarm() render live trace replay onto the wall
          (normalized 0..1 per tile, group.apply_heatmap).

H4 STREAMING (iter 14):
    - H4 row_cos gate: cosine similarity between H4 rows (W/Z/Y/X) across
      groups; avg |cos| < 0.1 => rows remain orthogonal => compact passes.
      Helpers: h4_row_cosine(a,b), row_cos_gate(groups, threshold=0.1),
      h4_streaming_metrics(groups) -> {w_energy_frac, row_cos_avg, row_cos_pass, w_mean}.
      W dominance ~61% on random streams (pinned); bridge's H4 compact flag
      stores W/Z/Y/X with h4_gate/h4_inverse lossless within 1e-6.
    - Bridge benchmark: HostBridge.benchmark(ticks, payload_keys) + latency_histogram()
      + BicameralPipeline.metrics() -> host-RAM FIFO throughput (ticks/s),
      queued depth, arrival histogram; wired to tiles viz above.

WGSL (iter 11/14/15):
    - Program.to_wgsl() emits a compute shader (@compute @workgroup_size(64),
      @group(0) bus/params/state/bridge) with per-block fn tick_<id> and
      host-RAM bridge comment ("no P2P, tick latency 1"). Iter 14 pins the
      shape still starts with "// WGSL" and contains host-RAM + tick refs.
      Iter 15: Display.validate_wgsl() / Program.validate_wgsl() run the
      structural check (and `naga` CLI if on PATH); tiles heatmap_animation
      drives per-tick heat replay onto the wall.

DECOMPOSE (iter 14/15):
    - decompose_python now covers full AST sweep: For/AsyncFor/While,
      ListComp/SetComp/DictComp/GeneratorExp, ClassDef (walk methods),
      FunctionDef/AsyncFunctionDef inner loops, Try/With/Assert/Delete,
      Lambda/JoinedStr/Starred/Await/Yield. Unknown nodes still fallback
      to gain passthrough so every snippet validates.
    - Iter 15 audit: async def/await, walrus `:=`, f-strings, comprehensions,
      starred/subscript/yield all validated against engine fidelity (pure
      arithmetic cases pin 7-2->5, 8/2->4); EEL2 multi-input via wire SUM.

SWARM (iter 15):
    - 16 agents on a 4x4 wall: Display(1600,1600,4,4) partitioned into 16
      single-tile groups, Swarm.run(parallel) == serial, consensus W over
      first 4 agents (e.g. 1+2+3+4=10), row_cos_gate + h4_streaming_metrics
      under load (W dominance pinned).

TRACE → QBF → REPLAY (iter 15):
    - FlowTrace at 10k frames (5000 ticks x 2 mods) round-trips via
      QbfTraceStore (10k frame blobs + manifest/ticks), replay bit-identical,
      ring wrap evicts oldest (seq continues), latency_histogram_from_trace()
      bins per-node latency_us, H4 codec lossless at 1e-5 over 1000 groups.

TILES LIVE VIZ (iter 15):
    - Display.heatmap_animation(trace, port, window) animates replay per tick
      (window ticks per frame, normalized 0..1 per tile) and swarm 16 heatmap;
      Display.validate_wgsl structural + naga.

RETRIEVAL (iter 16, zvec-grep):
    - Workspace-aware semantic retrieval via `zg` (npm @zvec/zvec-grep 0.2.1,
      model local/potion-code-16m-v2, 256-dim cosine, 42 files / 521 entities,
      100% coverage). Install: `npm install -g @zvec/zvec-grep --prefix ~/.local`
      + `zg --version`. MCP: `zg install --target opencode --yes` writes stdio
      server to `~/.config/opencode/opencode.json` + guidance to `AGENTS.md`
      (zvec_grep_search / zvec_grep_rg). Index: `zg index .`
      `--embedding local/potion-code-16m-v2` -> `.zvec-grep/index.zvec`
      (cached model under `~/.cache/zvec`; background refresh to 42/42).
      Query CLI: `zg query "H4 gate row layout" --human` hybrid FTS+vector
      + `zg query --rg -n "h4_gate"` managed ripgrep; Agent MCP same via
      `zvec_grep_search` / `_rg` (root = repo root). Wrapper
      `atomic/context.py` (`query`/`query_rg`/`is_available`/`zg_version`/
      `assert_retrieval`/`CANONICAL_QUERIES`) keeps harness stdlib-only and
      stays green when zg absent (skip with clear message). Fidelity pinned:
      "H4 gate row layout and W dominance" -> ATOMIC-PC-CORE.md, "tile heatmap
      animation from trace replay" -> tiles.py, "QBF portable store ring wrap" -> qbf.
        
    
REFERENCES:

    Pure Data - Node-based programming language
        - https://github.com/mganss/pdjs - JavaScript External for pd
        - https://github.com/sebpiq/webpd - WebPd ; run pd patches on Web
        - https://github.com/pure-data/pure-data - pd source code
        
    JSFX Programming - Scripting framework that can be applied to atomized functions
        - https://www.reaper.fm/sdk/js/js.php - JSFX Reference
        
    x86_64 Assembly - Assembly and ESIMD are the lowest-level programming layers
        - https://github.com/luamfb/intro_x86-64 - intro to x86-64 assembly
        
    EEL2 - buffer link for standard and multidimensional data streams:
        - https://www.cockos.com/EEL2/ - EEL2 reference 
        - `fabric/` (vendored) — bicameral functions and application
    
    hoa64 - dimensional calculator library
        - `hoa64/` (vendored) — Python library for spatial Hadamard/HOA
        
    TouchOSC - Used as framework for quickly generating app control surfaces
        - https://github.com/wcolding/pyTOSC - Python TouchOSC templates
        - https://github.com/theisaaclloyd/touchosc-layouts - TouchOSC Layouts
        - https://openstagecontrol.ammd.net/docs/getting-started/introduction/ - Open Stage Control, an open source version of the TouchOSC ecosystem
            - https://openstagecontrol.ammd.net/download/ - OSC download
    
    Memvid - converts information into .mv2 files
        - `memvid` (external) — source code (50 MB tier limit noted; superseded by QBF)
        
    XMX/Xe/ESIMD Kernel Development - For programming GPU functions
        - https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/overview.html - oneAPI optimization guide
        - https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-explicit-simd-kernels.html - optimizing explicit kernels
        - https://github.com/steveseguin/b70-optimization-lab/tree/main - Unofficial B70 optimization lab
            
    Three.JS Skills - for easier 3D visualization
        - https://github.com/cloudai-x/threejs-skills 
        
    Blender MCP - for Blender integration
        - https://www.blender.org/lab/mcp-server/ - Official MCP server
        
    Previous Research:
        - `fabric/docs/` — experimental methods of achieving atomic computing
        

    
