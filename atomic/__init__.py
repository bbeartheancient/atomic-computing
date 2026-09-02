"""ATOMIC-PC core: live streams routed through typed gate blocks.

This package is the Python twin of the fabric runtime's patch evaluator
(`evaluatePatch` in fabric/web/jsfx.js): one bus, one pinned per-tick
loop, and a catalog of atoms. Every atom is multi-target — one EEL2
body (what the microfx target compiles) plus a Python impl (what the
engine runs) — so the same gate can later compile to WGSL/flowchart.

Step 2 of the build order (ATOMIC-PC-STATE.md): bus + engine twin and
the CV atom catalog, pinned against the 6 conformance patches in
fabric/tests/jsfx_conformance.js.
"""

from .bus import Bus, Node, Port, Wire
from .engine import Engine
from .gates import ATOMS, Atom
from .program import Program, Block, ProgramError, compile_program
from .oracle import run, LiveOracle
from .tiles import Display, Tile, TileGroup, ControlFrame, TileError
from .trace import FrameEntry, FlowTrace, replay, replay_events, latency_histogram_from_trace
from .qbf import (QbfFile, QbfError, RAW, JSON, H4, FLAG_CHECKSUM, FLAG_H4,
                   h4_gate, h4_inverse, h4_encode, h4_decode)
from .qbfstore import QbfTraceStore, open_trace_store, close_all
from .swarm import Agent, Swarm, SwarmResult, PromptBank
from .decompose import decompose_python, decompose_eel2, decompose_expr, DecomposeError
from .teach import (TeacherRegistry, REGISTRY, register_example, match,
                     from_description, domain_vocab, DOMAINS, KEYWORDS_BY_DOMAIN,
                     save_registry, load_registry)
from .evolve import Evolver, EvolveResult
from .bridge import (HostBridge, BicameralPipeline, BicameralResult,
                     h4_row_cosine, row_cos_gate, h4_streaming_metrics,
                     latency_histogram, bridge_benchmark)
from .demo import swarm_evolve_teach_demo
from .video import (H3Client, H3Stub, H3File, H3Source, H3Session, H3Frame,
                     H3Error, PROMPT_BANK_DEFAULT)
from .jellyfin import (JFinExporter, JFinM3U, JFinChannel, JFinScheduler,
                        make_default_channels, DEFAULT_LIVETV_DIR, DEFAULT_HLS_DIR,
                        h4_gate as jfin_h4_gate)
from .gates import register_jfin_scheduler
from .context import (is_available as zg_available, zg_version, query as zg_query,
                      query_rg as zg_query_rg, status as zg_status, assert_retrieval,
                      CANONICAL_QUERIES)
from .trace import replay as trace_replay  # alias for external use

__all__ = ["Bus", "Node", "Port", "Wire", "Engine", "ATOMS", "Atom",
                "Program", "Block", "ProgramError", "compile_program",
                "run", "LiveOracle",
                "Display", "Tile", "TileGroup", "ControlFrame", "TileError",
                "FrameEntry", "FlowTrace", "replay", "replay_events",
                "latency_histogram_from_trace",
                "QbfFile", "QbfError", "RAW", "JSON", "H4",
                "FLAG_CHECKSUM", "FLAG_H4",
                "h4_gate", "h4_inverse", "h4_encode", "h4_decode",
                "QbfTraceStore", "open_trace_store", "close_all",
                "Agent", "Swarm", "SwarmResult", "PromptBank",
                "decompose_python", "decompose_eel2", "decompose_expr", "DecomposeError",
                "TeacherRegistry", "REGISTRY", "register_example", "match",
                "from_description", "domain_vocab", "DOMAINS", "KEYWORDS_BY_DOMAIN",
                "Evolver", "EvolveResult",
                "HostBridge", "BicameralPipeline", "BicameralResult",
                "h4_row_cosine", "row_cos_gate", "h4_streaming_metrics",
                "latency_histogram", "bridge_benchmark",
                "swarm_evolve_teach_demo",
                "zg_available", "zg_version", "zg_query", "zg_query_rg", "zg_status",
                "assert_retrieval", "CANONICAL_QUERIES",
                "H3Client", "H3Stub", "H3File", "H3Source",
                "H3Session", "H3Frame", "H3Error", "PROMPT_BANK_DEFAULT",
                "JFinExporter", "JFinM3U", "JFinChannel", "JFinScheduler",
                "make_default_channels", "register_jfin_scheduler",
                "DEFAULT_LIVETV_DIR", "DEFAULT_HLS_DIR"]
