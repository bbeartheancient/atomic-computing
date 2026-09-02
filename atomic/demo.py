"""Demo: swarm -> evolve -> teach -> QBF (iter 14, goal chain 4/5/6/7/10).

"The LLM assembles apps by matching function paths through gates."
This module closes the loop end-to-end:

  swarm (goal 10, cheap active params, tile-parallel)
    -> evolve (goal 7, hill-climb per agent, swarm-parallel scoring)
    -> teach (goal 4/5, promote best to domain registry, swappable vocab)
    -> QBF (goal 6, persist registry via save_registry, no 50 MB wall)

No LLM endpoint required: the fitness is a deterministic scalar
(H4 W consensus or a domain gain target) and the registry routing is
keyword-weighted (teach.REGISTRY).

API:
  swarm_evolve_teach_demo(ticks=20, generations=6, population=3,
                          shard_dir=None, path=None) -> dict
  - ticks: per-agent run ticks before evolve
  - generations: evolver generations per agent (hill-climb)
  - population: per-generation mutants when using swarm-parallel evolve
  - shard_dir / path: where to persist the registry QBF (temp if None)
  Returns summary dict with swarm consensus, best scores, registry
  size, persisted path, and replay check.
"""

from __future__ import annotations

import os
import tempfile

from .program import Block, Program, Wire
from .swarm import Agent, Swarm, SwarmResult
from .tiles import Display
from .evolve import Evolver
from .teach import TeacherRegistry, save_registry, load_registry
from .qbf import h4_gate


def _domain_program(domain: str, factor: float) -> Program:
    """Small domain-flavoured program: const -> gain -> threshold -> viz."""
    if domain == "spatial":
        # spatial: h4_slide in chain
        return Program(
            f"demo_{domain}_{factor}",
            blocks=[
                Block("c0", "const", {"value": 2.0}),
                Block("g1", "gain", {"factor": float(factor)}),
                Block("h1", "h4_slide"),
                Block("v0", "viz_series"),
            ],
            wires=[
                Wire("c0.cv", "g1.in"),
                Wire("g1.cv", "h1.in"),
                Wire("h1.w", "v0.in"),
            ],
            description=f"spatial demo gain {factor}",
        )
    if domain == "medical":
        return Program(
            f"demo_{domain}_{factor}",
            blocks=[
                Block("sig", "sensor"),
                Block("th", "threshold", {"hi": 0.8, "lo": 0.2}),
                Block("g1", "gain", {"factor": float(factor)}),
                Block("v0", "viz_series"),
            ],
            wires=[
                Wire("sig.cv", "th.in"),
                Wire("th.gate", "g1.in"),
                Wire("g1.cv", "v0.in"),
            ],
            description=f"medical threshold gain {factor}",
        )
    # general/signal/control: simple gain chain
    return Program(
        f"demo_{domain}_{factor}",
        blocks=[
            Block("c0", "const", {"value": 5.0}),
            Block("g1", "gain", {"factor": float(factor)}),
            Block("v0", "viz_series"),
        ],
        wires=[
            Wire("c0.cv", "g1.in"),
            Wire("g1.cv", "v0.in"),
        ],
        description=f"{domain} gain {factor}",
    )


def swarm_evolve_teach_demo(ticks: int = 20, generations: int = 6,
                            population: int = 3, shard_dir=None, path=None,
                            seed: int = 0):
    """Run the full chain and persist the teach registry to QBF.

    Returns dict with keys:
      swarm_result, consensus, best_hash, best_score, registry_size,
      persisted_path, load_ok, replay_match
    """
    from .engine import Engine  # local import to avoid cycles

    # 1) Display wall + Swarm (tile-parallel)
    display = Display(1200, 1200, 3, 3, frame_h=0)
    swarm = Swarm(display=display)
    domains = ["signal", "spatial", "medical", "control"]
    factors = [1.0, 1.5, 2.0, 2.5]
    for i, (dm, fac) in enumerate(zip(domains, factors)):
        prog = _domain_program(dm, fac)
        r, c = i % 3, i // 3
        grp = display.link(f"agent{i}", r, c, 1, 1)
        # give each agent its own trace for later heatmap demo
        swarm.add_agent(Agent(f"agent{i}", prog, tile_group=grp))

    # 2) Swarm parallel run: consensus is H4 W over up to 4 agents' g1.cv
    swarm_result: SwarmResult = swarm.run(ticks, parallel=True)
    consensus = float(swarm_result.consensus("g1.cv"))

    # verify parallel == serial determinism for this demo
    serial_result = swarm.run(ticks, parallel=False)
    assert all(
        swarm_result.by_id(f"agent{i}")["final"].get("g1.cv") ==
        serial_result.by_id(f"agent{i}")["final"].get("g1.cv")
        for i in range(4)
    ), "swarm parallel/serial diverged"

    # 3) Evolve: pick the best agent's program and hill-climb toward gain=2
    # (target domain signal). Fitness = -|g1.cv - 10| so best factor ~2 when c0=5.
    # The swarm's cheap param idea: each agent is a candidate; we evolve the
    # signal one in swarm-parallel mode.
    base_prog = _domain_program("signal", 1.0)

    def fitness(final):
        # final contains g1.cv after bias latency: c0=5 -> g1=5*factor
        return -abs(float(final.get("g1.cv", 0)) - 10.0)

    evolver = Evolver(base_prog, fitness, seed=seed, ticks=ticks)
    start_score = evolver.best_score
    # swarm-parallel evolution (population mutants scored concurrently)
    evolver.run_swarm(generations=generations, population=population, parallel=True)
    best_prog = evolver.best
    best_score = evolver.best_score

    # also verify swarm vs serial determinism of evolve (same mutants, different scoring)
    # by re-running serial and comparing best hash for same seed
    ev_check = Evolver(base_prog, fitness, seed=seed, ticks=ticks)
    ev_check.run_swarm(generations=generations, population=population, parallel=False)
    assert ev_check.best.hash == evolver.best.hash, "evolve parallel/serial divergence"

    # 4) Teach: promote best into a fresh registry under domain 'signal'
    registry = TeacherRegistry()
    # carry over existing seeded examples for realism? start fresh for demo isolation
    # register the demo best with a description that matches domain routing
    desc = f"signal filter smooth gain tuned {best_prog.hash}"
    registry.register(desc, best_prog, domain="signal", difficulty="tuned")
    # also register a second domain example to show swappable sets
    registry.register("spatial h4 wxyz scope tuned", _domain_program("spatial", 2.0), domain="spatial")
    registry_size = len(registry.examples)

    # 5) QBF persist: swappable domain vocab sets travel as one .qbf file
    if path is None and shard_dir is None:
        td = tempfile.mkdtemp(prefix="demo_qbf_")
        path = os.path.join(td, "demo_registry.qbf")
    persisted = save_registry(path=path, shard_dir=shard_dir,
                              store_name=None if path else "demo_registry",
                              registry=registry)

    # 6) Verify load round-trip and match routing
    # persisted may be a shard path or a .qbf file path
    loaded = load_registry(persisted, shard_dir=shard_dir) if isinstance(persisted, str) else load_registry(path, shard_dir=shard_dir)
    # if shard_dir used, persisted is shard dir file; loader handles both .qbf and shard name
    try:
        # try file path first
        if os.path.exists(persisted):
            loaded2 = TeacherRegistry.load_qbf(persisted)
            load_ok = loaded2.list(domain="signal") != []
        else:
            load_ok = loaded.list(domain="signal") != []
    except Exception:
        load_ok = loaded.list(domain="signal") != [] if hasattr(loaded, "list") else False

    # 7) Replay check: best program engine determinism
    patch = best_prog.compile("microfx")
    r1 = Engine(patch["modules"], patch["wires"]).run(ticks)
    r2 = Engine(patch["modules"], patch["wires"]).run(ticks)
    replay_match = r1["final"] == r2["final"]

    return {
        "swarm_result": swarm_result,
        "consensus": consensus,
        "start_score": start_score,
        "best_hash": best_prog.hash,
        "best_score": best_score,
        "improved": best_score > start_score,
        "registry_size": registry_size,
        "persisted_path": persisted,
        "load_ok": load_ok,
        "replay_match": replay_match,
        "registry": registry,
        "best_program": best_prog,
    }
