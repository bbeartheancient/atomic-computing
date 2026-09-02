"""Evolve: self-improvement loops (goal 7).

The model learns from its own traces: given a fitness function over
engine outputs, the evolver proposes parameter mutations, scores them
by running the harness engine (the verified twin), and keeps the best
program. History is persisted to the QBF trace store if desired, but
the core loop is pure Python + deterministic RNG — no LLM endpoint.

API:
  Evolver(base_program, fitness_fn, seed=0)
    fitness_fn(final: dict) -> float  # higher is better
    mutate(program) -> Program        # jitter numeric params
    score(program) -> float
    step() -> (new_best, scored)
    run(generations=20) -> best Program
    history: list of {gen, score, hash}
    best: Program
    best_score: float

Mutations are minimal so every child passes Program.validate():
  - pick one numeric param and jitter by +/- 10% (or +/- 0.2 absolute
    if the param is near zero)
  - never changes topology (blocks/wires/ids) — only param values.
  - deterministic via seeded random.Random.

This is the harness incarnation of goal 7's self-improvement loop:
the swarm cheaply evaluates many agents (goal 10) and the evolver
promotes the winning program into the teacher registry (goal 4).
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace

from .engine import Engine
from .program import Block, Program, Wire

__all__ = ["Evolver", "EvolveResult"]


class EvolveResult:
    def __init__(self, program, score, gen):
        self.program = program
        self.score = float(score)
        self.gen = int(gen)

    def __repr__(self):
        return "EvolveResult(gen=%d score=%.6f hash=%s)" % (self.gen, self.score, self.program.hash)


_NUMERIC_PARAMS = {"value", "factor", "add", "alpha", "hi", "lo", "thresh",
                   "bpm", "rate_hz", "amp", "offset", "per_tick", "wrap",
                   "n", "initial", "family", "cell_b", "cell_c"}


def _jitter(value, rng):
    # +/- 10% relative, or +/-0.2 absolute near zero
    if abs(value) < 1e-9:
        return rng.uniform(-0.2, 0.2)
    # for integer-like params (n, family) keep integer
    delta = value * 0.1
    return value + rng.uniform(-delta, delta)


def _clone_program(prog: Program) -> Program:
    blocks = [Block(b.id, b.primitive, dict(b.params), b.label) for b in prog.blocks]
    wires = [Wire(w.src, w.dst) for w in prog.wires]
    return Program(prog.name, blocks=blocks, wires=wires,
                   views=list(prog.views), entry=prog.entry,
                   tags=list(prog.tags), description=prog.description)


def _mutate_one(prog: Program, rng: random.Random) -> Program:
    child = _clone_program(prog)
    # collect mutable numeric params
    candidates = []
    for b in child.blocks:
        for k, v in list(b.params.items()):
            if isinstance(v, (int, float)) and k in _NUMERIC_PARAMS:
                candidates.append((b, k))
    if not candidates:
        return child
    block, key = rng.choice(candidates)
    old = float(block.params[key])
    new = _jitter(old, rng)
    # clamp integer params
    if key in ("n", "family", "cell_b", "cell_c", "initial"):
        new = int(round(new))
        if key == "n":
            new = max(1, min(64, new))
        elif key == "family":
            new = max(0, min(5, new))
        elif key in ("cell_b", "cell_c"):
            new = max(0, min(1, new))
        elif key == "initial":
            new = max(0, min(1, new))
    elif key == "alpha":
        new = max(0.01, min(0.99, new))
    elif key == "bpm":
        new = max(1, min(600, new))
    block.params[key] = float(new) if not isinstance(new, int) else new
    # recompute hash
    child.hash = child.compute_hash()
    return child


class Evolver:
    """Hill-climber over program params."""

    def __init__(self, base_program: Program, fitness_fn, seed=0, dt=1.0/30.0, ticks=60):
        self.base = base_program
        self.fitness_fn = fitness_fn
        self.rng = random.Random(seed)
        self.dt = float(dt)
        self.ticks = int(ticks)
        self.best = _clone_program(base_program)
        self.best_score = self.score(self.best)
        self.gen = 0
        self.history = [{"gen": 0, "score": self.best_score, "hash": self.best.hash}]

    def score(self, prog: Program) -> float:
        patch = prog.compile("microfx")
        res = Engine(patch["modules"], patch["wires"],
                     views=patch.get("views") or [], dt=self.dt).run(self.ticks)
        return float(self.fitness_fn(res["final"]))

    def mutate(self, prog: Program) -> Program:
        return _mutate_one(prog, self.rng)

    def step(self):
        self.gen += 1
        child = self.mutate(self.best)
        # ensure child is valid; if not, keep best
        if child.validate():
            sc = float("-inf")
        else:
            sc = self.score(child)
        improved = sc > self.best_score
        if improved:
            self.best = child
            self.best_score = sc
        self.history.append({"gen": self.gen, "score": sc,
                             "hash": child.hash, "kept": improved})
        return EvolveResult(child if improved else self.best,
                            sc if improved else self.best_score, self.gen)

    def run(self, generations=20):
        for _ in range(int(generations)):
            self.step()
        return self.best

    def run_swarm(self, generations=20, population=4, parallel=True):
        """Parallel swarm evaluation: each generation tests `population`
        mutants concurrently (ThreadPool), keeps the best. Mirrors the
        swarm's cheap-active-params insight (goal 10): small graphs
        evaluate cheaply in host threads. Deterministic when the RNG
        seed is fixed and parallel=False; with parallel=True the set of
        mutants is still drawn sequentially (so the population is
        identical), only scoring is concurrent — so the best is
        deterministic either way.
        """
        import concurrent.futures as _cf
        for _ in range(int(generations)):
            self.gen += 1
            # draw population sequentially for determinism
            children = [self.mutate(self.best) for _ in range(int(population))]
            # filter invalid early
            valid = [(c, c.validate()) for c in children]
            # score valid in parallel
            scores = {}
            def _score_one(item):
                c, errs = item
                if errs:
                    return (c.hash, float("-inf"))
                return (c.hash, self.score(c))
            if parallel and len(valid) > 1:
                with _cf.ThreadPoolExecutor(max_workers=len(valid)) as ex:
                    futs = {ex.submit(_score_one, item): item for item in valid}
                    for fut in _cf.as_completed(futs):
                        h, sc = fut.result()
                        scores[h] = sc
            else:
                for item in valid:
                    h, sc = _score_one(item)
                    scores[h] = sc
            # pick best of this generation
            best_child = None
            best_sc = float("-inf")
            for c, _ in valid:
                sc = scores.get(c.hash, float("-inf"))
                if sc > best_sc:
                    best_sc = sc
                    best_child = c
            improved = best_sc > self.best_score
            if improved and best_child is not None:
                self.best = best_child
                self.best_score = best_sc
            self.history.append({"gen": self.gen, "score": best_sc if best_child else float("-inf"),
                                 "hash": best_child.hash if best_child else self.best.hash,
                                 "kept": improved, "population": int(population)})
        return self.best

    def promote_to_registry(self, description="", domain="general"):
        """Copy best program into the teacher registry (goal 4)."""
        from .teach import REGISTRY
        desc = description or self.base.description or self.base.name
        return REGISTRY.register(desc, self.best, domain=domain)

    def save_history(self, path=None, shard_dir=None, store_name=None):
        """Persist evolve history + best program to a QBF trace shard (goal 6).

        If store_name is given, writes to a named .qbf shard under shard_dir
        (DEFAULT_DIR) as JSON blobs: 'evolve/history' + 'evolve/best' (patch).
        If path is given, writes a plain JSON file. Returns the written path
        or manifest.
        """
        import json as _json, os as _os, tempfile as _tf
        from pathlib import Path as _P
        payload = {"history": list(self.history), "best": self.best.compile("microfx"),
                   "best_score": self.best_score, "ticks": self.ticks}
        if store_name:
            from .qbfstore import open_trace_store as _ots
            store = _ots(store_name, shard_dir=shard_dir)
            store.file.put_json("evolve/history", self.history)
            store.file.put_json("evolve/best", payload["best"])
            store.file.put_json("evolve/meta", {"best_score": self.best_score, "ticks": self.ticks})
            store.file.write()
            return str(store.path)
        if path:
            _P(path).write_text(_json.dumps(payload, indent=2))
            return str(path)
        # default: temp QBF
        from .qbf import QbfFile as _Q
        td = _tf.mkdtemp(prefix="evolve_")
        p = _os.path.join(td, "evolve.qbf")
        f = _Q.create(p)
        f.put_json("evolve/history", self.history)
        f.put_json("evolve/best", payload["best"])
        f.put_json("evolve/meta", {"best_score": self.best_score, "ticks": self.ticks})
        f.write()
        return p

    @classmethod
    def load_history(cls, path_or_store, shard_dir=None):
        """Load a history saved by save_history (QBF path or shard name)."""
        import json as _json
        from pathlib import Path as _P
        from .qbf import QbfFile as _Q
        p = _P(str(path_or_store))
        if p.exists() and p.suffix == ".qbf":
            f = _Q.open(str(p))
            return {"history": f.get_json("evolve/history"),
                    "best": f.get_json("evolve/best"),
                    "meta": f.get_json("evolve/meta")}
        # try as shard name
        if shard_dir is not None or not p.exists():
            from .qbfstore import open_trace_store as _ots
            try:
                store = _ots(str(path_or_store), shard_dir=shard_dir)
                f = store.file
                if "evolve/history" in f:
                    return {"history": f.get_json("evolve/history"),
                            "best": f.get_json("evolve/best"),
                            "meta": f.get_json("evolve/meta")}
            except Exception:
                pass
        # plain json
        if p.exists():
            return _json.loads(p.read_text())
        raise FileNotFoundError(str(path_or_store))
