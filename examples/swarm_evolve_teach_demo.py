"""Swarm -> evolve -> teach -> QBF (iter 14-17 chain, goals 4/5/6/7/10).

The end-to-end bicameral demo wrapped from atomic.demo:

  - 4 agents on a 3x3 Display wall, each a small domain program.
  - Swarm parallel run: all 4 agents run concurrently (ThreadPoolExecutor).
  - H(4) consensus over g1.cv (W = amplitude = dominant ~61% energy).
  - Parallel == serial determinism (pinned by the harness).
  - Evolver: hill-climb the "signal" agent toward gain=10 (fitness
    minimises |g1.cv - 10|).
  - Swarm-parallel evolution: population mutants scored concurrently.
  - Teach: promote the best evolved program into the signal-domain
    registry, persist it as a .qbf shard.
  - QBF round-trip: load_registry -> TeacherRegistry, verify routing.
  - replay determinism: two fresh engines with the same patch produce
    identical final buses.

  ~/runtime/.venv/bin/python -m examples.swarm_evolve_teach_demo
"""

from __future__ import annotations

from atomic.demo import swarm_evolve_teach_demo as _raw


def demo(ticks: int = 20, generations: int = 4, population: int = 3,
         seed: int = 42):
    r = _raw(ticks=ticks, generations=generations,
             population=population, seed=seed)

    assert r["replay_match"], "replay determinism check failed"
    assert r["load_ok"], "QBF registry load failed"
    assert r["registry_size"] >= 2, "registry too small: %d" % r["registry_size"]
    assert r["improved"], "evolver did not improve: %s > %s" % (
        r["best_score"], r["start_score"])

    print("[swarm evolve teach demo] ok")
    print("  consensus (H4 W):  %.4f" % r["consensus"])
    print("  start score:      %.4f" % r["start_score"])
    print("  best score:       %.4f" % r["best_score"])
    print("  improved:         %s" % r["improved"])
    print("  best hash:        %s" % r["best_hash"])
    print("  registry size:    %d" % r["registry_size"])
    print("  persisted:        %s" % r["persisted_path"])
    print("  replay match:     %s" % r["replay_match"])
    print("  load ok:         %s" % r["load_ok"])

    return r


if __name__ == "__main__":
    demo()
