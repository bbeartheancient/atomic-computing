"""Examples: small end-to-end demos of the atomic harness (iter 18).

These are runnable scripts that exercise the harness from the outside:
  - the QBF round-trip: snapshot a trace, archive, load, replay
    (verifies goal 6 end-to-end without the 50 MB cap).
  - the Hadamard W/X/Y/Z scope: h4_slide on a 4x4 tile wall, every
    channel fed into a viz_series (verifies the keystone H4 gate).
  - the gated clock counter: from_description -> compile -> run, the
    canonical "LLM assembles apps by matching function paths through
    gates" demo.
  - the swarm->evolve->teach->QBF chain: the demo.swarm_evolve_teach_demo
    wrapper plus a short assertions layer.

Run any from the repo root with the harness interpreter, e.g.
  ~/runtime/.venv/bin/python -m examples.qbf_persistence_round_trip
  ~/runtime/.venv/bin/python -m examples.hadamard_wxyz_scope
  ~/runtime/.venv/bin/python -m examples.gated_clock_counter
  ~/runtime/.venv/bin/python -m examples.swarm_evolve_teach_demo

Each module is standalone: `__init__.py` is intentionally empty so
`python -m examples.<name>` doesn't double-import and warn.
"""

__all__ = []
