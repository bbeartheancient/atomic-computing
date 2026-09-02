"""Swarm: cheap active params => parallel agent swarms (goal 10).

The bicameral payoff: because each agent is a small atomic program
(a handful of gates, deterministic, replayable), many can run in
parallel on the same host. The swarm is a *tile-parallel* fabric:

  - one Display wall (3x3 or 4x4) whose tiles are partitioned among
    agents (each agent owns a TileGroup);
  - each agent is an isolated Engine (its own bus, state, trace);
  - the swarm runs all agents concurrently (ThreadPoolExecutor) and
    gathers their final buses;
  - consensus is the W channel of an H(4) gate over the agents' scalar
    outputs (amplitude/consensus = dominant ~61% energy).

No GPU needed for the harness: the "cheap active params" idea is
realized as small program graphs run in host threads.

Key invariants:
  - agents are ISOLATED: no bus aliasing, no shared mem.
  - determinism per agent (contract 8) => swarm determinism.
  - tile assignment is exclusive: two agents cannot own overlapping
    tiles (enforced via Display.link).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json

from .engine import Engine
from .qbf import h4_gate

__all__ = ["Agent", "Swarm", "SwarmResult"]


class Agent:
    """One swarm member: a named program bound to a tile group."""

    def __init__(self, agent_id, program, tile_group=None, dt=1.0/30.0,
                 ui_taps=None, feeds=None, views=None):
        self.agent_id = str(agent_id)
        self.program = program  # atomic.program.Program
        self.tile_group = tile_group
        self.dt = float(dt)
        self.ui_taps = list(ui_taps or [])
        self.feeds = dict(feeds or {})
        self.views = list(views or [])

    def _patch(self):
        p = self.program.compile("microfx") if hasattr(self.program, "compile") else self.program
        # already a patch dict?
        if isinstance(p, dict) and "modules" in p:
            return p
        return p

    def run(self, ticks):
        patch = self._patch()
        eng = Engine(patch["modules"], patch.get("wires", []),
                     views=patch.get("views", []) or self.views,
                     dt=self.dt, ui_taps=self.ui_taps, feeds=self.feeds)
        res = eng.run(ticks)
        return {"agent_id": self.agent_id, "final": res["final"],
                "series": res["series"], "bus": res["bus"]}


class SwarmResult:
    """Gathered outputs of one swarm tick-run."""

    def __init__(self, results, display=None):
        self.results = list(results)  # list of per-agent dicts
        self.display = display

    def by_id(self, agent_id):
        for r in self.results:
            if r["agent_id"] == agent_id:
                return r
        raise KeyError(agent_id)

    def scalar(self, agent_id, port="g1.cv"):
        """Convenience: final bus scalar for one agent."""
        return self.by_id(agent_id)["final"].get(port)

    def consensus(self, port="g1.cv"):
        """H(4) consensus (W) over up to 4 agents' scalar port values.

        Pads to 4 with 0.0; W = sum of the four values.
        """
        vals = []
        for r in self.results[:4]:
            v = r["final"].get(port, 0.0)
            try:
                vals.append(float(v))
            except Exception:
                vals.append(0.0)
        while len(vals) < 4:
            vals.append(0.0)
        w, _, _, _ = h4_gate(tuple(vals))
        return w

    def summary(self):
        return {"n_agents": len(self.results),
                "ids": [r["agent_id"] for r in self.results]}


class Swarm:
    """A tile-parallel swarm of atomic agents."""

    def __init__(self, display=None, max_workers=None):
        from .tiles import Display
        self.display = display
        self.agents = {}
        self.max_workers = max_workers

    def add_agent(self, agent):
        if agent.agent_id in self.agents:
            raise ValueError("duplicate agent id %r" % agent.agent_id)
        # tile exclusivity already enforced by Display.link; just track name
        if agent.tile_group is not None and self.display is not None:
            name = agent.tile_group.name if hasattr(agent.tile_group, "name") else str(agent.tile_group)
            for other in self.agents.values():
                og = other.tile_group
                if og is None:
                    continue
                on = og.name if hasattr(og, "name") else str(og)
                if on == name:
                    raise ValueError("tile group %r already owned" % name)
        self.agents[agent.agent_id] = agent
        return agent

    def remove_agent(self, agent_id):
        return self.agents.pop(agent_id, None)

    def run(self, ticks, parallel=True):
        """Run all agents for `ticks` ticks (parallel by default)."""
        agents = list(self.agents.values())
        if not agents:
            return SwarmResult([], display=self.display)
        if parallel and len(agents) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers or len(agents)) as ex:
                futs = {ex.submit(a.run, ticks): a.agent_id for a in agents}
                results = []
                for fut in concurrent.futures.as_completed(futs):
                    results.append(fut.result())
                # deterministic order by agent_id
                results.sort(key=lambda r: r["agent_id"])
        else:
            results = [a.run(ticks) for a in sorted(agents, key=lambda x: x.agent_id)]
        return SwarmResult(results, display=self.display)

    def __len__(self):
        return len(self.agents)

    def __repr__(self):
        return "Swarm(%d agents)" % len(self.agents)


# -- iter 27: prompt bank + H4 consensus routing ---------------------------

class PromptBank:
    """A bank of prompts + an H4-consensus picker for the next prompt.

    The bank is a list of strings (each entry is a "domain prompt");
    `consensus_pick(prev_prompts, prev_results)` runs the H(4) gate over
    the LAST 4 PREVIOUS PROMPTS (their hash-derived 4-tuple), the
    W (consensus) channel maps to an index in the bank, and that
    prompt is returned.

    This is the cheap-active-params the H3 video pipeline taps: the
    swarm's scalars drive which prompt H3 renders next. Same shape as
    `Swarm.consensus()` but applied to text -> bank indices.
    """

    def __init__(self, prompts=None):
        from .video import PROMPT_BANK_DEFAULT
        self.prompts = list(prompts) if prompts is not None else list(PROMPT_BANK_DEFAULT)

    def __len__(self):
        return len(self.prompts)

    def _h4_of_prompts(self, prev_prompts):
        # hash each -> 4-tuple of float in [0, 1) -> H(4) gate
        from .qbf import h4_gate
        vals = []
        for p in (prev_prompts or self.prompts)[:4]:
            h = int(hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:8], 16)
            vals.append((h % 1000) / 1000.0)
        while len(vals) < 4:
            vals.append(0.0)
        w, z, y, x = h4_gate(tuple(vals))
        return w, z, y, x

    def consensus_pick(self, prev_prompts=None, last_w=None):
        """H4 consensus over previous prompts -> next bank index.

        `last_w` (optional) overrides the W channel directly -- the
        caller can supply a swarm's W-channel consensus value (e.g. the
        outputs of 4 pulse_count agents) and skip the prompt hash
        path. Both code paths map to the same bank index domain.
        """
        if not self.prompts:
            return ""
        if last_w is not None:
            w = float(last_w)
        else:
            w, _, _, _ = self._h4_of_prompts(prev_prompts)
        # W is a 4-tuple sum of 4 values in [0, 4); normalize to [0, 1)
        norm = (w % 4.0) / 4.0
        idx = int(norm * len(self.prompts)) % len(self.prompts)
        return self.prompts[idx]

    def w_energy(self, prev_prompts=None):
        """W energy fraction of the 4-tuple gate (the CORE keystone's
        ~0.61 dominance metric)."""
        w, z, y, x = self._h4_of_prompts(prev_prompts)
        e = w * w + z * z + y * y + x * x
        return (w * w / e) if e > 0 else 0.0
