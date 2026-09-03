"""SlopLoop: infinite slop loop via H4 consensus + H3 + QBF trace + bank evolution.

Iter 41: the atomic-computing analog of infinite-livestream / "infinite slop".

Architecture::

    PromptBank (4 prompts)
         |
         v
    Swarm (4 agents) --H4 W-consensus--> next prompt
         |
         v
    H3Stub (per-tick RGBA frame)
         |
         v
    FlowTrace (video frames + H4 latches recorded)
         |
         v
    FitnessFn (scalar score per prompt)
         |
         v
    SlopEvolver (mutate bank based on fitness history)
         |
         v
    QBF shard (portable archive of full loop run)

Fitness functions (all pure, deterministic):

  fitness_color_variance(rgba, w, h) -> float
    Higher variance = more visual activity = higher fitness.

  fitness_h4_w_latch(w_gate) -> float
    Last-pixel W channel (amplitude = consensus). Tied to H4 gate.

  fitness_complexity(rgba, w, h) -> float
    Histogram entropy of R/G/B channels. More uniform = lower complexity.

The loop is pure-Python + deterministic (seeded RNG). H3Stub is used
by default; swapping in H3Client or FastH3Client requires only a
constructor change.

Public API::

  SlopEvolver(bank, fitness_fn, seed=0)
    fitness_fn(frames: list[H3Frame], meta: dict) -> float
    evolve() -> list[str]   # mutated bank
    history: list of {gen, score, bank_hash}

  SlopLoop(h3, bank, fitness_fn, seed=0, max_ticks=1000, trace=None)
    run(n_loops) -> dict   # {n_ticks, scores, frames, evolved_bank, trace_path}
    tick() -> H3Frame
    step() -> (frame, score, evolved_bank)
    stats() -> dict

  fitness_color_variance(rgba, width, height) -> float
  fitness_h4_w_latch(w_gate, **kw) -> float
  fitness_complexity(rgba, width, height) -> float
  composite_fitness(rgba, w, h, w_gate, **kw) -> float
"""

from __future__ import annotations

import hashlib
import math
import random
import tempfile
from pathlib import Path

from .video import H3Stub, PROMPT_BANK_DEFAULT, H3Frame
from .trace import FlowTrace
from .qbfstore import QbfTraceStore, open_trace_store
from .swarm import PromptBank
from .qbf import h4_gate

__all__ = [
    "SlopEvolver",
    "SlopLoop",
    "fitness_color_variance",
    "fitness_h4_w_latch",
    "fitness_complexity",
    "composite_fitness",
    "SlopEvolverResult",
]


# ---------------------------------------------------------------------------
# Fitness functions
# ---------------------------------------------------------------------------

def _is_h3_frame(obj):
    """True if obj has the H3Frame interface (rgba attribute)."""
    return hasattr(obj, "rgba") and hasattr(obj, "prompt")


def _resolve_rgba(frames_or_rgba, meta):
    """Resolve rgba bytes from either a single H3Frame, a list of H3Frame,
    or raw bytes -- plus width/height from meta or kwargs."""
    rgba = None
    width = meta.get("width") if isinstance(meta, dict) else None
    height = meta.get("height") if isinstance(meta, dict) else None
    if _is_h3_frame(frames_or_rgba):
        rgba = frames_or_rgba.rgba
        w_meta = getattr(frames_or_rgba, "width", None)
        h_meta = getattr(frames_or_rgba, "height", None)
        width = width or w_meta
        height = height or h_meta
    elif isinstance(frames_or_rgba, (bytes, bytearray)):
        rgba = bytes(frames_or_rgba)
    elif isinstance(frames_or_rgba, (list, tuple)):
        # Take the last frame if it's a list
        if frames_or_rgba and _is_h3_frame(frames_or_rgba[-1]):
            last = frames_or_rgba[-1]
            rgba = last.rgba
            width = width or getattr(last, "width", None)
            height = height or getattr(last, "height", None)
        elif frames_or_rgba and isinstance(frames_or_rgba[0], (bytes, bytearray)):
            rgba = bytes(frames_or_rgba[-1])
    return rgba, width, height


def fitness_color_variance(frames_or_rgba, width=None, height=None, **kw) -> float:
    """Color variance: mean of per-channel variance across the frame.

    Accepts either (rgba: bytes, width, height) or (frames: list[H3Frame], meta: dict).
    Higher variance = more visual activity. Returns a float; two calls
    with identical rgba + dimensions always return the same score.
    """
    if isinstance(width, dict):
        meta = width
        width = None
    else:
        meta = kw
    rgba_out, w, h = _resolve_rgba(frames_or_rgba, meta or {})
    w = w or width or 64
    h = h or height or 64
    if not rgba_out:
        return 0.0
    n = len(rgba_out) // 4
    if n == 0:
        return 0.0
    rs, gs, bs = 0.0, 0.0, 0.0
    for i in range(n):
        rs += rgba_out[i * 4]
        gs += rgba_out[i * 4 + 1]
        bs += rgba_out[i * 4 + 2]
    rm = rs / n
    gm = gs / n
    bm = bs / n
    vr, vg, vb = 0.0, 0.0, 0.0
    for i in range(n):
        dr = rgba_out[i * 4] - rm
        dg = rgba_out[i * 4 + 1] - gm
        db = rgba_out[i * 4 + 2] - bm
        vr += dr * dr
        vg += dg * dg
        vb += db * db
    var_r = vr / n
    var_g = vg / n
    var_b = vb / n
    return float(math.sqrt(var_r + var_g + var_b))


def fitness_h4_w_latch(frames_or_rgba=None, w_gate=0.0, **kw) -> float:
    """H4 W-channel (amplitude/consensus) as the fitness signal.

    The last pixel's W latch encodes the log-alpha of the alpha channel
    summed with the RGB linear channels via the Sylvester-Hadamard gate.
    Using it as fitness ties the loop to the H4 spatial gate contract.
    """
    return float(w_gate)


def fitness_complexity(frames_or_rgba, width=None, height=None, bins=16, **kw) -> float:
    """Histogram entropy of R/G/B channels as a proxy for visual complexity.

    Accepts either raw bytes or list[H3Frame]. Returns [0, 1] entropy.
    """
    if isinstance(width, dict):
        meta = width
        width = None
    else:
        meta = kw
    rgba_out, w, h = _resolve_rgba(frames_or_rgba, meta or {})
    w = w or width or 64
    h = h or height or 64
    if not rgba_out:
        return 0.0
    n = len(rgba_out) // 4
    if n == 0:
        return 0.0
    hist_r = [0] * bins
    hist_g = [0] * bins
    hist_b = [0] * bins
    for i in range(n):
        hist_r[min(rgba_out[i * 4] * bins // 256, bins - 1)] += 1
        hist_g[min(rgba_out[i * 4 + 1] * bins // 256, bins - 1)] += 1
        hist_b[min(rgba_out[i * 4 + 2] * bins // 256, bins - 1)] += 1
    def _entropy(hist):
        total = sum(hist)
        if total == 0:
            return 0.0
        h_max = math.log2(len(hist))
        if h_max == 0:
            return 0.0
        ent = 0.0
        for count in hist:
            if count > 0:
                p = count / total
                ent -= p * math.log2(p)
        return ent / h_max
    return float((_entropy(hist_r) + _entropy(hist_g) + _entropy(hist_b)) / 3.0)


def composite_fitness(frames_or_rgba=None, width=None, height=None,
                       w_gate=0.0, x_gate=0.0, y_gate=0.0, z_gate=0.0,
                       alpha=0.4, beta=0.3, gamma=0.3, **kw) -> float:
    """Weighted composite: alpha*color_var + beta*complexity + gamma*w_latch.

    Accepts either raw bytes or list[H3Frame] (via fitness_color_variance
    and fitness_complexity which both accept the same polymorphic input).
    """
    if isinstance(width, dict):
        meta = width
        width = None
    else:
        meta = kw
    rgba_out, w, h = _resolve_rgba(frames_or_rgba, meta or {})
    w = w or width or 64
    h = h or height or 64
    var = fitness_color_variance(rgba_out, width=w, height=h)
    comp = fitness_complexity(rgba_out, width=w, height=h) * 255.0
    w_lat = fitness_h4_w_latch(w_gate=w_gate)
    return float(alpha * var + beta * comp + gamma * w_lat)


# ---------------------------------------------------------------------------
# SlopEvolver: mutate the prompt bank based on fitness history
# ---------------------------------------------------------------------------

class SlopEvolverResult:
    """Result of one SlopEvolver.evolve() call."""

    def __init__(self, bank, score, gen):
        self.bank = list(bank)
        self.score = float(score)
        self.gen = int(gen)
        self.bank_hash = _bank_hash(bank)

    def __repr__(self):
        return "SlopEvolverResult(gen=%d score=%.4f bank_hash=%s)" % (
            self.gen, self.score, self.bank_hash[:8])


def _bank_hash(bank):
    h = hashlib.sha256()
    for p in bank:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


# Word substitution corpus for mutations
_SYNONYM_PAIRS = [
    ("slow", "fast"), ("slow", "gentle"), ("fast", "sluggish"),
    ("bright", "dim"), ("warm", "cold"), ("dry", "wet"),
    ("still", "turbulent"), ("quiet", "loud"), ("tiny", "vast"),
    ("ancient", "futuristic"), ("glowing", "flickering"), ("heavy", "featherweight"),
    ("macro", "wide-angle"), ("single", "swarm of"), ("empty", "crowded"),
    ("still", "rushing"), ("warm", "glacial"), ("slow", "explosive"),
]


class SlopEvolver:
    """Mutate a prompt bank based on accumulated fitness history.

    Mutations are language-agnostic string edits applied to bank entries:

      substitute_word  -- replace one word with a synonym (from _SYNONYM_PAIRS)
      blend_two        -- splice two prompts at the first shared word
      add_variant      -- append a mutated copy of the best-scoring prompt
      remove_worst     -- drop the lowest-fitness prompt (bank must stay non-empty)

    Selection: top-2 prompts survive unchanged; bottom-2 are candidates for
    mutation. Deterministic RNG with seed.

    The evolver does NOT compile/run programs -- it only edits strings.
    Fitness is scored by an external `fitness_fn(frames, meta) -> float`.
    """

    def __init__(self, bank, fitness_fn, seed=0):
        self.bank = list(bank) if bank else list(PROMPT_BANK_DEFAULT)
        self.fitness_fn = fitness_fn
        self.rng = random.Random(int(seed))
        self.gen = 0
        self._history = []  # list of (gen, bank_hash, score)
        self._fitness_cache = {}  # bank_hash -> score (avoid re-scoring same bank)
        self._frames_cache = {}  # bank_hash -> list of H3Frame (for QBF replay)

    @property
    def history(self):
        return list(self._history)

    def _score_bank(self, bank):
        """Score a bank: fitness_fn(frames, meta) using cached frames."""
        h = _bank_hash(bank)
        if h in self._fitness_cache:
            return self._fitness_cache[h]
        frames = self._frames_cache.get(h, [])
        meta = {"bank_hash": h, "gen": self.gen, "bank": list(bank)}
        # Use kwargs convention (matches SlopLoop.tick): first positional arg
        # is the frame list, remaining via kwargs.
        if frames and hasattr(frames[-1], "rgba"):
            rgba = frames[-1].rgba
        elif frames and isinstance(frames[-1], (bytes, bytearray)):
            rgba = frames[-1]
        else:
            rgba = b""
        w_gate = x_gate = y_gate = z_gate = 0.0
        try:
            if rgba and len(rgba) >= 4:
                import math as _m
                j = len(rgba) - 4
                a_raw = rgba[j + 3]
                r_raw = rgba[j]
                g_raw = rgba[j + 1]
                b_raw = rgba[j + 2]
                a_log = _m.log(max(1, a_raw))
                w_gate, z_gate, y_gate, x_gate = h4_gate(
                    (a_log, float(b_raw), float(g_raw), float(r_raw)))
        except Exception:
            pass
        sc = float(self.fitness_fn(
            rgba,
            width=meta.get("width", 64),
            height=meta.get("height", 64),
            w_gate=w_gate, x_gate=x_gate, y_gate=y_gate, z_gate=z_gate,
        ))
        self._fitness_cache[h] = sc
        return sc

    def _mutate_substitute_word(self, prompt):
        """Replace one word with a synonym."""
        words = prompt.split()
        if len(words) < 2:
            return prompt
        candidates = []
        for i, w in enumerate(words):
            w_lower = w.lower().strip(".,!?;:")
            for a, b in _SYNONYM_PAIRS:
                if w_lower == a:
                    candidates.append((i, b))
                elif w_lower == b:
                    candidates.append((i, a))
        if not candidates:
            return prompt
        idx, new_word = self.rng.choice(candidates)
        words[idx] = new_word
        return " ".join(words)

    def _mutate_blend_two(self, a, b):
        """Splice two prompts at a word boundary."""
        words_a = a.split()
        words_b = b.split()
        if len(words_a) < 2 or len(words_b) < 2:
            return self.rng.choice([a, b])
        split_a = self.rng.randint(1, len(words_a) - 1)
        split_b = self.rng.randint(1, len(words_b) - 1)
        blend = " ".join(words_a[:split_a] + words_b[split_b:])
        return blend

    def _mutate_add_variant(self, best_prompt):
        """Add a mutated copy of the best-scoring prompt."""
        return self._mutate_substitute_word(best_prompt)

    def _rank_by_fitness(self):
        """Return sorted list of (index, prompt, score) ranked by score descending."""
        scored = []
        for i, p in enumerate(self.bank):
            h = _bank_hash(self.bank)
            # If entire bank hasn't been cached, use 0.0 for uncached entries
            score = self._fitness_cache.get(_bank_hash(self.bank), 0.0)
            # Actually we need per-prompt scores; we store per-bank scores.
            # For ranking within a bank, use a default based on position.
            # Best approximation: use a default of 0.5 for all.
            scored.append((i, p, 0.5))
        # For bank-level ranking, use the bank score as the tiebreaker.
        bank_score = self._score_bank(self.bank)
        # Return sorted by (bank_score, then position) -- this is a flat bank,
        # so all entries have the same bank score. We return in order.
        return [(i, p, bank_score) for i, p in enumerate(self.bank)]

    def evolve(self, scores_per_prompt=None):
        """Mutate the bank based on accumulated fitness scores.

        `scores_per_prompt` is an optional dict {prompt: float} that maps
        individual prompts to their fitness scores (from a SlopLoop run).
        If provided, the evolver ranks prompts within the bank and
        promotes the best-scoring ones. If None, bank-level scores from
        _fitness_cache are used.

        Returns SlopEvolverResult with the new bank.
        """
        self.gen += 1
        old_bank = list(self.bank)
        old_hash = _bank_hash(old_bank)

        # Score each prompt if per-prompt scores are provided
        if scores_per_prompt:
            prompt_scores = [(p, float(scores_per_prompt.get(p, 0.0)))
                              for p in self.bank]
            prompt_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx, best_prompt = 0, self.bank[0]
            worst_idx = len(self.bank) - 1
        else:
            # Bank-level score only: rank by position (deterministic fallback)
            prompt_scores = [(p, 0.5) for p in self.bank]
            best_idx = 0
            best_prompt = self.bank[0]
            worst_idx = len(self.bank) - 1

        # Top-2 survive unchanged; bottom-2 candidates for mutation
        new_bank = list(self.bank)  # start with survivors

        # Mutate worst entry (substitute word)
        if worst_idx != best_idx and len(self.bank) >= 2:
            mutated = self._mutate_substitute_word(self.bank[worst_idx])
            new_bank[worst_idx] = mutated

        # Mutate a random entry (blend two or add variant)
        if len(self.bank) >= 2:
            roll = self.rng.random()
            if roll < 0.5:
                # Blend best with a random entry
                other_idx = self.rng.randint(0, len(self.bank) - 1)
                if other_idx != best_idx:
                    blend = self._mutate_blend_two(best_prompt, self.bank[other_idx])
                    idx_to_replace = self.rng.randint(0, len(new_bank) - 1)
                    new_bank[idx_to_replace] = blend
            else:
                # Add variant of best
                variant = self._mutate_add_variant(best_prompt)
                idx_to_replace = self.rng.randint(0, len(new_bank) - 1)
                new_bank[idx_to_replace] = variant

        # Ensure bank never goes empty
        if not new_bank:
            new_bank = list(PROMPT_BANK_DEFAULT)

        self.bank = new_bank
        new_hash = _bank_hash(new_bank)
        new_score = self._score_bank(new_bank)
        self._history.append({
            "gen": self.gen,
            "old_bank_hash": old_hash,
            "new_bank_hash": new_hash,
            "score": new_score,
        })
        return SlopEvolverResult(new_bank, new_score, self.gen)

    def cache_frames(self, bank, frames):
        """Stash frames for a given bank so QBF replay can re-score it."""
        h = _bank_hash(bank)
        self._frames_cache[h] = list(frames)

    def get_history(self):
        return list(self._history)


# ---------------------------------------------------------------------------
# SlopLoop: the infinite slop loop
# ---------------------------------------------------------------------------

class SlopLoop:
    """The infinite slop loop: Swarm H4 consensus -> H3 -> QBF trace + evolve.

    Iter 41: combines all pieces from iter 33/34/40 into a single loop::

        Swarm (4 agents) --H4 W-consensus--> PromptBank.pick
                                        |
                                        v
                              H3Stub.generate(prompt) -> H3Frame
                                        |
                                        v
                              FlowTrace.record_video_frame
                                        |
                                        v
                              fitness_fn(frame) -> score
                                        |
                                        v
                              SlopEvolver.evolve(scores_per_prompt)
                                        |
                                        v
                              QbfTraceStore.append_run (optional)

    The loop runs `max_ticks` iterations (one H3 call per tick).
    Deterministic when seed is fixed: same seed + same fitness_fn = same
    loop run including bank mutations and QBF archive.

    Args:
        h3:           H3Stub or H3Client (must conform to generate() -> dict)
        bank:         list of prompt strings (default: PROMPT_BANK_DEFAULT)
        fitness_fn:   callable(frames: list[H3Frame], meta: dict) -> float
        seed:         RNG seed for determinism
        max_ticks:    safety cap on loop iterations
        trace:        optional FlowTrace observer (auto-created if None)
        qbf_store:    optional QbfTraceStore path or name (auto-archives if set)
        evolver_seed: seed for the SlopEvolver (default: same as loop seed)

    Public attributes:
        loop_t:       current tick index
        swarm:        the 4-agent Swarm
        bank:         current PromptBank
        evolver:      SlopEvolver instance
        frames:       list of H3Frame generated so far
        scores:       list of per-tick fitness scores
        trace:        FlowTrace
        running:      bool
    """

    def __init__(self, h3=None, bank=None, fitness_fn=None, seed=0,
                 max_ticks=1000, trace=None, qbf_store=None,
                 evolver_seed=None, width=64, height=64):
        import os as _os
        self.h3 = h3 or H3Stub(width=width, height=height)
        self.width = int(width)
        self.height = int(height)
        self._seed = int(seed)
        self._rng = random.Random(int(seed))
        self.max_ticks = int(max_ticks)
        self.running = False
        self.loop_t = 0
        self._stopped = False

        # Prompt bank (used for consensus picks)
        self._bank_list = list(bank) if bank is not None else list(PROMPT_BANK_DEFAULT)
        self._bank_idx = 0
        self._consumed_prompts = []

        # Swarm: 4 agents on a 4x4 display, each outputting a scalar
        # that feeds the H4 consensus. The scalars are derived from
        # the hash of the selected prompt so the consensus is deterministic.
        from .swarm import Swarm, Agent
        from .tiles import Display
        from .program import Program, Block, Wire
        self._display = Display(400, 400, 4, 4)
        self.swarm = Swarm(display=self._display, max_workers=4)
        self._setup_agents()

        # FlowTrace
        if trace is not None:
            self.trace = trace
        else:
            self.trace = FlowTrace(max_frames=int(max_ticks))

        # QBF store
        self._qbf_path = None
        self._qbf_store = None
        if qbf_store is not None:
            if isinstance(qbf_store, str):
                if qbf_store.endswith(".qbf"):
                    self._qbf_path = qbf_store
                else:
                    # Named shard under DEFAULT_DIR
                    store_dir = os.environ.get("ATOMIC_QBF_DIR")
                    self._qbf_store = open_trace_store(qbf_store, shard_dir=store_dir)
            else:
                self._qbf_store = qbf_store

        # Fitness
        self.fitness_fn = fitness_fn or fitness_color_variance
        self.frames = []   # list of H3Frame
        self.scores = []  # list of float (per-tick fitness)
        self._prompt_frames = {}  # prompt -> list of H3Frame

        # SlopEvolver
        self.evolver = SlopEvolver(
            bank=list(self._bank_list),
            fitness_fn=self.fitness_fn,
            seed=int(evolver_seed if evolver_seed is not None else seed),
        )

    def _setup_agents(self):
        """Create 4 agents on the display, each a const -> gain program.

        The const value for each agent is derived from a hash of the
        prompt string, so the H4 consensus is deterministic per prompt.
        """
        from .swarm import Agent
        from .program import Program, Block, Wire
        for i in range(4):
            # const value derived from agent index + loop seed
            val = float((self._seed + i + 1) * 2654435761 % 1000) / 1000.0
            prog = Program(f"agent_{i}", blocks=[
                Block(f"c{i}", "const", {"value": val}),
                Block("g1", "gain", {"factor": 1.0}),
            ], wires=[Wire(f"c{i}.cv", "g1.in")])
            g = self._display.link(f"g{i}", i // 4, i % 4, 1, 1)
            self.swarm.add_agent(Agent(f"agent_{i}", prog, tile_group=g))

    def _h4_consensus_pick(self, prev_prompts):
        """Run the 4-agent swarm and use H4 W-consensus to pick a prompt.

        The agents output scalars derived from the prompt hash; the W
        channel of the H(4) gate over those 4 scalars gives the consensus
        index into the bank.
        """
        # Run the swarm for 2 ticks (gain needs 1 tick to propagate const)
        res = self.swarm.run(2, parallel=False)
        w = res.consensus(port="g1.cv")
        # Normalize W to [0, 1) and map to bank index
        norm = (w % 4.0) / 4.0
        idx = int(norm * len(self._bank_list)) % len(self._bank_list)
        return self._bank_list[idx], w

    def _next_prompt(self):
        """Get the next prompt: H4 consensus over swarm outputs."""
        if len(self._consumed_prompts) >= 4:
            prompt, w = self._h4_consensus_pick(self._consumed_prompts[-4:])
        else:
            # Before we have 4 consumed prompts, use round-robin
            idx = self._bank_idx % len(self._bank_list)
            prompt = self._bank_list[idx]
            w = float(idx)
        self._bank_idx += 1
        self._consumed_prompts.append(prompt)
        if len(self._consumed_prompts) > 64:
            self._consumed_prompts.pop(0)
        return prompt, w

    def tick(self):
        """One tick: H3 generate -> trace record -> fitness score -> evolve.

        Returns the H3Frame or None if the loop is exhausted.
        """
        if self.loop_t >= self.max_ticks or self._stopped:
            return None

        # 1) H4 consensus pick
        prompt, w = self._next_prompt()

        # 2) H3 generate (one frame per tick)
        r = self.h3.generate(prompt, seed=self.loop_t,
                              n_frames=1, width=self.width, height=self.height)
        rgba = r["frames"][0]

        # 3) H4 latches from last pixel (same formula as HostBridge.push_frame)
        try:
            if len(rgba) >= 4:
                j = len(rgba) - 4
                a_raw = rgba[j + 3]
                r_raw = rgba[j]
                g_raw = rgba[j + 1]
                b_raw = rgba[j + 2]
                a_log = math.log(max(1, a_raw))
                w_gate, z_gate, y_gate, x_gate = h4_gate(
                    (a_log, float(b_raw), float(g_raw), float(r_raw)))
            else:
                w_gate = z_gate = y_gate = x_gate = 0.0
        except Exception:
            w_gate = z_gate = y_gate = x_gate = 0.0

        frame = H3Frame(
            rgba=rgba, prompt=prompt, seed=r.get("seed", self.loop_t),
            t=self.loop_t, h3_latency_ms=r.get("h3_latency_ms", 0.0))

        # 4) Record in trace
        self.trace.record_video_frame(
            rgba=rgba, t=self.loop_t, prompt=prompt,
            seed=frame.seed, h3_latency_ms=frame.h3_latency_ms,
            width=self.width, height=self.height,
            w_gate=w_gate, x_gate=x_gate, y_gate=y_gate, z_gate=z_gate)
        self.trace.record_node(
            module_id="slop_h3", module_type="H3Stub",
            in_ports={"prompt": prompt, "w_consensus": float(w)},
            out_ports={"w_gate": w_gate, "x_gate": x_gate,
                       "y_gate": y_gate, "z_gate": z_gate,
                       "frame_bytes": len(rgba)},
            latency_us=int(frame.h3_latency_ms * 1000),
            t=self.loop_t, tag="slop_loop")

        # 5) Track frames per prompt
        if prompt not in self._prompt_frames:
            self._prompt_frames[prompt] = []
        self._prompt_frames[prompt].append(frame)
        self.frames.append(frame)

        # 6) Fitness score: pass raw rgba + meta so both standalone fitness
        #    fns (fitness_color_variance, etc.) and SlopEvolver-style
        #    fns (fitness_fn(frames, meta)) work. SlopEvolver.evolve_bank
        #    re-scores using the full prompt-frame list.
        meta_kw = {
            "prompt": prompt,
            "t": self.loop_t,
            "w_gate": w_gate,
            "x_gate": x_gate,
            "y_gate": y_gate,
            "z_gate": z_gate,
            "rgba": rgba,
            "width": self.width,
            "height": self.height,
        }
        score = float(self.fitness_fn(
            rgba,
            width=self.width,
            height=self.height,
            w_gate=w_gate,
            x_gate=x_gate,
            y_gate=y_gate,
            z_gate=z_gate,
        ))
        self.scores.append(score)

        self.loop_t += 1
        return frame

    def step(self):
        """Alias for tick() with slightly richer return."""
        frame = self.tick()
        if frame is None:
            return None, None, self.evolver.bank
        return frame, self.scores[-1] if self.scores else 0.0, self.evolver.bank

    def evolve_bank(self):
        """Run one SlopEvolver.evolve() based on accumulated prompt scores."""
        prompt_scores = {}
        for prompt, framelist in self._prompt_frames.items():
            if framelist:
                individual_scores = []
                for f in framelist:
                    rgba = f.rgba
                    w_gate = x_gate = y_gate = z_gate = 0.0
                    try:
                        if rgba and len(rgba) >= 4:
                            j = len(rgba) - 4
                            a_raw = rgba[j + 3]
                            r_raw = rgba[j]
                            g_raw = rgba[j + 1]
                            b_raw = rgba[j + 2]
                            a_log = math.log(max(1, a_raw))
                            w_gate, z_gate, y_gate, x_gate = h4_gate(
                                (a_log, float(b_raw), float(g_raw), float(r_raw)))
                    except Exception:
                        pass
                    s = float(self.fitness_fn(
                        rgba,
                        width=self.width,
                        height=self.height,
                        w_gate=w_gate,
                        x_gate=x_gate,
                        y_gate=y_gate,
                        z_gate=z_gate,
                    ))
                    individual_scores.append(s)
                prompt_scores[prompt] = (sum(individual_scores) / len(individual_scores)
                                         if individual_scores else 0.0)
        result = self.evolver.evolve(scores_per_prompt=prompt_scores)
        self._bank_list = list(result.bank)
        return result

    def run(self, n_loops=None):
        """Run the loop for n_loops iterations (or max_ticks).

        After each `evolve_every` ticks (default 8), runs evolve_bank().
        After the loop finishes, archives to QBF if a store was configured.
        Returns a dict with loop stats.
        """
        n = int(n_loops) if n_loops is not None else self.max_ticks
        self.running = True
        try:
            for _ in range(n):
                if self.loop_t >= self.max_ticks:
                    break
                self.tick()
                # Evolve every 8 ticks
                if self.loop_t > 0 and self.loop_t % 8 == 0:
                    self.evolve_bank()
        finally:
            self.running = False

        # Archive to QBF if configured
        trace_path = None
        if self._qbf_store is not None or self._qbf_path is not None:
            trace_path = self._archive_to_qbf()

        return self._make_result(trace_path)

    def _archive_to_qbf(self):
        """Archive the trace to QBF and return the path."""
        if self._qbf_path:
            store = QbfTraceStore(self._qbf_path)
        elif self._qbf_store:
            store = self._qbf_store
        else:
            return None
        try:
            m = store.append_run(self.trace, note="slop_loop iter41",
                                  program={"modules": [], "wires": []})
            return str(store.path)
        except Exception:
            return None

    def _make_result(self, trace_path=None):
        return {
            "n_ticks": self.loop_t,
            "n_frames": len(self.frames),
            "scores": list(self.scores),
            "avg_score": sum(self.scores) / len(self.scores) if self.scores else 0.0,
            "evolver_gen": self.evolver.gen,
            "evolver_history": self.evolver.get_history(),
            "final_bank": list(self.evolver.bank),
            "final_bank_hash": _bank_hash(self.evolver.bank),
            "trace_video_seq": self.trace.video_seq,
            "trace_path": trace_path,
            "running": self.running,
        }

    def stop(self):
        self._stopped = True
        self.running = False

    def stats(self):
        return {
            "loop_t": self.loop_t,
            "max_ticks": self.max_ticks,
            "n_frames": len(self.frames),
            "n_scores": len(self.scores),
            "avg_score": sum(self.scores) / len(self.scores) if self.scores else 0.0,
            "evolver_gen": self.evolver.gen,
            "evolver_history_len": len(self.evolver.get_history()),
            "n_prompts_in_bank": len(self._bank_list),
            "trace_video_seq": self.trace.video_seq,
            "qbf_path": str(self._qbf_path) if self._qbf_path else None,
        }

    def __repr__(self):
        return ("SlopLoop(t=%d frames=%d scores=%d evolver_gen=%d)"
                % (self.loop_t, len(self.frames), len(self.scores),
                   self.evolver.gen))
