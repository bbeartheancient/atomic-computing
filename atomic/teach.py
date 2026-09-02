"""Teach: teacher examples + domain routing (goals 4/5).

Goal 4: "the model learns function paths from teacher examples" --
programs are stored as {description, program, domain} pairs that
teach the mapping from natural language to block graphs.

Goal 5: "swappable domain vocab/n-gram sets routed by signal path" --
a domain selects which keyword weights apply when matching a
description to a program (e.g. medical vs audio vs signal).

This module is the harness twin of fabric/atomic_program.py's
teacher registry, but adapted to the local harness:

  - register_example(description, program, domain, difficulty)
  - match(description, domain) -> best Program or None
  - from_description(description, domain) -> Program (matched or synthesized)
  - domain_vocab(domain) -> keyword set
  - list_domains(), list_examples(domain)

Matching is keyword-gated: tokenize description, score each example
by overlap weighted by domain vocab. No LLM required for the
harness path; the sibling's LLM path remains available via
`from_description_llm()` if a vLLM endpoint is up.

The registry is in-memory (plus optional QBF persistence via
qbfstore if desired). For goal 5's swappable vocab, each domain
carries its own keyword->weight map; routing picks the domain
whose vocab best covers the query.
"""

from __future__ import annotations

import re
import hashlib
import json

from .program import Block, Program, Wire
from .gates import ATOMS

__all__ = ["TeacherRegistry", "REGISTRY", "register_example",
           "match", "from_description", "domain_vocab",
           "DOMAINS", "KEYWORDS_BY_DOMAIN"]

# domain -> {keyword: weight}
DOMAINS = ["general", "audio", "medical", "signal", "control", "spatial"]

KEYWORDS_BY_DOMAIN = {
    "general": {"counter": 2, "chart": 2, "clock": 2, "gain": 1, "smooth": 1, "filter": 1, "accum": 1},
    "audio": {"audio": 3, "hadamard": 3, "h4": 3, "spatial": 2, "wxyz": 2, "mdct": 2, "flux": 2, "sensor": 1, "spectrum": 2, "frequency": 2, "beat": 1, "rhythm": 2, "mix": 2, "pan": 1},
    "medical": {"first": 1, "aid": 2, "medical": 3, "threshold": 2, "sensor": 1, "monitor": 2, "ecg": 2, "pulse": 2, "alarm": 2, "vital": 2, "heart": 2},
    "signal": {"signal": 3, "filter": 2, "smooth": 2, "average": 2, "moving": 2, "threshold": 2, "gain": 1, "hysteresis": 2, "clamp": 2, "bias": 1, "accum": 1},
    "control": {"clock": 2, "bpm": 2, "toggle": 2, "accum": 2, "trigger": 2, "gate": 1, "counter": 2, "divider": 2, "sequencer": 1, "tap": 1},
    "spatial": {"hadamard": 3, "h4": 3, "spatial": 3, "wxyz": 3, "3d": 2, "scope": 1, "consensus": 2, "rotation": 2, "sylvester": 2, "axis": 1, "quadrature": 2, "amplitude": 1, "dominant": 2},
}

# primitive keyword -> program fragment hint
PRIMITIVE_KEYWORDS = {
    "clock": ("clock_bpm", {"bpm": 60}),
    "bpm": ("clock_bpm", {"bpm": 60}),
    "counter": ("accum", {"per_tick": 1}),
    "accum": ("accum", {"per_tick": 1}),
    "smooth": ("smooth", {"alpha": 0.2}),
    "average": ("moving_avg", {"n": 8}),
    "moving average": ("moving_avg", {"n": 8}),
    "threshold": ("threshold", {"hi": 0.5, "lo": -0.5}),
    "hysteresis": ("threshold", {"hi": 0.5, "lo": -0.5}),
    "gain": ("gain", {"factor": 2.0}),
    "scale": ("gain", {"factor": 2.0}),
    "bias": ("bias", {"add": 1.0}),
    "clamp": ("clamp", {"lo": -1.0, "hi": 1.0}),
    "hadamard": ("h4_slide", {}),
    "h4": ("h4_slide", {}),
    "wxyz": ("h4_slide", {}),
    "sine": ("sine_lfo", {"rate_hz": 1.0, "amp": 1.0}),
    "lfo": ("sine_lfo", {"rate_hz": 1.0}),
    "oscillator": ("sine_lfo", {"rate_hz": 1.0}),
    "spectrum": ("mdct_flux", {}),
    "mdct": ("mdct_flux", {}),
    "flux": ("mdct_flux", {}),
    "toggle": ("toggle", {"initial": 0}),
    "divider": ("toggle", {"initial": 0}),
    "sequencer": ("clock_bpm", {"bpm": 120}),
    "tap": ("tap", {}),
    "gate and": ("gate_and", {}),
    "gate or": ("gate_or", {}),
    "xor": ("gate_xor", {}),
    "chart": ("viz_series", {}),
    "scope": ("viz_xy", {}),
    "3d": ("viz_wxyz3d", {}),
    "consensus": ("viz_series", {}),
    "dominant": ("viz_series", {}),
    "amplitude": ("viz_series", {}),
}


def _tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class TeacherRegistry:
    def __init__(self):
        self.examples = []  # list of {hash, description, domain, difficulty, program, keywords}

    def register(self, description, program, domain="general", difficulty="beginner"):
        h = hashlib.sha256((description + program.name).encode()).hexdigest()[:12]
        entry = {
            "hash": h,
            "description": description,
            "domain": domain,
            "difficulty": difficulty,
            "program": program,
            "keywords": set(_tokenize(description)),
        }
        # upsert by hash
        for i, e in enumerate(self.examples):
            if e["hash"] == h:
                self.examples[i] = entry
                return h
        self.examples.append(entry)
        return h

    def list(self, domain="", limit=20):
        out = self.examples
        if domain:
            out = [e for e in out if e["domain"] == domain]
        return out[:limit]

    def domains(self):
        return sorted(set(e["domain"] for e in self.examples) | set(DOMAINS))

    def match(self, description, domain=""):
        """Best example by keyword overlap (domain-weighted)."""
        if not self.examples:
            return None
        toks = set(_tokenize(description))
        # domain routing: if domain given, only consider that domain + general
        candidates = self.examples
        if domain:
            candidates = [e for e in candidates if e["domain"] in (domain, "general")]
            if not candidates:
                candidates = self.examples
        best = None
        best_score = -1
        vocab = KEYWORDS_BY_DOMAIN.get(domain, {}) if domain else {}
        for e in candidates:
            overlap = len(toks & e["keywords"])
            # domain vocab bonus
            bonus = sum(vocab.get(t, 0) for t in toks & e["keywords"])
            score = overlap * 2 + bonus
            if score > best_score:
                best_score = score
                best = e
        if best_score <= 0:
            return None
        return best

    def synthesize(self, description, domain="general"):
        """Keyword-gated synthesis: tokens -> blocks + wires."""
        toks = _tokenize(description)
        joined = " ".join(toks)
        # collect requested primitives
        wants = []
        for kw, (prim, params) in PRIMITIVE_KEYWORDS.items():
            if kw in joined:
                wants.append((prim, dict(params)))
        # deduplicate by primitive
        seen = set()
        uniq = []
        for prim, params in wants:
            if prim not in seen:
                seen.add(prim)
                uniq.append((prim, params))
        if not uniq:
            # default: const -> gain -> viz
            uniq = [("const", {"value": 1.0}), ("gain", {"factor": 1.0})]
        # ensure sinks last
        non_sinks = [(p, pr) for p, pr in uniq if not p.startswith("viz_")]
        sinks = [(p, pr) for p, pr in uniq if p.startswith("viz_")]
        if not sinks:
            sinks = [("viz_series", {})]
        blocks = []
        wires = []
        prev_port = None
        for i, (prim, params) in enumerate(non_sinks + sinks):
            bid = "b%d" % i
            # special: const with value 1
            blocks.append(Block(bid, prim, dict(params)))
            if i == 0:
                prev_port = None
            else:
                # wire previous block's main output to this block's main input
                prev_prim = non_sinks[i-1][0] if i-1 < len(non_sinks) else sinks[i - len(non_sinks) - 1][0] if sinks else ""
                # resolve ports
                prev_atom = ATOMS.get(non_sinks[i-1][0] if i-1 < len(non_sinks) else (non_sinks[-1][0] if non_sinks else "")) if i > 0 else None
                cur_atom = ATOMS.get(prim)
                # pick a valid output port from prev, input port for cur
                if prev_atom and prev_atom.outputs:
                    src = blocks[i-1].id + "." + prev_atom.outputs[0]
                elif i > 0:
                    src = blocks[i-1].id + ".cv"
                else:
                    src = None
                if cur_atom and cur_atom.inputs:
                    # respect node rule: only wire if cur has inputs
                    if src:
                        dst = bid + "." + cur_atom.inputs[0]
                        wires.append(Wire(src, dst))
                elif src:
                    # inputless sink? viz needs in
                    if prim.startswith("viz_"):
                        # viz_series expects "in"
                        atom_in = "in"
                        wires.append(Wire(src, bid + "." + atom_in))
        # handle h4_slide domain-specific wiring: if h4_slide present, fan its w to sink
        # already wired via chain above
        prog = Program("synthesized", blocks=blocks, wires=wires,
                       description=description)
        # validate, fallback if needed
        errs = prog.validate()
        if errs:
            # minimal valid
            fb = [Block("c0", "const", {"value": 1.0}), Block("v0", "viz_series")]
            fw = [Wire("c0.cv", "v0.in")]
            prog = Program("synthesized", blocks=fb, wires=fw, description=description)
        return prog

    def from_description(self, description, domain="general"):
        """Match or synthesize."""
        hit = self.match(description, domain)
        if hit is not None:
            return hit["program"]
        return self.synthesize(description, domain)

    def clear(self):
        self.examples.clear()

    # -- QBF persistence (goal 6: swappable domain sets) -------------------

    def save_qbf(self, store_name=None, shard_dir=None, path=None):
        """Persist registry examples to a QBF shard or file.

        If store_name given, writes JSON blobs 'teach/index' and
        'teach/example/<hash>' (program as microfx patch) into a named
        shard under shard_dir (or DEFAULT_DIR). If path given, writes a
        single .qbf file or a plain JSON file (suffix .json). Returns
        the written path.
        """
        import json as _json, os as _os, tempfile as _tf
        from pathlib import Path as _P
        # serialize: each example -> {description, domain, difficulty, patch}
        payload = []
        for e in self.examples:
            patch = e["program"].compile("microfx")
            payload.append({"hash": e["hash"], "description": e["description"],
                            "domain": e["domain"], "difficulty": e["difficulty"],
                            "patch": patch, "name": e["program"].name})
        if store_name:
            from .qbfstore import open_trace_store as _ots
            store = _ots(store_name, shard_dir=shard_dir)
            # clear previous teach blobs by rewriting? just overwrite keys
            store.file.put_json("teach/index", [{"hash": p["hash"], "domain": p["domain"]} for p in payload])
            for p in payload:
                store.file.put_json("teach/example/%s" % p["hash"], p)
            store.file.write()
            return str(store.path)
        if path:
            pp = _P(path)
            if pp.suffix == ".qbf":
                from .qbf import QbfFile as _Q
                f = _Q.create(str(pp))
                f.put_json("teach/index", [{"hash": p["hash"], "domain": p["domain"]} for p in payload])
                for p in payload:
                    f.put_json("teach/example/%s" % p["hash"], p)
                f.write()
                return str(pp)
            pp.write_text(_json.dumps(payload, indent=2))
            return str(pp)
        # default: temp qbf
        from .qbf import QbfFile as _Q
        td = _tf.mkdtemp(prefix="teach_")
        p = _os.path.join(td, "teach.qbf")
        f = _Q.create(p)
        f.put_json("teach/index", [{"hash": pp["hash"], "domain": pp["domain"]} for pp in payload])
        for pp in payload:
            f.put_json("teach/example/%s" % pp["hash"], pp)
        f.write()
        return p

    @classmethod
    def load_qbf(cls, path_or_store, shard_dir=None):
        """Load a registry persisted by save_qbf; returns a TeacherRegistry."""
        import json as _json
        from pathlib import Path as _P
        from .qbf import QbfFile as _Q
        from .program import Program as _Program
        reg = cls()
        p = _P(str(path_or_store))
        # try file path first
        if p.exists() and p.suffix == ".qbf":
            f = _Q.open(str(p))
            idx = f.get_json("teach/index")
            for entry in idx:
                h = entry["hash"]
                data = f.get_json("teach/example/%s" % h)
                prog = _Program.from_patch(data["patch"], name=data.get("name", "restored"))
                prog.description = data.get("description", "")
                reg.register(data["description"], prog, domain=data.get("domain", "general"),
                             difficulty=data.get("difficulty", "beginner"))
            return reg
        if p.exists() and p.suffix == ".json":
            payload = _json.loads(p.read_text())
            for data in payload:
                prog = _Program.from_patch(data["patch"], name=data.get("name", "restored"))
                prog.description = data.get("description", "")
                reg.register(data["description"], prog, domain=data.get("domain", "general"),
                             difficulty=data.get("difficulty", "beginner"))
            return reg
        # try shard name
        if shard_dir is not None or not p.exists():
            from .qbfstore import open_trace_store as _ots
            try:
                store = _ots(str(path_or_store), shard_dir=shard_dir)
                f = store.file
                if "teach/index" in f:
                    idx = f.get_json("teach/index")
                    for entry in idx:
                        h = entry["hash"]
                        data = f.get_json("teach/example/%s" % h)
                        prog = _Program.from_patch(data["patch"], name=data.get("name", "restored"))
                        prog.description = data.get("description", "")
                        reg.register(data["description"], prog, domain=data.get("domain", "general"),
                                     difficulty=data.get("difficulty", "beginner"))
                    return reg
            except Exception:
                pass
        if p.exists():
            payload = _json.loads(p.read_text())
            for data in payload:
                prog = _Program.from_patch(data["patch"], name=data.get("name", "restored"))
                prog.description = data.get("description", "")
                reg.register(data["description"], prog, domain=data.get("domain", "general"),
                             difficulty=data.get("difficulty", "beginner"))
            return reg
        raise FileNotFoundError(str(path_or_store))


REGISTRY = TeacherRegistry()


def _seed():
    if REGISTRY.examples:
        return
    # 14 canonical examples covering all 6 domains + CORE concepts
    p1 = Program("gated_clock_counter", description="A gated clock fires a counter; the count is smoothed before driving a chart.",
                 blocks=[Block("c1", "clock_bpm", {"bpm": 60}), Block("a1", "accum", {"per_tick": 1}), Block("s1", "smooth", {"alpha": 0.1}), Block("v1", "viz_series")],
                 wires=[Wire("c1.trig", "a1.in"), Wire("a1.acc", "s1.in"), Wire("s1.cv", "v1.in")])
    p2 = Program("hadamard_wxyz_scope", description="An audio stream feeds a H4 transform; the four W/X/Y/Z outputs drive a 3D scope.",
                 blocks=[Block("sig", "sensor"), Block("h4", "h4_slide"), Block("v1", "viz_wxyz3d")],
                 wires=[Wire("sig.cv", "h4.in"), Wire("h4.w", "v1.w"), Wire("h4.x", "v1.x")])
    p3 = Program("threshold_gate_mixer", description="Two signals enter a hysteresis threshold; the output gates a third signal.",
                 blocks=[Block("s1", "sensor"), Block("th1", "threshold", {"hi": 0.5, "lo": -0.5}), Block("g1", "gain", {"factor": 1.0}), Block("v1", "viz_series")],
                 wires=[Wire("s1.cv", "th1.in"), Wire("th1.gate", "g1.in"), Wire("g1.cv", "v1.in")])
    p4 = Program("medical_threshold", description="A medical sensor threshold gates an alarm chart.",
                 blocks=[Block("sig", "sensor"), Block("th", "threshold", {"hi": 0.8, "lo": 0.2}), Block("v", "viz_series")],
                 wires=[Wire("sig.cv", "th.in"), Wire("th.gate", "v.in")])
    # --- spatial domain (CORE H4 keystone) ---
    p5 = Program("spatial_h4_consensus", description="A sensor drives H4 spatial gate; dominant W consensus axis drives a series chart.",
                 blocks=[Block("sig", "sensor"), Block("h4", "h4_slide"), Block("v", "viz_series")],
                 wires=[Wire("sig.cv", "h4.in"), Wire("h4.w", "v.in")])
    p6 = Program("spatial_wxyz_rotation", description="H4 W/X/Y/Z outputs feed a 3D scope showing spatial rotation and dominant amplitude.",
                 blocks=[Block("sig", "sensor"), Block("h4", "h4_slide"), Block("v", "viz_wxyz3d")],
                 wires=[Wire("sig.cv", "h4.in"), Wire("h4.w", "v.w"), Wire("h4.x", "v.x"), Wire("h4.y", "v.y"), Wire("h4.z", "v.z")])
    # --- medical domain (vital sign / alarm concepts) ---
    p7 = Program("medical_vital_monitor", description="A medical sensor feeds a smooth filter then a threshold alarm chart.",
                 blocks=[Block("sig", "sensor"), Block("sm", "smooth", {"alpha": 0.1}), Block("th", "threshold", {"hi": 0.7, "lo": 0.3}), Block("v", "viz_series")],
                 wires=[Wire("sig.cv", "sm.in"), Wire("sm.cv", "th.in"), Wire("th.gate", "v.in")])
    p8 = Program("medical_ecg_pulse_counter", description="ECG pulse sensor triggers threshold then accum counter on a chart.",
                 blocks=[Block("sig", "sensor"), Block("th", "threshold", {"hi": 0.6, "lo": 0.2}), Block("acc", "accum", {"per_tick": 1}), Block("v", "viz_series")],
                 wires=[Wire("sig.cv", "th.in"), Wire("th.gate", "acc.in"), Wire("acc.acc", "v.in")])
    # --- audio domain (hadamard / mdct / spatial mix) ---
    p9 = Program("audio_spectrum_flux", description="Audio sensor feeds MDCT flux detector driving a series chart.",
                 blocks=[Block("sig", "sensor"), Block("fl", "mdct_flux", {}), Block("v", "viz_series")],
                 wires=[Wire("sig.cv", "fl.in"), Wire("fl.flux", "v.in")])
    p10 = Program("audio_spatial_mix", description="Two audio sensors mix through H4 spatial gate; W dominant consensus drives the chart.",
                  blocks=[Block("s1", "sensor"), Block("s2", "sensor"), Block("h4", "h4_slide"), Block("v", "viz_series")],
                  wires=[Wire("s1.cv", "h4.in"), Wire("s2.cv", "h4.in"), Wire("h4.w", "v.in")])
    # --- signal domain (filter / smooth / hysteresis) ---
    p11 = Program("signal_moving_average_filter", description="Signal sensor feeds a moving average filter; threshold gates a chart.",
                  blocks=[Block("sig", "sensor"), Block("ma", "moving_avg", {"n": 8}), Block("th", "threshold", {"hi": 0.5, "lo": -0.5}), Block("v", "viz_series")],
                  wires=[Wire("sig.cv", "ma.in"), Wire("ma.cv", "th.in"), Wire("th.gate", "v.in")])
    p12 = Program("signal_hysteresis_filter", description="Signal passes through a gain then a hysteresis threshold; clamped output goes to chart.",
                  blocks=[Block("sig", "sensor"), Block("g", "gain", {"factor": 2.0}), Block("cl", "clamp", {"lo": -1.0, "hi": 1.0}), Block("v", "viz_series")],
                  wires=[Wire("sig.cv", "g.in"), Wire("g.cv", "cl.in"), Wire("cl.cv", "v.in")])
    # --- control domain (clock / toggle / divider / sequencer) ---
    p13 = Program("control_clock_divider", description="A clock divider fires a toggle on every other beat; the toggle state drives a chart.",
                  blocks=[Block("clk", "clock_bpm", {"bpm": 120}), Block("tg", "toggle", {"initial": 0}), Block("v", "viz_series")],
                  wires=[Wire("clk.trig", "tg.trig"), Wire("tg.state", "v.in")])
    p14 = Program("control_bpm_accumulator", description="A fast BPM clock accumulates pulses; the count feeds a smoothed chart.",
                  blocks=[Block("clk", "clock_bpm", {"bpm": 240}), Block("acc", "accum", {"per_tick": 1}), Block("sm", "smooth", {"alpha": 0.05}), Block("v", "viz_series")],
                  wires=[Wire("clk.trig", "acc.in"), Wire("acc.acc", "sm.in"), Wire("sm.cv", "v.in")])
    # --- general domain (cross-domain concepts) ---
    p15 = Program("general_counter_chart", description="A counter accumulates and drives a chart.",
                  blocks=[Block("acc", "accum", {"per_tick": 1}), Block("v", "viz_series")],
                  wires=[Wire("acc.acc", "v.in")])
    p16 = Program("general_gain_scope", description="A gain block scales a constant value and drives a scope.",
                  blocks=[Block("c0", "const", {"value": 2.0}), Block("g", "gain", {"factor": 3.0}), Block("v", "viz_series")],
                  wires=[Wire("c0.cv", "g.in"), Wire("g.cv", "v.in")])
    REGISTRY.register(p1.description, p1, domain="control")
    REGISTRY.register(p2.description, p2, domain="spatial")
    REGISTRY.register(p3.description, p3, domain="signal")
    REGISTRY.register(p4.description, p4, domain="medical")
    REGISTRY.register(p5.description, p5, domain="spatial")
    REGISTRY.register(p6.description, p6, domain="spatial")
    REGISTRY.register(p7.description, p7, domain="medical")
    REGISTRY.register(p8.description, p8, domain="medical")
    REGISTRY.register(p9.description, p9, domain="audio")
    REGISTRY.register(p10.description, p10, domain="audio")
    REGISTRY.register(p11.description, p11, domain="signal")
    REGISTRY.register(p12.description, p12, domain="signal")
    REGISTRY.register(p13.description, p13, domain="control")
    REGISTRY.register(p14.description, p14, domain="control")
    REGISTRY.register(p15.description, p15, domain="general")
    REGISTRY.register(p16.description, p16, domain="general")


_seed()


def register_example(description, program, domain="general", difficulty="beginner"):
    return REGISTRY.register(description, program, domain, difficulty)


def match(description, domain=""):
    hit = REGISTRY.match(description, domain)
    return hit["program"] if hit else None


def from_description(description, domain="general"):
    return REGISTRY.from_description(description, domain)


def domain_vocab(domain):
    return dict(KEYWORDS_BY_DOMAIN.get(domain, {}))


def save_registry(store_name=None, shard_dir=None, path=None, registry=None):
    """Persist a TeacherRegistry to QBF (module-level helper)."""
    reg = registry or REGISTRY
    return reg.save_qbf(store_name=store_name, shard_dir=shard_dir, path=path)


def load_registry(path_or_store, shard_dir=None):
    """Load a TeacherRegistry from QBF (module-level helper)."""
    return TeacherRegistry.load_qbf(path_or_store, shard_dir=shard_dir)
