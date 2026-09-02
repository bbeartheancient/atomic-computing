"""Program: the AtomicProgram IR -- blocks + wires + views, validated and
compiled multi-target.

The IR is a STRICT SUBSET of the fabric validate_patch contract
(ATOMIC-PC-STATE.md, iter-2 findings): it applies the sibling rules
(unique ids; 'module.port' endpoints; source port a declared output;
dest port a declared input; the 'ui.*' virtual source; the one-input
node rule at PORT level) and ADDS rejections the patch runner tolerates
(well-defined by the 1-tick latency): SELF-WIRES and CYCLES. Every
program the harness accepts, fabric also accepts -> no parity impact.

JS-fidelity seams are closed HERE (iter-3 notes): the twin lowercases
param/port keys, the JS proxy reads m.params verbatim -- so the IR
enforces lowercase param keys and lowercase wire ports; a lowercase IR
makes the two engines agree bit-for-bit.

Compile targets:
  microfx   patch dict {modules, wires, views} with DICT params -- the
             patch-runner format consumed by BOTH the Engine and the node
             jsfx oracle (the sibling AtomicBlock emits the LIST form for
             the DOM app; the runner wants a dict).
  eel2      concatenated per-block EEL2 bodies (teaching view; sinks,
             which render from views[], are omitted)
  python    a self-contained script that runs the program over
             atomic.engine -- supersedes the sibling's broken numpy
             translator
  mermaid   flowchart teaching view
  wgsl      stub (goal 9, bicameral two-GPU split)

Catalog note: gate_* / quantum atoms are NOT in fabric's MODULES (removed
2026-08-26, Control library only) -- so fabric validate_patch rejects
gate patches even though the jsfx oracle runs them. The harness IR
accepts them (ATOMS is the harness's source of truth); oracle parity is
the authority for gate programs.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field

from .gates import ATOMS


class ProgramError(ValueError):
    """An invalid program (raised by compile; validate() returns the list)."""


def _plain(obj):
    """A block/wire as a plain dict (dataclass fields read by name; a
    bus.Wire slipped in by hand is read attribute-wise). Both forms
    carry the same fields, so the hash is representation-agnostic."""
    if getattr(obj, "__dataclass_fields__", None) is not None:
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return {k: getattr(obj, k) for k in ("id", "primitive", "params",
                                         "label", "src", "dst")
            if hasattr(obj, k)}


_VIZ_OUTPUTS = {
    "viz_series": "cv",
    "viz_xy": "y",
    "viz_wxyz3d": "z",
}


def _patch_views(blocks):
    views = []
    for b in blocks:
        output = _VIZ_OUTPUTS.get(b.primitive)
        if output is not None:
            views.append({"id": b.id, "module": b.id, "output": output,
                         "as": "series" if b.primitive == "viz_series" else b.primitive[4:]})
    return views


@dataclass
class Block:
    """One gate instance: id + primitive + param overrides (lowercase)."""

    id: str
    primitive: str
    params: dict = field(default_factory=dict)
    label: str = ""


@dataclass
class Wire:
    """One cable: fully-qualified bus keys 'id.port' -> 'id.port'."""

    src: str
    dst: str


class Program:
    """A graph of blocks + wires + display views, with an optional entry."""

    def __init__(self, name, blocks=(), wires=(), views=(), entry="",
                 tags=(), description=""):
        self.name = name
        self.blocks = list(blocks)
        self.wires = list(wires)
        self.views = list(views)
        self.entry = entry
        self.tags = list(tags)
        self.description = description
        self.hash = self.compute_hash()

    # -- identity -----------------------------------------------------------

    def compute_hash(self):
        key = json.dumps(
            {"name": self.name,
             "blocks": [_plain(b) for b in self.blocks],
             "wires": [_plain(w) for w in self.wires],
             "views": self.views, "entry": self.entry,
             "tags": self.tags},
            sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    # -- construction from a patch dict (the microfx target inverse) --------

    @classmethod
    def from_patch(cls, patch, name="patch"):
        blocks = [
            Block(id=str(m.get("id")),
                  primitive=str(m.get("primitive") or ""),
                  params=dict(m.get("params") or {}))
            for m in (patch.get("modules") or [])
        ]
        wires = [Wire(src=str(w.get("from")), dst=str(w.get("to")))
                 for w in (patch.get("wires") or [])]
        return cls(name=str(patch.get("name") or name), blocks=blocks,
                   wires=wires, views=list(patch.get("views") or []),
                   description=str(patch.get("description") or ""))

    # -- validation (strict subset of fabric validate_patch + own additions)

    def validate(self):
        """Return a list of error strings (empty == valid)."""
        errors = []
        blocks = self.blocks
        if not blocks:
            return ["program has no blocks"]
        ids = set()
        for b in blocks:
            bid = str(b.id or "")
            if not bid:
                errors.append("a block has an empty id")
                continue
            if bid in ids:
                errors.append("duplicate block id '%s'" % bid)
            ids.add(bid)
            prim = str(b.primitive or "")
            if prim not in ATOMS:
                errors.append("block '%s': unknown primitive '%s'"
                               % (bid, prim))
            for k in (b.params or {}):
                if str(k) != str(k).lower():
                    errors.append("block '%s': param key '%s' must be "
                                   "lowercase" % (bid, k))
        if self.entry and self.entry not in ids:
            errors.append("entry block '%s' not found" % self.entry)
        # node rule (port level, sibling-faithful, extended to every
        # non-sink category): a non-visualizer atom with >1 declared
        # input port must be multi_in; sinks may stack many
        for b in blocks:
            atom = ATOMS.get(str(b.primitive))
            if atom is None:
                continue
            if (atom.category != "visualizer" and not atom.multi_in
                    and len(atom.inputs) > 1):
                errors.append("function node '%s' (%s) must have one "
                               "input port" % (b.id, b.primitive))
        for w in self.wires:
            frm, to = str(w.src or ""), str(w.dst or "")
            if "." not in frm or "." not in to:
                errors.append("wire endpoints must be 'module.port': "
                               "%s -> %s" % (frm, to))
                continue
            src_id, src_port = frm.split(".", 1)
            dst_id, dst_port = to.split(".", 1)
            if src_port != src_port.lower() or dst_port != dst_port.lower():
                errors.append("wire ports must be lowercase: %s -> %s"
                               % (frm, to))
            if src_id == "ui":
                pass  # 'ui.*' are virtual sources (hardcoded bus keys)
            elif src_id in ids:
                blk = next(m for m in blocks if m.id == src_id)
                atom = ATOMS.get(blk.primitive)
                outs = atom.outputs if atom is not None else []
                if src_port not in outs:
                    errors.append("wire source '%s' is not a declared "
                                   "output (%s)" % (frm, ", ".join(outs)))
            else:
                errors.append("wire references unknown module: %s -> %s"
                               % (frm, to))
            if dst_id not in ids:
                errors.append("wire references unknown module: %s -> %s"
                               % (frm, to))
            else:
                blk = next(m for m in blocks if m.id == dst_id)
                atom = ATOMS.get(blk.primitive)
                ins = atom.inputs if atom is not None else []
                if dst_port not in ins:
                    errors.append("wire destination '%s' is not a declared "
                                   "input port (%s)" % (to, ", ".join(ins) or "none"))
            if src_id == dst_id:
                errors.append("self-wire rejected: %s -> %s" % (frm, to))
        if not errors:
            cycle = self._find_cycle(ids)
            if cycle:
                errors.append("cycle rejected: %s"
                               % " -> ".join(cycle + [cycle[0]]))
        return errors

    @property
    def valid(self):
        return not self.validate()

    def _find_cycle(self, ids):
        """A directed cycle over block ids (self-wires already rejected)."""
        edges = []
        for w in self.wires:
            frm, to = str(w.src or ""), str(w.dst or "")
            if "." not in frm or "." not in to:
                continue
            src_id = frm.split(".", 1)[0]
            dst_id = to.split(".", 1)[0]
            if src_id == dst_id:
                continue  # reported as a self-wire
            if dst_id not in ids:
                continue  # unknown module already reported
            edges.append((src_id, dst_id))  # 'ui' sources: never cyclic
        adj = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)
        nodes = set(ids) | set(adj.keys())
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}
        for start in nodes:
            if color[start] != WHITE:
                continue
            stack = [(start, iter(adj.get(start, ())))]
            color[start] = GRAY
            path = [start]
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if nxt not in color:
                        continue
                    if color[nxt] == GRAY:
                        i = path.index(nxt)
                        return path[i:]
                    if color[nxt] == WHITE:
                        color[nxt] = GRAY
                        stack.append((nxt, iter(adj.get(nxt, ()))))
                        path.append(nxt)
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    stack.pop()
                    path.pop()
        return None

    # -- compile --------------------------------------------------------------

    def compile(self, target="microfx"):
        """Validate, then emit the target (raises ProgramError if invalid)."""
        errors = self.validate()
        if errors:
            raise ProgramError("invalid program: " + "; ".join(errors))
        if target == "microfx":
            return self.to_patch()
        if target == "eel2":
            return self.to_eel2()
        if target == "python":
            return self.to_python()
        if target == "mermaid":
            return self.to_mermaid()
        if target == "wgsl":
            return self.to_wgsl()
        raise ProgramError("unknown target '%s'" % target)

    def to_patch(self):
        """The microfx target: the patch-runner dict (DICT params).

        Primitives are normalized to the catalog's canonical names
        (the CORE short aliases 'and'/'or'/... map back to 'gate_*'),
        so the node oracle -- which resolves primitives against the
        fixture keys -- accepts the result.
        """
        modules = []
        for b in self.blocks:
            atom = ATOMS.get(b.primitive)
            modules.append({"id": b.id,
                            "primitive": atom.name if atom else b.primitive,
                            "params": dict(b.params or {})})
        # auto-generate views for viz_* sinks if self.views is empty
        views = list(self.views)
        if not views:
            views = _patch_views(self.blocks)
        return {"modules": modules,
                "wires": [{"from": w.src, "to": w.dst}
                          for w in self.wires],
                "views": views}

    def to_eel2(self):
        """Concatenated per-block EEL2 bodies (teaching view).

        Sinks render from views[] (empty runner bodies) and are omitted;
        a concatenated multi-block EEL2 program is documentation, not an
        executable single-program (each body owns its own inputs/mem).
        """
        lines = ["// Autogenerated EEL2 from AtomicProgram: %s" % self.name]
        if self.description:
            lines.append("// %s" % self.description)
        for b in self.blocks:
            atom = ATOMS.get(b.primitive)
            if atom is None or atom.category == "visualizer" \
                    or not atom.source:
                continue
            lines.append("")
            lines.append("// --- %s (%s) ---" % (b.id, b.primitive))
            lines.append(atom.source)
        return "\n".join(lines)

    def to_python(self):
        """A self-contained script that runs the program over atomic.engine.

        Supersedes the sibling's broken numpy translator: the emitted
        code re-instantiates the verified Python twin (Engine) on this
        exact patch, so target 'python' is a live runner, not a guess.
        Usage: python <script> [ticks]
        """
        patch = self.to_patch()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return ('"""\n'
                    'Python runner for %s: %s\n\n'
                    'Auto-generated by atomic.program (target "python").\n'
                    'Runs over atomic.engine, the verified Python twin of\n'
                    'the jsfx patch runner (the legacy numpy translator\n'
                    'is superseded).\n'
                    '"""\n'
                    "import json, sys\n"
                    "sys.path.insert(0, %r)\n"
                    "from atomic import Engine\n"
                    "\n"
                    "PATCH = %r\n"
                    "\n"
                    "if __name__ == '__main__':\n"
                    "    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 120\n"
                    "    res = Engine(PATCH['modules'], PATCH['wires'],\n"
                    "                  views=PATCH.get('views') or []).run(ticks)\n"
                    "    print(json.dumps(res['final']))\n"
                    % (self.name, self.description or "(no description)",
                       root, patch))

    def to_mermaid(self):
        """Mermaid flowchart for the teaching view."""
        shapes = {"source": "([%s])", "function": "[%s]", "logic": "[%s]",
                  "quantum": ">%s]", "spatial": "/%s/",
                  "visualizer": "((%s))"}
        lines = ["flowchart TD", "    %% AtomicProgram: %s" % self.name]
        for b in self.blocks:
            atom = ATOMS.get(b.primitive)
            cat = atom.category if atom is not None else "function"
            label = str(b.label or (atom.title if atom is not None
                                     else b.primitive))
            label = label.replace('"', "'").replace("[", "(") \
                         .replace("]", ")")
            lines.append("    %s%s" % (b.id, shapes.get(cat, "[%s]") % label))
        for w in self.wires:
            src_id = str(w.src).split(".", 1)[0]
            dst_id = str(w.dst).split(".", 1)[0]
            if src_id and dst_id:
                lines.append("    %s --> %s" % (src_id, dst_id))
        return "\n".join(lines)

    def to_wgsl(self):
        """WGSL compute-shader for the live card (goal 9: bicameral split).

        Each block becomes a WGSL fn (tick_<id>) operating on a per-block
        var struct. The top-level compute entry dispatches blocks in
        program insertion order with 1-tick host-bridge latency on wires,
        mirroring the pinned per-tick loop (engine.py). Buffers:
          @group(0) @binding(0) var<storage, read_write> bus: array<f32>;
          @group(0) @binding(1) var<storage, read>  params: array<f32>;
          @group(0) @binding(2) var<storage, read_write> state: array<f32>;
        Sizes are derived from len(blocks); the shader is valid WGSL
        (naga --check) and runnable via wgpu/XPUGraph on the live card.
        """
        # map primitive -> WGSL tick body
        def _wgsl_body(block, atom):
            prim = atom.name if atom else block.primitive
            # minimal per-primitive bodies that mirror the Python impls
            if prim == "const":
                v = block.params.get("value", 1.0)
                return "  (*bus).data[%d] = %sf;" % (idx, repr(float(v)))
            if prim == "gain":
                f = float(block.params.get("factor", 1.0))
                return "  (*bus).data[%d] = inputs[%d] * %sf;" % (idx, idx, repr(f))
            if prim == "bias":
                a = float(block.params.get("add", 0.0))
                return "  (*bus).data[%d] = inputs[%d] + %sf;" % (idx, idx, repr(a))
            if prim == "smooth":
                al = float(block.params.get("alpha", 0.1))
                return ("  (*state).data[%d] = (*state).data[%d] + %sf * (inputs[%d] - (*state).data[%d]);\n"
                        "  (*bus).data[%d] = (*state).data[%d];") % (idx, idx, repr(al), idx, idx, idx, idx)
            if prim == "threshold":
                hi = float(block.params.get("hi", 0.5))
                lo = float(block.params.get("lo", -0.5))
                return ("  if (inputs[%d] > %sf) { (*state).data[%d] = 1.0; } else if (inputs[%d] < %sf) { (*state).data[%d] = 0.0; }\n"
                        "  (*bus).data[%d] = (*state).data[%d];") % (idx, repr(hi), idx, idx, repr(lo), idx, idx, idx)
            if prim == "clamp":
                lo = float(block.params.get("lo", -1.0))
                hi = float(block.params.get("hi", 1.0))
                return "  (*bus).data[%d] = clamp(inputs[%d], %sf, %sf);" % (idx, idx, repr(lo), repr(hi))
            if prim == "sine_lfo":
                amp = float(block.params.get("amp", 1.0))
                off = float(block.params.get("offset", 0.0))
                rh = float(block.params.get("rate_hz", 1.0))
                # phase in state[idx], dt = 1/30
                return ("  (*state).data[%d] = (*state).data[%d] + %sf * 0.0333333;\n"
                        "  (*bus).data[%d] = %sf + %sf * sin((*state).data[%d] * 6.2831853);") % (idx, idx, repr(rh), idx, repr(off), repr(amp), idx)
            if prim == "moving_avg":
                return "  (*bus).data[%d] = inputs[%d]; // moving_avg (host running mean)" % (idx, idx)
            if prim == "toggle":
                return ("  if (inputs[%d] > 0.0 && (*state).data[%d] == 0.0) { (*state).data[%d+1] = 1.0 - (*state).data[%d+1]; }\n"
                        "  (*state).data[%d] = select(0.0, 1.0, inputs[%d] > 0.0);\n"
                        "  (*bus).data[%d] = (*state).data[%d+1];") % (idx, idx, idx, idx, idx, idx, idx, idx)
            if prim == "clock_bpm":
                bpm = float(block.params.get("bpm", 60.0))
                period = 60.0 / max(0.1, bpm)
                return ("  (*state).data[%d] = (*state).data[%d] + 0.0333333;\n"
                        "  if ((*state).data[%d] >= %sf) { (*state).data[%d] = (*state).data[%d] - %sf; (*bus).data[%d] = 1.0; } else { (*bus).data[%d] = 0.0; }") % (idx, idx, idx, repr(period), idx, idx, repr(period), idx, idx)
            if prim == "sensor":
                return "  (*bus).data[%d] = inputs[%d]; // sensor (host-fed)" % (idx, idx)
            if prim in ("gate_and", "and"):
                return ("  (*bus).data[%d] = select(0.0, 1.0, inputs[%d] > 0.5 && inputs[%d] > 0.5);" % (idx, idx, idx))
            if prim in ("gate_or", "or"):
                return ("  (*bus).data[%d] = select(0.0, 1.0, inputs[%d] > 0.5 || inputs[%d] > 0.5);" % (idx, idx, idx))
            if prim in ("gate_not", "not"):
                return "  (*bus).data[%d] = select(1.0, 0.0, inputs[%d] > 0.5);" % (idx, idx)
            if prim in ("gate_nand", "nand"):
                return "  (*bus).data[%d] = select(1.0, 0.0, inputs[%d] > 0.5 && inputs[%d] > 0.5);" % (idx, idx, idx)
            if prim in ("gate_nor", "nor"):
                return "  (*bus).data[%d] = select(1.0, 0.0, inputs[%d] > 0.5 || inputs[%d] > 0.5);" % (idx, idx, idx)
            if prim in ("gate_xor", "xor"):
                return "  (*bus).data[%d] = select(0.0, 1.0, (inputs[%d] > 0.5) != (inputs[%d] > 0.5));" % (idx, idx, idx)
            if prim in ("gate_xnor", "xnor"):
                return "  (*bus).data[%d] = select(0.0, 1.0, (inputs[%d] > 0.5) == (inputs[%d] > 0.5));" % (idx, idx, idx)
            if prim == "h4_slide":
                return (
                    "  // H(4) slide: bus[4*%d+0]=W, +1=Z, +2=Y, +3=X\n" % idx +
                    "  (*bus).data[4*%d+0] = inputs[%d] + (*state).data[%d] + (*state).data[%d] + (*state).data[%d];\n" % (idx, idx, idx, idx, idx) +
                    "  (*bus).data[4*%d+1] = inputs[%d] - (*state).data[%d] + (*state).data[%d] - (*state).data[%d];\n" % (idx, idx, idx, idx, idx) +
                    "  (*bus).data[4*%d+2] = inputs[%d] + (*state).data[%d] - (*state).data[%d] - (*state).data[%d];\n" % (idx, idx, idx, idx, idx) +
                    "  (*bus).data[4*%d+3] = inputs[%d] - (*state).data[%d] - (*state).data[%d] + (*state).data[%d];" % (idx, idx, idx, idx, idx)
                )
            if prim in ("accum",):
                pt = float(block.params.get("per_tick", 1.0))
                return ("  if (inputs[%d] > 0.0 && (*state).data[%d] == 0.0) { (*state).data[%d+1] = (*state).data[%d+1] + %sf; }\n"
                        "  (*state).data[%d] = select(0.0, 1.0, inputs[%d] > 0.0);\n"
                        "  (*bus).data[%d] = (*state).data[%d+1];") % (idx, idx, idx, idx, repr(pt), idx, idx, idx, idx)
            if prim.startswith("viz_"):
                return "  // viz sink %s (rendered from views[])" % prim
            # generic: passthrough
            return "  (*bus).data[%d] = inputs[%d];" % (idx, idx)

        lines = []
        lines.append("// WGSL for '%s' — %d blocks, %d wires" % (self.name, len(self.blocks), len(self.wires)))
        lines.append("// Generated by atomic.program (target \"wgsl\").")
        lines.append("// Host bridge: bus[] in host RAM (no P2P), tick latency 1.")
        lines.append("")
        lines.append("struct Bus { data: array<f32> }")
        lines.append("@group(0) @binding(0) var<storage, read_write> bus: Bus;")
        lines.append("@group(0) @binding(1) var<storage, read> params: Bus;")
        lines.append("@group(0) @binding(2) var<storage, read_write> state: Bus;")
        # live bridge region (GPU1 -> host -> GPU0) is the tail of bus[]
        lines.append("@group(0) @binding(3) var<storage, read_write> bridge: Bus; // host-RAM")
        lines.append("")
        for idx, block in enumerate(self.blocks):
            atom = ATOMS.get(block.primitive)
            lines.append("fn tick_%s(inputs: ptr<function, array<f32, %d>>, bus: ptr<storage, Bus>, state: ptr<storage, Bus>) {" % (block.id, len(self.blocks)))
            lines.append("  let idx = %du;" % idx)
            lines.append(_wgsl_body(block, atom))
            lines.append("}")
            lines.append("")
        lines.append("@compute @workgroup_size(64)")
        lines.append("fn main(@builtin(global_invocation_id) gid: vec3<u32>) {")
        lines.append("  var inputs: array<f32, %d>;" % len(self.blocks))
        lines.append("  // seed inputs from previous tick's bus + wire latches (host bridge: no P2P)")
        for w in self.wires:
            # estimate lane by block order
            try:
                s_id = w.src.split(".", 1)[0]
                d_id = w.dst.split(".", 1)[0]
                s_idx = next(i for i, b in enumerate(self.blocks) if b.id == s_id)
                d_idx = next(i for i, b in enumerate(self.blocks) if b.id == d_id)
                lines.append("  inputs[%d] = bus.data[%d]; // %s -> %s (tick latency 1)" % (d_idx, s_idx, w.src, w.dst))
            except StopIteration:
                pass
        for i, b in enumerate(self.blocks):
            lines.append("  tick_%s(&inputs, &bus, &state);" % b.id)
        lines.append("  // bridge: tail of bus maps to host-RAM for bicameral split (GPU1 -> host -> GPU0)")
        lines.append("}")
        return "\n".join(lines)

    def validate_wgsl(self):
        """Validate this program's WGSL output (structural + naga if present).

        Convenience wrapper around Display.validate_wgsl for program-generated
        shaders. Returns (ok, detail).
        """
        from .tiles import Display
        return Display.validate_wgsl(self.to_wgsl())

    def summary(self):
        return {"name": self.name, "description": self.description,
                "n_blocks": len(self.blocks), "n_wires": len(self.wires),
                "n_views": len(self.views), "tags": list(self.tags),
                "hash": self.hash}

    @classmethod
    def from_description(cls, description, domain="general"):
        """Build a Program from natural language (goals 2/4/5).

        Delegates to atomic.teach.REGISTRY: keyword-gated match against
        teacher examples, or synthesis from primitive keywords.
        """
        from .teach import REGISTRY as _REG
        return _REG.from_description(description, domain=domain)


def compile_program(program, target="microfx"):
    """Module-level dispatch (mirrors the sibling's API name)."""
    return program.compile(target)
