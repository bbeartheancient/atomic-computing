"""Engine: the Python twin of evaluatePatch (fabric/web/jsfx.js:1089-1182).

The pinned per-tick mirror (ATOMIC-PC-STATE.md, iter 2 findings):
  (1) bus["ui.tap"] = 1 if t in ui_taps else 0  — the key is HARDCODED
      (oracle parity requires the tap module id to be "ui");
  (1b) LIVE FEED (mode 2): a per-tick feed may OVERRIDE (a) the tap for
      this tick and (b) a module's params BEFORE the module ticks. A
      live param is read at eval time (Q1, iter 2) so a between-tick
      write is seen by the next tick; this mirrors the resident driver's
      per-tick hook (atomic/oracle.py MODE 2). A feed entry is
      {tick: {"taps": [..] (optional), "params": {id: {key: v}}}}; an
      absent tick = "no live change" (the static ui_taps still applies);
      "taps" overrides the static ui_taps for that tick (empty list =
      force no tap); "params" is merged over the node's params (the IR
      keeps keys lowercase, so the twin's lowercased read agrees with
      the JS proxy's verbatim m.params).
  (2) modules tick in PATCH INSERTION order (no topological sort);
      @init once at t==0, then @tick; host sources run their hostTick
      (NOT inside the try/catch — a host-source throw kills the run,
      exactly like the JS runner);
  (3) end-of-tick wire latches: case-sensitive bus[from]; None (JS
      undefined) -> skip (no contribution, no 0); non-finite -> 0;
      (sums.get(to) || 0) + v in wire order; then
      target.inputs[inp.toLowerCase()] = sum  =>  1-TICK LATENCY per
      hop (a consumer's tick t+1 reads what its source wrote at tick t);
  (4) views: append bus[key] (skip None), shift when the window > 512
      => oracle runs must be <= 512 ticks; longer runs compare final.
Per-run self-containment: a fresh Engine is a fresh run — hostState,
vars and mem never carry across runs. `run(ticks)` is the batch path
(the mode-1 twin of evaluatePatch); `tick()` is the live path (mode 2)
that steps one tick at a time on the SAME persistent state.

(5) OPTIONAL FLOW TRACE (step 6, "the trace is the bridge"): pass
    trace=<FlowTrace> to record per-tick stimulus + one FrameEntry
    per node (in_ports latched at tick start, out_ports on the bus at
    tick end, wall-clock latency_us). The trace is a PURE OBSERVER —
    it reads node.inputs / the bus and writes only into its own
    rings, so a traced run is bit-identical to an untraced one
    (pinned by tests/test_trace.py).
"""

import time

from .bus import Bus, Node, Wire
from .jsnum import is_finite, js_falsy, js_number, js_or0

VIEW_WINDOW = 512


class Engine:
    """Reference stream engine: one bus, the pinned per-tick loop."""

    def __init__(self, modules, wires, views=(), dt=1.0 / 30.0,
                 ui_taps=None, atoms=None, feeds=None, trace=None):
        if atoms is None:
            from .gates import ATOMS
            atoms = ATOMS
        self.atoms = atoms
        self.dt = float(dt) if not js_falsy(dt) else 1.0 / 30.0
        self.taps = set(ui_taps or [])
        # mode-2 live feeds: {tick: {"taps": [...], "params": {id:{k:v}}}}
        self.feeds = feeds or {}
        # step-6 flow observer (atomic.trace.FlowTrace); None = off
        self.trace = trace
        self._t = 0
        self.bus = Bus()
        self.nodes = []
        self.by_id = {}
        # params = defaultsFor(primitive) merged with the patch params
        # (Object.assign semantics: patch keys win, keys kept verbatim)
        for mod in modules:
            nid = mod["id"]
            prim = mod.get("primitive", "")
            atom = atoms.get(prim)
            params = dict(atom.params) if atom is not None else {}
            params.update(mod.get("params") or {})
            node = Node(nid, prim, params, self.bus)
            self.nodes.append(node)
            self.by_id[nid] = node
        self.wires = []
        for w in wires:
            if isinstance(w, Wire):
                self.wires.append(w)
            elif isinstance(w, dict):
                self.wires.append(Wire(w["from"], w["to"]))
            else:
                self.wires.append(Wire(w[0], w[1]))
        self.series = {}
        for view in (views or []):
            out = view.get("output")
            key = str(view.get("module", "")) + "." + \
                (str(out) if not js_falsy(out) else "cv")
            self.series[key] = []

    # -- run API -----------------------------------------------------------

    def run(self, ticks):
        """Self-contained run of `ticks` ticks (fresh state per Engine)."""
        for _ in range(int(ticks)):
            self.tick()
        return self.snapshot()

    def snapshot(self):
        return {"bus": self.bus.snapshot(),
                "series": {k: list(v) for k, v in self.series.items()},
                "final": self.bus.snapshot()}

    def tick(self):
        """One step of the engine (live in mode 2; state persists)."""
        self._tick(self._t)
        self._t += 1

    # -- the pinned loop ----------------------------------------------------

    def _tick(self, t):
        bus = self.bus
        bus.set("ui.tap", 1 if t in self.taps else 0)
        # mode-2 live hook: a per-tick feed may override the tap and/or
        # merge live params BEFORE the module ticks (mirrors the resident
        # driver; an absent tick = no override, static ui_taps applies).
        feed = self.feeds.get(t)
        if feed:
            live_taps = feed.get("taps")
            if live_taps is not None:
                bus.set("ui.tap", 1 if t in live_taps else 0)
            live_params = feed.get("params")
            if live_params:
                for mid, prm in live_params.items():
                    node = self.by_id.get(mid)
                    if node is not None:
                        node.params.update(prm)
        dt = self.dt
        # flow trace (pure observer; None = off, zero overhead):
        # record the resolved external stimulus of this tick...
        tracing = self.trace is not None and self.trace.active
        if tracing:
            self.trace.begin_tick(t, bus.get("ui.tap"),
                                   feed.get("params") if feed else None)
        for node in self.nodes:
            atom = self.atoms.get(node.primitive)
            if atom is None:
                continue  # unknown primitive: empty body -> no-op
            in_snap = dict(node.inputs) if tracing else None
            t0 = time.perf_counter() if tracing else 0.0
            if atom.host is not None:
                atom.host(node, t, dt, bus)
            else:
                try:
                    if t == 0 and atom.init is not None:
                        atom.init(node)
                    if atom.tick is not None:
                        atom.tick(node)
                except Exception as e:
                    bus.set(node.id + ".error", str(e))
            if tracing:
                # the tap's trigger lives on the RUNNER-OWNED key
                # "ui.tap" (contract 11), not its declared "trig"
                if node.id == "ui" and node.primitive == "tap":
                    ports = ["tap"]
                else:
                    ports = list(atom.outputs)
                self.trace.record_node(
                    node.id, node.primitive, in_snap,
                    {p: bus.get(node.id + "." + p) for p in ports
                     if bus.get(node.id + "." + p) is not None},
                    (time.perf_counter() - t0) * 1e6, t)
        # wire latches (Rack v2: cables into one input SUM in wire order)
        sums = {}
        for w in self.wires:
            v = bus.get(w.src)
            if v is None:
                continue
            v = js_number(v)
            if not is_finite(v):
                v = 0.0
            sums[w.dst] = js_or0(sums.get(w.dst)) + v
        for key, s in sums.items():
            parts = key.split(".")
            target = self.by_id.get(parts[0])
            if target is not None:
                inp = parts[1] if len(parts) > 1 else ""
                target.inputs[inp.lower()] = s
        for key, arr in self.series.items():
            v = bus.get(key)
            if v is not None:
                arr.append(v)
                if len(arr) > VIEW_WINDOW:
                    arr.pop(0)
