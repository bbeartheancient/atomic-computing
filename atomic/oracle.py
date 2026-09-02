"""Oracle: drives the SIBLING jsfx.js patch runner under node.

MODE 1 (step 5a): batch protocol. A temp node script reads one JSON
request {patch, ticks, dt, ui_taps} on stdin, registers the 29-entry
fixture (every gate body), calls evaluatePatch, and prints
{final, series} as JSON. Runs must be <= 512 ticks (the series window
is not exposed growing); longer runs compare `final` only.

JSON note: node's JSON.stringify DROPS keys whose value is undefined,
so a bus key holding JS `undefined` arrives as a MISSING key. The
Python side must treat "key absent" == "key present as None" (see
tests/test_parity.py comparison).

MODE 2 (step 5b): LIVE protocol. A RESIDENT node process (LiveOracle)
reads line-delimited JSON requests over stdin and answers each with a
line-delimited {final, series} on stdout; the process stays alive
across runs (per-run state is still fresh: each run rebuilds modules,
hostState, the bus and the interpreter, so contract #12 holds). The
driver RE-IMPLEMENTS the pinned per-tick loop (a faithful twin of
evaluatePatch) and adds a per-tick HOOK: before each tick it (a) may
set bus["ui.tap"] from a live tap list and (b) may mutate a module's
params. Q1 (iter 2) is what makes live params possible: the patch
proxy reads m.params AT EVAL TIME, so mutating m.params between ticks
is picked up by the next tick -- something the batch evaluatePatch
cannot do (it copies params once at call start). A request is
{patch, ticks, dt, ui_taps, feeds} where feeds is {tick:
{"taps": [..], "params": {id: {key: v}}}} (JSON string keys on the
wire; the driver re-integers the tick). An empty/absent feeds makes
the driver byte-for-byte the same loop as evaluatePatch.

Parity chain (pinned in tests/test_oracle_live.py): FIRST driver loop
== evaluatePatch on the 6 static conformance patches (no feeds),
proving the re-implementation is a faithful twin; THEN driver loop
(live params + live taps) == the Python engine fed with the SAME feed
(the engine gains the matching per-tick feed in atomic/engine.py).

Sensor feeds are an ENGINE-ONLY extension (the sibling patch runner
leaves sensor a no-op; the DOM path's env.sensorValue is unreachable --
see the iter-2 defect list), so MODE 2 covers live TAPS + live PARAMS
only; harness-defined sensor feed semantics remain a decision flag.
"""

import json
import os
import subprocess
import tempfile

def _fabric_root():
    # standalone: prefer vendored ATOMIC-PC/fabric, fallback to sibling
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(here, "fabric"), os.path.expanduser("~/M1Multitronic/fabric")):
        if os.path.exists(os.path.join(cand, "web", "jsfx.js")):
            return cand
    return os.path.expanduser("~/M1Multitronic/fabric")

_FABRIC = _fabric_root()
JSFX = os.path.join(_FABRIC, "web", "jsfx.js")
FIXTURE = os.path.join(_FABRIC, "tests", "microfx_modules.json")
NODE = "node"

_DRIVER = """\
const fs = require("fs");
const M = require(process.argv[2]);
const FIX = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
M.registerModules(FIX.source, FIX.params);
let s = "";
process.stdin.on("data", (c) => { s += c; });
process.stdin.on("end", () => {
  const req = JSON.parse(s);
  const opts = { dt: req.dt || 1 / 30 };
  if (req.ui_taps) opts.ui_taps = req.ui_taps;
  const r = M.evaluatePatch(req.patch, req.ticks, opts);
  console.log(JSON.stringify({ final: r.final, series: r.series }));
});
"""


def run(patch, ticks, dt=1.0 / 30.0, ui_taps=None, node=NODE, timeout=120):
    """One batch oracle run; returns (final_dict, series_dict).

    Raises RuntimeError if the node process fails or prints nothing.
    """
    req = {"patch": patch, "ticks": ticks, "dt": dt}
    if ui_taps:
        req["ui_taps"] = list(ui_taps)
    fd, path = tempfile.mkstemp(suffix=".js", prefix="atomic_oracle_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(_DRIVER)
        out = subprocess.run(
            [node, path, JSFX, FIXTURE],
            input=json.dumps(req), capture_output=True, text=True,
            timeout=timeout)
        if out.returncode != 0:
            raise RuntimeError("node oracle failed: " + out.stderr[:800])
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError("node oracle printed nothing: " + out.stderr[:800])
        r = json.loads(lines[-1])
    finally:
        os.unlink(path)
    return r.get("final", {}), r.get("series", {})


# --------------------------------------------------------------------- MODE 2
#
# A RESIDENT node process: one spawn, many line-delimited JSON requests
# over stdin, one line-delimited {final,series} reply per request. The
# driver re-implements the pinned per-tick loop (a faithful twin of
# evaluatePatch, jsfx.js:1089-1182) and inserts a per-tick HOOK that the
# batch evaluatePatch cannot express: it may overwrite bus["ui.tap"] from
# a live tap list and may mutate m.params BEFORE the module ticks of tick
# t. That hook is exactly what makes params (and taps) LIVE: Q1 (iter 2)
# pinned that the patch-context proxy reads m.params at eval time, so a
# between-tick mutation is visible to the next tick. With an empty/absent
# `feeds` the hook is a no-op and the driver reduces to evaluatePatch --
# that is step 1 of the parity chain (driver loop == evaluatePatch on the
# 6 static conformance patches).

_LIVE_DRIVER = """\
const fs = require("fs");
const M = require(process.argv[2]);
const FIX = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
M.registerModules(FIX.source, FIX.params);
// defaultsFor is not exported by jsfx.js; replicate it from the
// registered param table (registerModules() set __MFX_MODULES_PARAMS = FIX.params).
const HOST = new Set(["clock_bpm", "sensor"]);
const defaultsFor = (name) => {
  const out = {};
  for (const prm of (FIX.params[name] || []))
    out[prm.name] = prm.default;
  return out;
};
// A faithful re-implementation of evaluatePatch (1089-1182) plus the
// per-tick live hook. A throw is reported (not fatal) so the resident
// process stays up; parity on the 6 static patches proves the loop.
function runLive(req) {
  const dt = req.dt || 1 / 30;
  const patch = req.patch;
  const ticks = req.ticks | 0;
  const feeds = req.feeds || {};
  const modules = (patch.modules || []).map((mod) => ({
    id: mod.id,
    primitive: mod.primitive,
    params: Object.assign({}, defaultsFor(mod.primitive), mod.params || {}),
    inputs: {}, outputs: {}, hostState: {},
  }));
  const byId = {};
  modules.forEach((m) => { byId[m.id] = m; });
  const wires = patch.wires || [];
  const bus = {};
  const series = {};
  for (const view of patch.views || [])
    if (view.as === "series")
      series[view.module + "." + (view.output || "cv")] = [];
  const instances = modules.map((m) => {
    const name = m.primitive;
    if (HOST.has(name)) {
      return { id: m.id, hostTick: (t) => {
        if (name === "clock_bpm") {
          const bpm = Number(m.params.bpm) || 60;
          const period = 60 / Math.max(0.1, bpm);
          m.hostState.acc = (m.hostState.acc || 0) + dt;
          if (m.hostState.acc >= period) {
            m.hostState.acc -= period;
            bus[m.id + ".trig"] = 1;
          } else bus[m.id + ".trig"] = 0;
        }
      }};
    }
    const srcSrc = FIX.source[name] || "";
    const compiled = srcSrc ? M.compile({ source: srcSrc })
      : { init: null, tick: null };
    const interp = new M.Interpreter({
      globals: new Proxy({}, {
        get: (_t, k) => {
          k = String(k);
          if (k in m.params) return Number(m.params[k]) || 0;
          return undefined;
        },
        has: () => true,
      }),
      input: (nm) => m.inputs[String(nm).toLowerCase()] || 0,
      output: (nm, v) => { bus[m.id + "." + nm] = v; },
      store: (k, v) => { m.hostState[k] = v; },
      load: (k) => m.hostState[k] || 0,
    });
    interp.deadline = Infinity;
    return { id: m.id, interp, compiled };
  });
  const taps = new Set(req.ui_taps || []);
  for (let t = 0; t < ticks; t++) {
    // --- per-tick live hook (the mode-2 extension; no-op when empty) ---
    const feed = feeds[String(t)];
    // JSON arrays have no .has(): wrap the live tap list in a Set.
    const tapList = (feed && feed.taps) ? new Set(feed.taps) : taps;
    bus["ui.tap"] = tapList.has(t) ? 1 : 0;
    if (feed && feed.params) {
      for (const id of Object.keys(feed.params)) {
        const m = byId[id];
        if (m) { const p = feed.params[id];
          for (const k of Object.keys(p)) m.params[k] = p[k]; }
      }
    }
    // --- the pinned loop (mirrors evaluatePatch exactly) ----------------
    for (const inst of instances) {
      if (inst.hostTick) { inst.hostTick(t * dt); continue; }
      try {
        if (inst.compiled.init && t === 0)
          inst.interp.evalNode(inst.compiled.init);
        if (inst.compiled.tick)
          inst.interp.evalNode(inst.compiled.tick);
      } catch (e) { bus[inst.id + ".error"] = e.message; }
    }
    const sums = new Map();
    for (const w of wires) {
      let v = bus[w.from];
      if (v === undefined) continue;
      v = Number(v);
      if (!Number.isFinite(v)) v = 0;
      sums.set(w.to, (sums.get(w.to) || 0) + v);
    }
    for (const [key, sum] of sums) {
      const [mid, inp] = key.split(".");
      const target = modules.find((x) => x.id === mid);
      if (target) target.inputs[inp.toLowerCase()] = sum;
    }
    for (const key of Object.keys(series)) {
      const v = bus[key];
      if (v !== undefined) {
        series[key].push(v);
        if (series[key].length > 512) series[key].shift();
      }
    }
  }
  return { final: Object.fromEntries(Object.entries(bus)), series };
}
let buf = "";
process.stdin.on("data", (c) => {
  buf += c;
  let i;
  while ((i = buf.indexOf("\\n")) >= 0) {
    const line = buf.slice(0, i);
    buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    let req;
    try { req = JSON.parse(line); }
    catch (e) { process.stdout.write(JSON.stringify({ error: "bad request: " + e.message }) + "\\n"); continue; }
    try {
      const r = runLive(req);
      process.stdout.write(JSON.stringify({ final: r.final, series: r.series }) + "\\n");
    } catch (e) {
      process.stdout.write(JSON.stringify({ error: String(e && e.message || e) }) + "\\n");
    }
  }
});
process.stdin.on("end", () => process.exit(0));
process.on("SIGINT", () => process.exit(0));
process.on("SIGTERM", () => process.exit(0));
"""


class LiveOracle:
    """MODE 2: a resident node driver; line-delimited JSON over stdio.

    One node process serves many `run()` calls (each request is a fresh
    self-contained run -- contract #12 -- because the driver rebuilds the
    modules/bus/interpreter per request). Use as a context manager so the
    process is always reaped. `run` raises RuntimeError if the request
    errors, the node process dies, or a reply does not arrive in time.

    Transport: UNBUFFERED binary pipes (bufsize=0). Replies are framed
    with a small Python-side accumulator and a select() deadline, so a
    wedged driver is detected (and the process reaped) instead of
    blocking the test forever.
    """

    def __init__(self, node=NODE):
        self.node = node
        self.proc = None
        self._path = None
        self._buf = b""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def start(self):
        import time
        fd, path = tempfile.mkstemp(suffix=".js", prefix="atomic_live_")
        self._path = path
        with os.fdopen(fd, "w") as f:
            f.write(_LIVE_DRIVER)
        self._buf = b""
        self.proc = subprocess.Popen(
            [self.node, path, JSFX, FIXTURE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0)
        # A driver with a syntax error exits at boot with a stderr message;
        # catch that here instead of letting the first run() time out.
        time.sleep(0.2)
        if self.proc.poll() is not None:
            err = self.proc.communicate()[1]
            self.proc = None
            if self._path and os.path.exists(self._path):
                os.unlink(self._path)
            self._path = None
            raise RuntimeError("live oracle node exited at boot: "
                               + err.decode("utf8", "replace")[:800])
        return self

    def close(self):
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        if self._path and os.path.exists(self._path):
            os.unlink(self._path)
        self._path = None

    def _readline(self, timeout):
        """One newline-terminated line from the driver, within `timeout` s."""
        import select
        while b"\n" not in self._buf:
            ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
            if not ready:
                raise RuntimeError("live oracle: no reply within %gs" % timeout)
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(
                    "live oracle: node closed stdout (rc=%r)"
                    % self.proc.poll())
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("utf8")

    def run(self, patch, ticks, dt=1.0 / 30.0, ui_taps=None, feeds=None,
            timeout=60):
        """One live request against the resident driver.

        Returns (final_dict, series_dict). `feeds` is {tick:
        {"taps": [..], "params": {id: {key: v}}}}; an absent tick means
        "no live change this tick" (the static ui_taps still applies).
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(
                "live oracle node process is not running (rc=%r)"
                % (getattr(self.proc, "returncode", None),))
        req = {"patch": patch, "ticks": ticks, "dt": dt}
        if ui_taps:
            req["ui_taps"] = list(ui_taps)
        if feeds:
            req["feeds"] = {str(t): f for t, f in feeds.items()}
        self.proc.stdin.write(json.dumps(req).encode("utf8") + b"\n")
        self.proc.stdin.flush()
        line = self._readline(timeout)
        resp = json.loads(line)
        if resp.get("error"):
            self.close()
            raise RuntimeError("driver reported: " + resp["error"])
        return resp.get("final", {}), resp.get("series", {})
