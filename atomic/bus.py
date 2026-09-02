"""Bus: the one signal store, plus the per-node host bridge.

Node mirrors exactly the surface jsfx's evaluatePatch hands each module
(fabric/web/jsfx.js):

  input(nm)     m.inputs[nm.toLowerCase()] || 0          (line 1136)
  output(nm,v)  bus[m.id + "." + nm] = v                 (line 1137)
  id read       lowercase key: raw vars[key] if the var
                exists, else Number(params[key]) || 0,
                else undefined                            (lines 313-320)
  var(name)     vars[key] || 0  (assign-source read, 300-303)
  store/load    m.hostState, fresh per run                (1138-1139)
  mem[base+idx] addr = ToInt32(floor(base + idx + 1e-5)) (328, 424);
                a missing/out-of-range slot reads as 0, an
                out-of-range write is a silent no-op

A None on the bus means JS `undefined`: the wire latch skips it, while
a real NaN is finitized to 0; the two must stay distinct.
"""

import math

from .jsnum import js_falsy, js_number, js_or0, to_int32

MEM_SIZE = 1 << 20
MEM_EPS = 0.00001  # jsfx.js:328/424, float-drift guard


class Port:
    """A typed socket on a block (used by the IR; the bus is name-keyed)."""

    def __init__(self, name, kind="cv", direction="out"):
        self.name = name
        self.kind = kind
        self.direction = direction


class Node:
    """One module instance: one JS `m` bag + one Interpreter (1127-1141)."""

    def __init__(self, nid, primitive, params, bus):
        self.id = nid
        self.primitive = primitive
        self.params = params  # defaults (lowercase) + patch keys verbatim
        self.inputs = {}
        self.outputs = {}
        self.hostState = {}
        self.vars = {}
        self.mem = {}  # sparse stand-in for the 2**20 Float64Array
        self.bus = bus

    # -- host bridge --------------------------------------------------------

    def input(self, nm):
        return js_or0(self.inputs.get(str(nm).lower()))

    def output(self, nm, v):
        self.bus.set(self.id + "." + str(nm), v)

    def read(self, key):
        """Id resolution (313-320): raw var, else param (Number||0), else None."""
        k = str(key).lower()
        if k in self.vars:
            return self.vars[k]
        if k in self.params:
            n = js_number(self.params[k])
            return 0.0 if js_falsy(n) else n
        return None

    def var0(self, key):
        """Interpreter.var (300-303): vars[key] || 0 (assign-source read)."""
        return js_or0(self.vars.get(str(key).lower(), None))

    def set_var(self, name, v):
        self.vars[str(name).lower()] = v

    def store(self, k, v):
        self.hostState[str(k)] = v

    def load(self, k):
        return js_or0(self.hostState.get(str(k)))

    def mem_addr(self, base, idx):
        return to_int32(math.floor(js_number(base) + js_number(idx) + MEM_EPS))

    def mem_read(self, base, idx):
        a = self.mem_addr(base, idx)
        return js_or0(self.mem.get(a)) if 0 <= a < MEM_SIZE else js_or0(None)

    def mem_write(self, base, idx, v):
        a = self.mem_addr(base, idx)
        if 0 <= a < MEM_SIZE:
            self.mem[a] = v


class Wire:
    """One cable: fully-qualified bus keys. Rack v2: cables into one
    input SUM in wire order (the latch does the adding)."""

    def __init__(self, src, dst):
        self.src = src
        self.dst = dst


class Bus:
    """The one signal bus; a missing key == JS `undefined` (None)."""

    def __init__(self):
        self._d = {}

    def set(self, key, v):
        self._d[key] = v

    def get(self, key):
        return self._d.get(key)

    def keys(self):
        return list(self._d.keys())

    def items(self):
        return list(self._d.items())

    def snapshot(self):
        return dict(self._d)
