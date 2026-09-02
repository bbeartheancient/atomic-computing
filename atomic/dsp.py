"""mdct/imdct ports of the jsfx builtins (fabric/web/jsfx.js:480-522).

Operation order is bit-faithful to the JS: Math.sin/cos is the only
shared-libm seam (V8 vs CPython may differ in the last ulp); every other
step is pure float64 arithmetic and must stay op-for-op. Sizes are
64..4096; windows must not cross a 65536-item boundary.
"""

import math

from .bus import MEM_SIZE
from .jsnum import js_number, js_or0, to_int32

MDCT_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
SEG = 65536
INV_SQRT2 = math.sqrt(0.5)  # Math.SQRT1_2 (the same double)
SQRT2 = math.sqrt(2.0)      # Math.SQRT2


def _rd(mem, off, i):
    """this.mem[(off+i)|0] || 0 — a missing/out-of-range slot folds to 0."""
    a = to_int32(off + i)
    return js_or0(mem.get(a)) if 0 <= a < MEM_SIZE else js_or0(None)


def _validate(off, N):
    off = math.floor(js_number(off))
    N = int(math.floor(js_number(N)))
    if N not in MDCT_SIZES:
        raise ValueError("mdct size must be 64|128|256|512|1024|2048|4096")
    if off % SEG + N > SEG:
        raise ValueError("mdct window crosses 64k boundary")
    return off, N, N >> 1


def mdct(mem, off, N):
    """Forward: N windowed samples -> N/2 coeffs written to the FIRST
    half only (jsfx.js:501-512); the second half of the window is
    untouched (TDAC overlap lives in the caller)."""
    off, N, half = _validate(off, N)
    tmp = [0.0] * half
    for k in range(half):
        acc = 0.0
        for nn in range(N):
            acc = acc + _rd(mem, off, nn) \
                * math.sin(math.pi / N * (nn + 0.5)) \
                * math.cos(math.pi / N * (nn + (half + 1) / 2) * (2 * k + 1))
        tmp[k] = INV_SQRT2 * acc
    for k in range(half):
        a = to_int32(off + k)
        if 0 <= a < MEM_SIZE:
            mem[a] = tmp[k]
    return 0


def imdct(mem, off, N):
    """Inverse: N/2 coeffs (first half) -> N windowed samples
    (jsfx.js:513-521); the caller overlap-adds adjacent blocks."""
    off, N, half = _validate(off, N)
    tmp2 = [0.0] * N
    for nn in range(N):
        acc = 0.0
        for k in range(half):
            acc = acc + _rd(mem, off, k) \
                * math.cos(math.pi / N * (nn + (half + 1) / 2) * (2 * k + 1))
        tmp2[nn] = (4 / N) * SQRT2 * acc * math.sin(math.pi / N * (nn + 0.5))
    for nn in range(N):
        a = to_int32(off + nn)
        if 0 <= a < MEM_SIZE:
            mem[a] = tmp2[nn]
    return 0
