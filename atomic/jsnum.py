"""JS numeric fidelity for the Python twin of fabric/web/jsfx.js.

The twin must reproduce V8's numeric semantics at every seam where the twin
has to share: Number() conversion, the `x || 0` falsy fold, ToInt32
(`| 0`) addressing, Number.isFinite, and the EEL2 condition threshold
(Math.abs(c) > 1e-5).
"""

import math


def is_nan(x):
    return isinstance(x, float) and x != x


def is_finite(x):
    """Number.isFinite: true only for real, finite numbers."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return math.isfinite(x)


def to_int32(x):
    """ToInt32 (JS `| 0`): truncate toward zero, wrap mod 2**32, signed."""
    if x is None:
        return 0
    if isinstance(x, bool):
        n = 1 if x else 0
    elif isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return 0
        n = int(math.trunc(x))
    else:
        n = int(x)
    n %= 1 << 32
    return n - (1 << 32) if n >= (1 << 31) else n


def js_number(x):
    """Number(x): the conversion the wire latch and id resolution use."""
    if x is None:
        return float("nan")
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return 0.0
        try:
            return float(s)
        except ValueError:
            return float("nan")
    return float("nan")


def js_falsy(x):
    """JS truthiness: false, 0, -0, NaN, "", null/undefined are falsy."""
    if x is None or x is False:
        return True
    if isinstance(x, bool):
        return not x
    if isinstance(x, (int, float)):
        return is_nan(x) or x == 0
    if isinstance(x, str):
        return x == ""
    return False


def js_or0(x):
    """The `x || 0` fold (falsy values become 0)."""
    return 0.0 if js_falsy(x) else x


def cond_truthy(c):
    """EEL2 condition test (jsfx.js:336-340/382-386): Math.abs(c) > 1e-5.
    undefined/NaN are falsy (Math.abs(undefined) === NaN in JS)."""
    if c is None or isinstance(c, str):
        return False
    if isinstance(c, bool):
        c = 1.0 if c else 0.0
    if is_nan(c):
        return False
    return abs(c) > 1e-5
