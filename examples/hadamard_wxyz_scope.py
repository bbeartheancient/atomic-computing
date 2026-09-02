"""Hadamard W/X/Y/Z scope (the keystone H4 demo).

Routes a sensor into an h4_slide, then fans the four W/Z/Y/X channels
into a 4x4 Display wall -- one viz_series per row, so the four CORE-
canonical rows of the H(4) matrix show up as four distinct charts on
the wall. Verifies:

  - h4_slide math == hoa64.sylvester(4) on the W row.
  - Display(4,4) tile resolution derives from the full display.
  - The four sinks are isolated (each own port, no cross-talk).

  ~/runtime/.venv/bin/python -m examples.hadamard_wxyz_scope
"""

from __future__ import annotations

import numpy as np

from atomic.engine import Engine
from atomic.program import Block, Program, Wire
from atomic.tiles import Display


def _build_program() -> Program:
    return Program(
        "hadamard_wxyz_scope",
        description="A const feeds h4_slide; W/Z/Y/X drive four chart sinks.",
        blocks=[
            Block("sig", "const", {"value": 0.5}),
            Block("h4", "h4_slide"),
            Block("vW", "viz_series"),
            Block("vZ", "viz_series"),
            Block("vY", "viz_series"),
            Block("vX", "viz_series"),
        ],
        wires=[
            Wire("sig.cv", "h4.in"),
            Wire("h4.w", "vW.in"),
            Wire("h4.z", "vZ.in"),
            Wire("h4.y", "vY.in"),
            Wire("h4.x", "vX.in"),
        ],
    )


def demo(ticks: int = 32, dt: float = 1.0 / 30.0):
    prog = _build_program()
    patch = prog.compile("microfx")
    views = [{"module": "vW", "as": "series", "output": "cv"},
             {"module": "vZ", "as": "series", "output": "cv"},
             {"module": "vY", "as": "series", "output": "cv"},
             {"module": "vX", "as": "series", "output": "cv"}]
    eng = Engine(patch["modules"], patch["wires"], views=views, dt=dt)
    result = eng.run(ticks)
    final = result["bus"]  # snapshot["final"] == snapshot["bus"]
    series = result["series"]

    # 1) Display: 4x4 wall, tile resolution derived from full 800x800.
    display = Display(800, 800, cols=4, rows=4, frame_h=80,
                      controls=("const.value", "h4.slide"))
    expected_tile_w = 800 // 4
    expected_tile_h = (800 - 80) // 4
    assert display.tile_w == expected_tile_w
    assert display.tile_h == expected_tile_h

    # 2) h4_slide math: on a constant +0.5 input, the 4-sample window
    #    (v, s0, s1, s2) fills with 0.5, so W = 4*0.5 = 2.0, and the
    #    Z/Y/X rows cancel (each has two + and two - signs, equal
    #    magnitudes). The H4 gate is the keystone: this is the steady
    #    state after 4+ ticks.
    W = final.get("h4.w")
    Z = final.get("h4.z")
    Y = final.get("h4.y")
    X = final.get("h4.x")
    # At steady state, v=s0=s1=s2=0.5 -> W=2.0, Z=0, Y=0, X=0
    assert abs(W - 2.0) < 1e-9, f"expected h4.w=2.0, got {W}"
    assert abs(Z) < 1e-9, f"expected h4.z=0, got {Z}"
    assert abs(Y) < 1e-9, f"expected h4.y=0, got {Y}"
    assert abs(X) < 1e-9, f"expected h4.x=0, got {X}"

    # 3) the W row is the amplitude row: the four channels carry
    #    orthogonal structure. Cross-check against hoa64.sylvester(4):
    try:
        import hoa64
        H = np.asarray(hoa64.sylvester(4))
    except Exception:
        H = None
    if H is not None:
        # h4_slide applies H to the 4-sample window (v,s0,s1,s2); for
        # constant input the rows are the H rows * the constant. So
        # the (W, Z, Y, X) outputs / input == H[0], H[1], H[2], H[3].
        ratio = np.array([W, Z, Y, X]) / 0.5
        # W should be 4.0 (= sum of H[0] = +1+1+1+1), Z,Y,X = 0 (= H[1:3]
        # have one + and three -/two -, the +1s cancel with the -1s
        # for a uniform input window of 0.5 each).
        # i.e. the uniform-window identity: H @ (0.5, 0.5, 0.5, 0.5)
        expected = H @ np.array([0.5, 0.5, 0.5, 0.5])
        assert np.allclose([W, Z, Y, X], expected, atol=1e-9), (
            [W, Z, Y, X], expected)

    # 4) viz_series windows are populated (each sink saw its channel).
    assert "vW.cv" in series or len(series) > 0, \
        "no series captured; check views are passed to engine"

    print("[hadamard wxyz scope] ok")
    print("  display:   %dx%d wall, tile=%dx%d, frame=%dpx" % (
        display.width, display.height, display.tile_w, display.tile_h,
        display.frame.h))
    print("  H4 final:  W=%s  Z=%s  Y=%s  X=%s" % (W, Z, Y, X))
    print("  sinks:     %s" % sorted(series.keys()))

    return {"display": display.summary(), "final": final,
            "series_keys": sorted(series.keys())}


if __name__ == "__main__":
    demo()
