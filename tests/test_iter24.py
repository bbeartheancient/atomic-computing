"""Goal A: wgsl naga hard-validate (iter 24).

Naga 30.0.1 (`cargo install naga-cli`, ~/.cargo/bin/naga) is the
canonical WGSL validator. The harness's WGSL codegen uses module-scope
@group(0) storage vars (bus 4*n, params/state/inputs n) with per-block
fn tick_<id>() writing to bus/inputs/state. No ptr<storage> args.
"""
import os
import shutil
import subprocess
import tempfile

from atomic import Program, Block, Wire


def _naga():
    n = shutil.which("naga")
    if not n:
        return None
    return n


def _validate(naga, wgsl):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".wgsl", delete=False) as fh:
        fh.write(wgsl)
        tmp = fh.name
    try:
        out = subprocess.run([naga, tmp], capture_output=True, text=True, timeout=10)
        return out.returncode == 0, out.stderr[:400]
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def test_naga_installed():
    naga = _naga()
    assert naga is not None, "naga not on PATH (install: cargo install naga-cli)"
    out = subprocess.run([naga, "--version"], capture_output=True, text=True, timeout=5)
    assert out.returncode == 0
    assert out.stdout.strip(), "naga --version returned empty"


def test_wgsl_h4_naga():
    naga = _naga()
    if not naga:
        return
    p = Program("h4", blocks=[
        Block("c0", "const", {"value": 1.0}),
        Block("g1", "gain", {"factor": 2.0}),
        Block("h1", "h4_slide"),
        Block("v0", "viz_series"),
    ], wires=[
        Wire("c0.cv", "g1.in"),
        Wire("g1.cv", "h1.in"),
        Wire("h1.w", "v0.in"),
    ])
    w = p.compile("wgsl")
    assert w.startswith("// WGSL")
    assert "struct Bus" in w
    assert "struct ParamsBus" in w
    assert "@group(0) @binding(0)" in w
    assert "@group(0) @binding(4)" in w
    assert "fn tick_c0" in w
    assert "fn tick_h1" in w
    ok, err = _validate(naga, w)
    assert ok, f"naga rejected H4 shader: {err}"


def test_wgsl_extended_naga():
    naga = _naga()
    if not naga:
        return
    p = Program("ext", blocks=[
        Block("c0", "const", {"value": 1}),
        Block("g1", "gain", {"factor": 2}),
        Block("b1", "bias", {"add": 0.5}),
        Block("th", "threshold", {"hi": 0.5, "lo": -0.5}),
        Block("cl", "clamp", {"lo": -1, "hi": 1}),
        Block("lf", "sine_lfo", {"rate_hz": 1.0}),
        Block("sm", "smooth", {"alpha": 0.1}),
        Block("h1", "h4_slide"),
        Block("v0", "viz_series"),
    ], wires=[
        Wire("c0.cv", "g1.in"),
        Wire("g1.cv", "b1.in"),
        Wire("b1.cv", "th.in"),
        Wire("th.gate", "sm.in"),
        Wire("b1.cv", "h1.in"),
        Wire("sm.cv", "v0.in"),
    ])
    w = p.compile("wgsl")
    ok, err = _validate(naga, w)
    assert ok, f"naga rejected extended shader: {err}"


def test_wgsl_simple_naga():
    naga = _naga()
    if not naga:
        return
    p = Program("simple", blocks=[
        Block("c0", "const", {"value": 1}),
        Block("g1", "gain", {"factor": 3}),
        Block("v0", "viz_series"),
    ], wires=[
        Wire("c0.cv", "g1.in"),
        Wire("g1.cv", "v0.in"),
    ])
    w = p.compile("wgsl")
    ok, err = _validate(naga, w)
    assert ok, f"naga rejected simple shader: {err}"


def test_wgsl_struct_shape():
    p = Program("struct", blocks=[
        Block("c0", "const", {"value": 1}),
        Block("g1", "gain", {"factor": 2}),
        Block("v0", "viz_series"),
    ], wires=[Wire("c0.cv", "g1.in"), Wire("g1.cv", "v0.in")])
    w = p.compile("wgsl")
    assert "struct Bus" in w
    assert "struct ParamsBus" in w
    assert "@group(0) @binding(0) var<storage, read_write> bus: Bus;" in w
    assert "@group(0) @binding(4) var<storage, read_write> inputs: ParamsBus;" in w
    assert "fn tick_c0()" in w
    assert "fn main(@builtin(global_invocation_id) gid: vec3<u32>)" in w
    assert "host-RAM" in w
    assert "tick latency 1" in w


def test_wgsl_bus_4n():
    p = Program("bus4n", blocks=[
        Block("c0", "const", {"value": 1.0}),
        Block("g1", "gain", {"factor": 2.0}),
        Block("h1", "h4_slide"),
        Block("v0", "viz_series"),
    ], wires=[
        Wire("c0.cv", "g1.in"),
        Wire("g1.cv", "h1.in"),
        Wire("h1.w", "v0.in"),
    ])
    w = p.compile("wgsl")
    # 4 blocks -> Bus is array<f32, 16>
    assert "struct Bus { data: array<f32, 16> }" in w
    # ParamsBus stays n=4
    assert "struct ParamsBus { data: array<f32, 4> }" in w
