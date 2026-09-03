"""VideoSynth (iter 35): pure-Python deterministic video synthesizer.

No GPU, pure numpy. Generates animated RGBA frames per tick using five
effects. Each effect is seeded by (prompt_hash + seed + tick), so the
same prompt always produces the same sequence.

H4-friendly color mapping:
  W = log-luma (amplitude / consensus channel)
  X = linear red channel
  Y = linear green channel
  Z = linear blue channel

Wire contract (compatible with InfiniteVideoLoop):
  video_synth.tick() -> H3Frame-like dict {rgba, prompt, seed, t}
  InfiniteVideoLoop.accept_source(VideoSynth) -> VideoSynthSource wrapper

Effects:
  wave       Sine-wave interference patterns (H4 W = amplitude, X/Y/Z = spatial axes)
  noise_field Perlin-style value noise with temporal evolution
  pixel_sort  Column-sort by brightness ("slop" aesthetic)
  mandelbrot  Fractal zoom (deterministic, seeded)
  fluid       Simple diffusion/advection simulation

Run: ~/runtime/.venv/bin/python -m atomic.video_synth
"""
from __future__ import annotations

import hashlib
import math
import random
import struct
import time

__all__ = [
    "VideoSynth",
    "VideoSynthSource",
    "EFFECTS",
    "PERLIN_PERMS",
]


EFFECTS = ("wave", "noise_field", "pixel_sort", "mandelbrot", "fluid")

_PERLIN_BASE = list(range(256))


def _build_perlin(seed: int):
    rng = random.Random(seed)
    p = _PERLIN_BASE[:]
    rng.shuffle(p)
    return p * 2


PERLIN_PERMS = {seed: _build_perlin(seed) for seed in range(16)}


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _grad(hash_val: int, x: float, y: float) -> float:
    h = hash_val & 3
    u = x if h & 1 == 0 else y
    v = y if h & 2 == 0 else x
    return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)


def _perlin_noise(x: float, y: float, perm: list[int]) -> float:
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    xf = x - math.floor(x)
    yf = y - math.floor(y)
    u = _fade(xf)
    v = _fade(yf)
    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]
    x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1.0, yf), u)
    x2 = _lerp(_grad(ab, xf, yf - 1.0), _grad(bb, xf - 1.0, yf - 1.0), u)
    return _lerp(x1, x2, v)


class VideoSynth:
    """Pure-Python deterministic video synthesizer.

    No GPU, no torch. Generates animated RGBA frames from a prompt + seed
    + tick triple. H4-friendly color mapping: W=log-luma, X/Y/Z=linear RGB.

    Args:
        width:     frame width in pixels (default 64)
        height:    frame height in pixels (default 64)
        effect:    one of EFFECT names (default "wave")
        seed:      base seed (default 0); actual seed = f(prompt, seed, tick)
        fps:       nominal frames per second (for time computation)

    The per-tick seed is: hash(prompt + str(seed) + str(tick)), so the
    same (prompt, seed, tick) triple always produces the same frame.
    """

    def __init__(self, width: int = 64, height: int = 64,
                 effect: str = "wave", seed: int = 0, fps: int = 30):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.effect = str(effect).lower()
        if self.effect not in EFFECTS:
            self.effect = "wave"
        self.seed = int(seed)
        self.fps = max(1, int(fps))
        self._t = 0
        self._calls = 0

    @property
    def t(self) -> int:
        return self._t

    def _prompt_hash(self, prompt: str) -> int:
        h = hashlib.sha256((str(prompt) + str(self.seed)).encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little")

    def _tick_seed(self, prompt: str, tick: int) -> int:
        combined = (self._prompt_hash(prompt) ^ (tick * 2654435761)) & 0xFFFFFFFF
        return combined

    def _rgb_from_wxyz(self, w: float, x: float, y: float, z: float) -> tuple[int, int, int]:
        r = max(0, min(255, int((x * 0.5 + 0.5) * 255)))
        g = max(0, min(255, int((y * 0.5 + 0.5) * 255)))
        b = max(0, min(255, int((z * 0.5 + 0.5) * 255)))
        return r, g, b

    def _luma(self, r: float, g: float, b: float) -> float:
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _log_luma(self, r: float, g: float, b: float) -> float:
        ly = self._luma(r, g, b)
        return math.log(max(1e-6, ly))

    def _render_wave(self, prompt: str, tick: int) -> bytes:
        w, h = self.width, self.height
        ts = tick / max(1, self.fps)
        rng = random.Random(self._tick_seed(prompt, tick))
        n = w * h
        out = bytearray(n * 4)
        cx, cy = w / 2.0, h / 2.0
        freq_scale = 0.08 + rng.random() * 0.04
        for i in range(n):
            x, y = i % w, i // w
            dx, dy = x - cx, y - cy
            d = math.sqrt(dx * dx + dy * dy)
            wave1 = math.sin(freq_scale * d * 6.2832 - tick * 0.5)
            wave2 = math.sin((x / w) * 3.0 + ts * 2.0) * math.cos((y / h) * 3.0 + ts * 1.5)
            wave3 = math.sin(((x + y) / (w + h)) * 4.0 - tick * 0.3)
            v = (wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2)
            v = v * 0.5 + 0.5
            r_amp = v + rng.uniform(-0.1, 0.1)
            g_amp = (1.0 - v) + rng.uniform(-0.1, 0.1)
            b_amp = math.sin(v * 3.14159 + tick * 0.2) * 0.5 + 0.5
            r = max(0, min(255, int(r_amp * 255)))
            g = max(0, min(255, int(g_amp * 255)))
            b = max(0, min(255, int(b_amp * 255)))
            j = i * 4
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            out[j + 3] = 255
        return bytes(out)

    def _render_noise_field(self, prompt: str, tick: int) -> bytes:
        w, h = self.width, self.height
        rng = random.Random(self._tick_seed(prompt, tick))
        n = w * h
        out = bytearray(n * 4)
        p_seed = rng.random() * 1000
        perlin_perm = PERLIN_PERMS[int(p_seed) % len(PERLIN_PERMS)]
        scale = 3.0 + rng.uniform(0.0, 2.0)
        octaves = 3
        for i in range(n):
            x, y = i % w, i // w
            nx, ny = x / w, y / h
            val = 0.0
            amp = 1.0
            freq = scale
            total = 0.0
            for _ in range(octaves):
                val += _perlin_noise(nx * freq + tick * 0.05,
                                      ny * freq + tick * 0.03,
                                      perlin_perm) * amp
                total += amp
                amp *= 0.5
                freq *= 2.0
            val /= total
            val = val * 0.5 + 0.5
            t_shift = tick / max(1.0, self.fps)
            r = max(0, min(255, int((_perlin_noise(nx * 2 + t_shift, ny * 2, perlin_perm) * 0.5 + 0.5) * 255)))
            g = max(0, min(255, int((_perlin_noise(nx * 2, ny * 2 + t_shift * 0.7, perlin_perm) * 0.5 + 0.5) * 255)))
            b = max(0, min(255, int(val * 255)))
            j = i * 4
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            out[j + 3] = 255
        return bytes(out)

    def _render_pixel_sort(self, prompt: str, tick: int) -> bytes:
        w, h = self.width, self.height
        rng = random.Random(self._tick_seed(prompt, tick))
        n = w * h
        base_r = int(rng.random() * 200) + 30
        base_g = int(rng.random() * 200) + 30
        base_b = int(rng.random() * 200) + 30
        raw = bytearray(n * 4)
        for i in range(n):
            x, y = i % w, i // w
            v = (math.sin(x / max(1, w) * 3.0 + tick * 0.5) *
                 math.cos(y / max(1, h) * 3.0 + tick * 0.3)) * 0.5 + 0.5
            noise_v = rng.random() * 0.3
            v = max(0, min(1, v + noise_v))
            r = max(0, min(255, int((base_r / 255.0 + v * 0.5) * 255)))
            g = max(0, min(255, int((base_g / 255.0 + v * 0.3) * 255)))
            b = max(0, min(255, int((base_b / 255.0 + v * 0.7) * 255)))
            j = i * 4
            raw[j] = r
            raw[j + 1] = g
            raw[j + 2] = b
            raw[j + 3] = 255
        out = bytearray(n * 4)
        sort_rows = max(1, int(h * rng.uniform(0.2, 0.8)))
        for row in range(h):
            row_pixels = raw[row * w * 4:(row + 1) * w * 4]
            if row < sort_rows and len(row_pixels) >= 4:
                cols = []
                for col_i in range(w):
                    j = col_i * 4
                    if j + 3 < len(row_pixels):
                        lum = self._luma(raw[row * w * 4 + j],
                                         raw[row * w * 4 + j + 1],
                                         raw[row * w * 4 + j + 2])
                        cols.append((lum, row_pixels[j:j + 4]))
                cols.sort(key=lambda p: p[0])
                for col_i, (_, px) in enumerate(cols):
                    j = col_i * 4
                    out[row * w * 4 + j:row * w * 4 + j + 4] = px
            else:
                out[row * w * 4:row * w * 4 + len(row_pixels)] = row_pixels
        return bytes(out)

    def _render_mandelbrot(self, prompt: str, tick: int) -> bytes:
        w, h = self.width, self.height
        rng = random.Random(self._tick_seed(prompt, tick))
        n = w * h
        out = bytearray(n * 4)
        zoom = 1.0 + tick * 0.02 + rng.uniform(0.0, 0.1)
        center_x = rng.uniform(-0.5, -0.3)
        center_y = rng.uniform(0.4, 0.6)
        max_iter = 50 + int(tick % 50)
        for i in range(n):
            x0 = (i % w - w / 2.0) / (0.5 * zoom * w) + center_x
            y0 = (i // w - h / 2.0) / (0.5 * zoom * h) + center_y
            x, y = 0.0, 0.0
            iter_count = 0
            for _ in range(max_iter):
                if x * x + y * y > 4.0:
                    break
                x_new = x * x - y * y + x0
                y = 2.0 * x * y + y0
                x = x_new
                iter_count += 1
            t_val = iter_count / max_iter
            t_val = math.pow(t_val, 0.5)
            r = max(0, min(255, int(t_val * 255)))
            g = max(0, min(255, int(t_val * 180)))
            b = max(0, min(255, int((1.0 - t_val) * 255)))
            j = i * 4
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            out[j + 3] = 255
        return bytes(out)

    def _render_fluid(self, prompt: str, tick: int) -> bytes:
        w, h = self.width, self.height
        rng = random.Random(self._tick_seed(prompt, tick))
        n = w * h
        out = bytearray(n * 4)
        density = [0.0] * n
        vx = [rng.uniform(-0.5, 0.5) for _ in range(n)]
        vy = [rng.uniform(-0.5, 0.5) for _ in range(n)]
        ts = tick / max(1, self.fps)
        src_x = int(rng.random() * (w - 1))
        src_y = int(rng.random() * (h - 1))
        src_i = src_y * w + src_x
        for step in range(20):
            density[src_i] += 0.05
            advect = [0.0] * n
            for i in range(n):
                xi, yi = i % w, i // w
                src_j = int(xi + vx[i]) % w + (int(yi + vy[i]) % h) * w
                advect[i] = density[src_j]
            density = advect
            for i in range(n):
                vx[i] += math.sin(i / max(1, n) * 6.0 + ts) * 0.02
                vy[i] += math.cos(i / max(1, n) * 6.0 + ts) * 0.02
        mx = max(max(density), 1e-6)
        for i in range(n):
            v = density[i] / mx
            v = max(0, min(1, v))
            r = max(0, min(255, int(v * 255)))
            g = max(0, min(255, int(v * 180)))
            b = max(0, min(255, int((1.0 - v) * 255)))
            j = i * 4
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            out[j + 3] = 255
        return bytes(out)

    def _render(self, prompt: str, tick: int) -> bytes:
        if self.effect == "noise_field":
            return self._render_noise_field(prompt, tick)
        elif self.effect == "pixel_sort":
            return self._render_pixel_sort(prompt, tick)
        elif self.effect == "mandelbrot":
            return self._render_mandelbrot(prompt, tick)
        elif self.effect == "fluid":
            return self._render_fluid(prompt, tick)
        else:
            return self._render_wave(prompt, tick)

    def tick(self, prompt: str = "") -> dict:
        """Generate one frame for the given prompt at the current tick.

        Returns a dict compatible with InfiniteVideoLoop's H3Frame interface:
          {rgba, prompt, seed, t, h3_latency_ms}
        """
        t0 = time.perf_counter()
        prompt = str(prompt)
        seed = self._tick_seed(prompt, self._t)
        rgba = self._render(prompt, self._t)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._calls += 1
        self._t += 1
        return {
            "rgba": rgba,
            "prompt": prompt,
            "seed": seed,
            "t": self._t - 1,
            "h3_latency_ms": elapsed,
        }

    def run(self, ticks: int, prompt: str = "") -> list[dict]:
        """Generate `ticks` frames."""
        out = []
        for _ in range(int(ticks)):
            out.append(self.tick(prompt))
        return out

    def generate(self, prompt: str, seed: int | None = None,
                 n_frames: int = 1, width: int | None = None,
                 height: int | None = None) -> dict:
        """H3Source-compatible generate() interface.

        Ignores seed (deterministic from prompt+tick), uses self.width/height
        unless overridden. Returns {prompt, seed, frames, h3_latency_ms, width, height}.
        """
        w = int(width) if width else self.width
        h = int(height) if height else self.height
        frames = []
        for i in range(int(n_frames)):
            result = self.tick(prompt)
            f = result["rgba"]
            if w != self.width or h != self.height:
                f = self._resize(f, w, h)
            frames.append(f)
        return {
            "prompt": prompt,
            "seed": self._tick_seed(prompt, 0),
            "frames": frames,
            "h3_latency_ms": result["h3_latency_ms"],
            "width": w,
            "height": h,
        }

    def _resize(self, rgba: bytes, new_w: int, new_h: int) -> bytes:
        if new_w == self.width and new_h == self.height:
            return rgba
        old_n = self.width * self.height
        new_n = new_w * new_h
        out = bytearray(new_n * 4)
        for ny in range(new_h):
            oy = int(ny * self.height / new_h)
            for nx in range(new_w):
                ox = int(nx * self.width / new_w)
                si = (oy * self.width + ox) * 4
                di = (ny * new_w + nx) * 4
                if si + 3 < len(rgba):
                    out[di:di + 4] = rgba[si:si + 4]
                else:
                    out[di:di + 4] = b"\x00\x00\x00\xff"
        return bytes(out)

    def h4_channels(self, rgba: bytes) -> tuple[float, float, float, float]:
        """Decode H4 channels from an RGBA frame.

        W = log-luma (amplitude / consensus)
        X = linear red
        Y = linear green
        Z = linear blue

        Uses the center pixel as a sample.
        """
        w, h = self.width, self.height
        n = w * h
        if len(rgba) < n * 4:
            return (0.0, 0.0, 0.0, 0.0)
        ci = (h // 2) * w + (w // 2)
        j = ci * 4
        r = rgba[j] / 255.0
        g = rgba[j + 1] / 255.0
        b = rgba[j + 2] / 255.0
        w_val = self._log_luma(r, g, b)
        x_val = r
        y_val = g
        z_val = b
        return (w_val, x_val, y_val, z_val)

    def stats(self) -> dict:
        return {
            "t": self._t,
            "calls": self._calls,
            "effect": self.effect,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    def __repr__(self) -> str:
        return (f"VideoSynth({self.width}x{self.height} "
                f"{self.effect} t={self._t})")


class VideoSynthSource:
    """VideoSynth wrapped as an H3Source-compatible object.

    Wraps a VideoSynth instance to expose the same .generate() interface
    as H3Stub/H3Client so InfiniteVideoLoop can accept either interchangeably.

    Usage:
        vs = VideoSynth(width=64, height=64, effect="wave")
        loop = InfiniteVideoLoop(vs, viewer, ...)
    """

    def __init__(self, synth: VideoSynth | None = None,
                 width: int = 64, height: int = 64,
                 effect: str = "wave", seed: int = 0):
        if synth is not None:
            self._synth = synth
        else:
            self._synth = VideoSynth(width=width, height=height,
                                      effect=effect, seed=seed)
        self.width = self._synth.width
        self.height = self._synth.height
        self._calls = 0

    def generate(self, prompt: str, seed: int | None = None,
                 n_frames: int = 1, width: int | None = None,
                 height: int | None = None) -> dict:
        self._calls += 1
        return self._synth.generate(prompt, seed=seed, n_frames=n_frames,
                                    width=width, height=height)

    @property
    def synth(self) -> VideoSynth:
        return self._synth

    @property
    def t(self) -> int:
        return self._synth.t

    @property
    def effect(self) -> str:
        return self._synth.effect

    def stats(self) -> dict:
        d = self._synth.stats()
        d["source_calls"] = self._calls
        return d

    def __repr__(self) -> str:
        return f"VideoSynthSource({self._synth!r})"


def _demo():
    synth = VideoSynth(width=8, height=8, effect="wave", seed=42)
    print(f"VideoSynth demo: {synth}")
    frames = synth.run(3, prompt="test animation")
    for f in frames:
        print(f"  t={f['t']} seed={f['seed']} bytes={len(f['rgba'])}")
    print(f"Stats: {synth.stats()}")
    w, x, y, z = synth.h4_channels(frames[-1]["rgba"])
    print(f"H4 channels: W={w:.3f} X={x:.3f} Y={y:.3f} Z={z:.3f}")
    src = VideoSynthSource(synth=synth)
    r = src.generate("prompt", n_frames=2)
    print(f"Source: {len(r['frames'])} frames, {r['h3_latency_ms']:.2f}ms")


if __name__ == "__main__":
    _demo()
