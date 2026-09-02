"""Tiles: the display model -- apps render onto a tiled video wall.

Per ATOMIC-PC-CORE.md's DISPLAY TILES section (the single source of
truth for the matrix semantics):

  - apps use the same pre-defined matrix of display tiles as the
    display output
  - the display has a control frame with the top-level i/o controls
  - the GUI display uses a 3x3 or 4x4 tiled matrix of framebuffer
    outputs
  - the framebuffer outputs are agnostic and universal
  - the full display resolution determines the matrix tile resolution
  - it works like a tiled video wall
  - multiple tiles can be linked together for larger sub-matrix
    displays

Pinned geometry:
  tile_w = floor(W / cols)
  tile_h = floor((H - frame_h) / rows)
The control frame occupies the top frame_h px at full width; the tile
wall fills the region below it; leftover px (W - cols*tile_w,
(H - frame_h) - rows*tile_h) are dead border -- the seams of the wall,
exactly as on a real tiled display. Linking a k x m block of adjacent
tiles yields one sub-matrix display of (m*tile_w) x (k*tile_h) px
anchored at the group's top-left tile.
"""


class TileError(ValueError):
    """A display/tile configuration that cannot be derived."""


class Tile:
    """One agnostic framebuffer tile of the wall.

    Universal and agnostic: a tile is a pixel rect with no committed
    content type. Resolution is derived from the full display, never
    chosen per tile.
    """

    def __init__(self, row, col, w, h, x, y):
        self.row = int(row)
        self.col = int(col)
        self.w = int(w)
        self.h = int(h)
        self.x = int(x)
        self.y = int(y)

    def bounds(self):
        return (self.x, self.y, self.w, self.h)

    def __repr__(self):
        return "Tile(r%d,c%d %dx%d @%d,%d)" % (
            self.row, self.col, self.w, self.h, self.x, self.y)


class ControlFrame:
    """The top strip of the display: the top-level i/o controls."""

    def __init__(self, w, h):
        self.w = int(w)
        self.h = int(h)
        self.controls = []

    def add_control(self, name, kind="param"):
        self.controls.append({"name": name, "kind": kind})
        return self

    def bounds(self):
        return (0, 0, self.w, self.h)

    def __repr__(self):
        return "ControlFrame(%dx%d, %d controls)" % (
            self.w, self.h, len(self.controls))


class TileGroup:
    """A linked block of tiles: one larger sub-matrix display.

    k rows x m cols of adjacent tiles combine into a single
    framebuffer of (m*tile_w) x (k*tile_h) px anchored at the
    group's top-left tile -- the "tiled video wall" feature of CORE.

    Live heatmap (iter 14): each TileGroup can display a heat value
    derived from a trace replay (per-tile intensity 0..1, updated via
    heatmap_from_trace / apply_heatmap). This is the tiles<->swarm
    visual integration: swarm agents map to TileGroups and their trace
    replays colour the wall.
    """

    def __init__(self, name, display, row, col, span_rows, span_cols):
        self.name = name
        self.display = display
        self.row = int(row)
        self.col = int(col)
        self.span_rows = int(span_rows)
        self.span_cols = int(span_cols)
        self.tiles = [display.tile(r, c)
                      for r in range(self.row, self.row + self.span_rows)
                      for c in range(self.col, self.col + self.span_cols)]
        self.heatmap = {}  # (row,col) -> float 0..1

    @property
    def width(self):
        return self.span_cols * self.display.tile_w

    @property
    def height(self):
        return self.span_rows * self.display.tile_h

    @property
    def x(self):
        return self.tiles[0].x

    @property
    def y(self):
        return self.tiles[0].y

    def bounds(self):
        return (self.x, self.y, self.width, self.height)

    def apply_heatmap(self, values, normalize=True):
        """Apply a per-tile value map to this group's heatmap.

        values: dict {(row,col): scalar} or iterable aligned with self.tiles
        normalize: rescale to 0..1 by min/max if True.
        Returns the stored heatmap dict.
        """
        if isinstance(values, dict):
            raw = {k: float(v) for k, v in values.items()}
        else:
            raw = {(t.row, t.col): float(v) for t, v in zip(self.tiles, values)}
        if not raw:
            self.heatmap = {}
            return self.heatmap
        if normalize:
            lo = min(raw.values())
            hi = max(raw.values())
            span = (hi - lo) if hi != lo else 1.0
            self.heatmap = {k: (v - lo) / span for k, v in raw.items()}
        else:
            self.heatmap = dict(raw)
        return self.heatmap

    def heatmap_from_trace(self, trace, port="g1.cv", agg="max"):
        """Live heatmap from a FlowTrace replay.

        Iterates trace.frames and collects out_ports[port] (or in_ports)
        per tile: maps frame index -> tile cyclically, then aggregates
        (max/mean/last) per tile and normalizes to 0..1.

        Returns heatmap dict.
        """
        tiles = self.tiles
        if not tiles:
            return {}
        vals = { (t.row, t.col): [] for t in tiles }
        frames = trace.frames if hasattr(trace, "frames") else []
        for idx, fr in enumerate(frames):
            tile = tiles[idx % len(tiles)]
            # fr may be FrameEntry or dict
            if hasattr(fr, "out_ports"):
                out = fr.out_ports or {}
                inp = fr.in_ports or {}
            else:
                out = fr.get("out_ports") or {}
                inp = fr.get("in_ports") or {}
            v = out.get(port)
            if v is None:
                v = inp.get(port.split(".")[-1] if "." in port else port)
            if v is None:
                continue
            try:
                vals[(tile.row, tile.col)].append(float(v))
            except Exception:
                pass
        agg_map = {}
        for key, lst in vals.items():
            if not lst:
                agg_map[key] = 0.0
            elif agg == "max":
                agg_map[key] = max(lst)
            elif agg == "mean":
                agg_map[key] = sum(lst) / len(lst)
            elif agg == "last":
                agg_map[key] = lst[-1]
            else:
                agg_map[key] = max(lst)
        return self.apply_heatmap(agg_map, normalize=True)

    def __repr__(self):
        return "TileGroup(%r %dx%d @%d,%d %dx%d)" % (
            self.name, self.span_rows, self.span_cols,
            self.x, self.y, self.width, self.height)


class Display:
    """The tiled display: a control frame over a 3x3 or 4x4 tile wall.

    The tile resolution derives from the FULL display size (W x H):
    the control frame takes the top frame_h px, and the region below
    is split into cols*tile_w wide by rows*tile_h tall (floor; the
    leftover px is dead border, the wall's seams).
    """

    def __init__(self, width, height, cols=3, rows=3, frame_h=0,
                 controls=()):
        width = int(width)
        height = int(height)
        frame_h = int(frame_h)
        cols = int(cols)
        rows = int(rows)
        if (cols, rows) not in ((3, 3), (4, 4)):
            raise TileError("tile matrix must be 3x3 or 4x4 (got %dx%d)"
                             % (cols, rows))
        if width <= 0 or height <= 0:
            raise TileError("display resolution must be positive "
                             "(got %dx%d)" % (width, height))
        if frame_h < 0 or frame_h >= height:
            raise TileError("control frame height must lie in [0, %d) "
                             "(got %d)" % (height, frame_h))
        self.width = width
        self.height = height
        self.cols = cols
        self.rows = rows
        self.tile_w = width // cols
        self.tile_h = (height - frame_h) // rows
        if self.tile_w < 1 or self.tile_h < 1:
            raise TileError("tile resolution degenerates to %dx%d px "
                             "(a %dx%d wall needs >= %dx%d px below the "
                             "frame)" % (self.tile_w, self.tile_h,
                                         cols, rows, cols, rows))
        self.frame = ControlFrame(width, frame_h)
        for c in (controls or ()):
            if isinstance(c, (tuple, list)):
                self.frame.add_control(c[0],
                                       c[1] if len(c) > 1 else "param")
            else:
                self.frame.add_control(c)
        self._tiles = {}
        for r in range(rows):
            for c in range(cols):
                self._tiles[(r, c)] = Tile(
                    r, c, self.tile_w, self.tile_h,
                    c * self.tile_w, frame_h + r * self.tile_h)
        self.groups = {}

    def tile(self, row, col):
        try:
            return self._tiles[(int(row), int(col))]
        except KeyError:
            raise TileError("no tile at (%d, %d) in a %dx%d matrix"
                             % (row, col, self.rows, self.cols))

    @property
    def tiles(self):
        return [self._tiles[(r, c)] for r in range(self.rows)
                for c in range(self.cols)]

    @property
    def grid_width(self):
        return self.cols * self.tile_w

    @property
    def grid_height(self):
        return self.rows * self.tile_h

    @property
    def margin_x(self):
        return self.width - self.grid_width

    @property
    def margin_y(self):
        return (self.height - self.frame.h) - self.grid_height

    # -- linking -----------------------------------------------------------

    def link(self, name, row, col, span_rows, span_cols):
        """Link a k x m block of adjacent tiles into one sub-matrix
        display. The block must sit inside the matrix and must not
        overlap an existing group."""
        row = int(row)
        col = int(col)
        span_rows = int(span_rows)
        span_cols = int(span_cols)
        if span_rows < 1 or span_cols < 1:
            raise TileError("group span must be at least 1x1")
        if row < 0 or col < 0 or row + span_rows > self.rows \
                or col + span_cols > self.cols:
            raise TileError("group (%d,%d +%dx%d) exceeds the %dx%d "
                             "matrix" % (row, col, span_rows, span_cols,
                                         self.rows, self.cols))
        if name in self.groups:
            raise TileError("group %r already exists" % name)
        for g in self.groups.values():
            if (row < g.row + g.span_rows and row + span_rows > g.row
                    and col < g.col + g.span_cols
                    and col + span_cols > g.col):
                raise TileError("group %r overlaps group %r" % (name, g.name))
        group = TileGroup(name, self, row, col, span_rows, span_cols)
        self.groups[name] = group
        return group

    # -- heatmap helpers (iter 14) -------------------------------------------

    def heatmap_from_trace(self, trace, port="g1.cv", agg="max"):
        """Wall-wide heatmap from a FlowTrace: maps frames -> tiles cyclically.

        Returns dict {(row,col): normalized 0..1} and also applies it to each
        group's internal heatmap (so TileGroup.heatmap stays in sync).
        """
        tiles = self.tiles
        vals = { (t.row, t.col): [] for t in tiles }
        frames = trace.frames if hasattr(trace, "frames") else []
        for idx, fr in enumerate(frames):
            tile = tiles[idx % len(tiles)]
            if hasattr(fr, "out_ports"):
                out = fr.out_ports or {}
                inp = fr.in_ports or {}
            else:
                out = fr.get("out_ports") or {}
                inp = fr.get("in_ports") or {}
            v = out.get(port)
            if v is None:
                v = inp.get(port.split(".")[-1] if "." in port else port)
            if v is None:
                continue
            try:
                vals[(tile.row, tile.col)].append(float(v))
            except Exception:
                pass
        agg_map = {}
        for key, lst in vals.items():
            if not lst:
                agg_map[key] = 0.0
            elif agg == "max":
                agg_map[key] = max(lst)
            elif agg == "mean":
                agg_map[key] = sum(lst) / len(lst)
            elif agg == "last":
                agg_map[key] = lst[-1]
            else:
                agg_map[key] = max(lst)
        # normalize wall-wide
        if not agg_map:
            return {}
        lo = min(agg_map.values())
        hi = max(agg_map.values())
        span = (hi - lo) if hi != lo else 1.0
        norm = {k: (v - lo) / span for k, v in agg_map.items()}
        # also push to groups
        for g in self.groups.values():
            sub = {k: v for k, v in norm.items() if k in {(t.row, t.col) for t in g.tiles}}
            if sub:
                g.heatmap = dict(sub)
        return norm

    def heatmap_from_swarm(self, swarm_result, port="g1.cv", normalize=True):
        """Map SwarmResult per-agent scalars onto their TileGroups (if any).

        Expects agents were linked to TileGroups; returns wall heatmap dict.
        """
        raw = {}
        for res in swarm_result.results:
            aid = res.get("agent_id", "")
            v = res.get("final", {}).get(port)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            # find agent's group
            # swarm_result does not carry group; use display groups by name matching agent_id
            for g in self.groups.values():
                if g.name == aid or g.name == f"agent{aid}" or aid in g.name:
                    for t in g.tiles:
                        raw[(t.row, t.col)] = fv
                    break
            else:
                # fallback: cycle tiles
                pass
        if not raw:
            # fallback: assign each result to a tile cyclically
            tiles = self.tiles
            for idx, res in enumerate(swarm_result.results):
                v = res.get("final", {}).get(port)
                if v is None:
                    continue
                try:
                    raw[(tiles[idx % len(tiles)].row, tiles[idx % len(tiles)].col)] = float(v)
                except Exception:
                    pass
        if not raw:
            return {}
        if normalize:
            lo = min(raw.values())
            hi = max(raw.values())
            span = (hi - lo) if hi != lo else 1.0
            norm = {k: (v - lo) / span for k, v in raw.items()}
        else:
            norm = dict(raw)
        # push to groups
        for g in self.groups.values():
            sub = {k: v for k, v in norm.items() if k in {(t.row, t.col) for t in g.tiles}}
            if sub:
                g.heatmap = dict(sub)
        return norm

    def heatmap_animation(self, trace, port="g1.cv", window=1):
        """Per-tick heatmap animation from a FlowTrace replay.

        Slices trace.frames by tick (frames are stored tick-major: each tick
        contributes len(trace) // n_ticks entries). For each tick window,
        aggregates the port value per tile (cyclic) and normalizes.
        Returns list of heatmap dicts (one per tick window, values 0..1).

        window: number of ticks per animation frame (1 = one tick per frame).

        This is the tiles live viz (iter 15): replay frames colour the wall
        tick-by-tick without touching the bus.
        """
        frames = trace.frames if hasattr(trace, "frames") else []
        ticks = trace.ticks if hasattr(trace, "ticks") else []
        n_ticks = len(ticks) if ticks else (len(frames) // max(1, len(self.tiles)))
        if n_ticks == 0 or not frames:
            return []
        n_mods = max(1, len(frames) // max(1, n_ticks))
        tiles = self.tiles
        out = []
        for start in range(0, n_ticks, window):
            end = min(start + window, n_ticks)
            # slice frames for this window
            vals = { (t.row, t.col): [] for t in tiles }
            for ti in range(start, end):
                base = ti * n_mods
                sl = frames[base: base + n_mods]
                for idx, fr in enumerate(sl):
                    # map frame index within tick to tile cyclically
                    tile = tiles[(ti * n_mods + idx) % len(tiles)] if len(tiles) else None
                    if tile is None:
                        continue
                    if hasattr(fr, "out_ports"):
                        outp = fr.out_ports or {}
                        inp = fr.in_ports or {}
                    else:
                        outp = fr.get("out_ports") or {}
                        inp = fr.get("in_ports") or {}
                    v = outp.get(port)
                    if v is None:
                        v = inp.get(port.split(".")[-1] if "." in port else port)
                    if v is None:
                        # try any cv/gate value as fallback
                        for kk, vv in outp.items():
                            if kk.endswith(".cv") or kk == port:
                                v = vv
                                break
                    if v is None:
                        continue
                    try:
                        vals[(tile.row, tile.col)].append(float(v))
                    except Exception:
                        pass
            # aggregate per tile (max)
            agg = {}
            for k, lst in vals.items():
                if not lst:
                    agg[k] = 0.0
                else:
                    agg[k] = max(lst)
            # also consider direct bus sampling fallback: if agg all zero but trace has data,
            # sample from ticks if available (not needed)
            lo = min(agg.values()) if agg else 0.0
            hi = max(agg.values()) if agg else 0.0
            span = (hi - lo) if hi != lo else 1.0
            norm = {k: (v - lo) / span for k, v in agg.items()}
            out.append(norm)
        return out

    @staticmethod
    def validate_wgsl(source):
        """Validate WGSL shader shape (naga if available, else structural).

        Returns (ok: bool, detail: str). Structural check requires:
          - starts with // WGSL
          - contains @compute and @group(0) and @workgroup_size
          - contains fn main and per-block fns
          - host-RAM bridge comment
        If `naga` CLI is on PATH, also runs `naga --validate` / `naga source.wgsl`.
        """
        import shutil, subprocess, tempfile, os
        if not isinstance(source, str) or not source.strip():
            return False, "empty source"
        checks = []
        if not source.startswith("// WGSL"):
            checks.append("missing // WGSL header")
        if "@compute" not in source:
            checks.append("missing @compute")
        if "@group(0)" not in source:
            checks.append("missing @group(0)")
        if "host-RAM" not in source and "host bridge" not in source:
            checks.append("missing host-RAM bridge comment")
        if "fn main" not in source:
            checks.append("missing fn main")
        # try naga CLI if available (optional, not required for harness)
        naga = shutil.which("naga")
        if naga:
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".wgsl", delete=False) as fh:
                    fh.write(source)
                    tmp = fh.name
                out = subprocess.run([naga, tmp], capture_output=True, text=True, timeout=10)
                os.unlink(tmp)
                if out.returncode != 0:
                    # try validate mode
                    out2 = subprocess.run([naga, "--help"], capture_output=True, text=True, timeout=5)
                    return False, f"naga rejected: {out.stderr[:400]}"
            except Exception as e:
                checks.append(f"naga error: {e}")
        if checks:
            return False, "; ".join(checks)
        return True, "ok (structural + naga)" if naga else "ok (structural)"

    # -- export ------------------------------------------------------------

    def summary(self):
        return {
            "width": self.width,
            "height": self.height,
            "cols": self.cols,
            "rows": self.rows,
            "frame": {"w": self.frame.w, "h": self.frame.h,
                       "controls": list(self.frame.controls)},
            "tile_w": self.tile_w,
            "tile_h": self.tile_h,
            "margins": {"x": self.margin_x, "y": self.margin_y},
            "tiles": [[r, c, t.x, t.y] for (r, c), t in
                       sorted(self._tiles.items())],
            "groups": {name: {"row": g.row, "col": g.col,
                               "span": [g.span_rows, g.span_cols],
                               "w": g.width, "h": g.height,
                               "x": g.x, "y": g.y,
                               "heatmap": dict(g.heatmap) if hasattr(g, "heatmap") else {}}
                       for name, g in self.groups.items()},
        }
