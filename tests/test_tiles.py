"""Tiles (build-order step 6): the display model.

Per ATOMIC-PC-CORE.md "DISPLAY TILES": a control frame (top-level
i/o) over a 3x3 or 4x4 matrix of universal agnostic framebuffer
tiles; the FULL display resolution determines the tile resolution
(tile_w = W // cols, tile_h = (H - frame_h) // rows; leftover px =
dead border, the wall's seams); linked tile groups form larger
sub-matrix displays (k*tile_w x m*tile_h at the group's top-left
tile) -- a tiled video wall.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic import Display, TileError  # noqa: E402


def test_tile_resolution_3x3():
    d = Display(1920, 1080, 3, 3, frame_h=120)
    assert d.tile_w == 1920 // 3            # 640
    assert d.tile_h == (1080 - 120) // 3    # 320
    assert d.grid_width == 1920
    assert d.grid_height == 960
    assert d.margin_x == 0
    assert d.margin_y == 0


def test_leftover_pixels_are_dead_border():
    d = Display(3841, 2167, 4, 4, frame_h=64)
    assert d.tile_w == 960                   # 3841 // 4
    assert d.tile_h == 525                   # (2167-64) // 4 = 2103 // 4
    assert d.margin_x == 1                    # 3841 - 4*960
    assert d.margin_y == 3                    # 2103 - 4*525
    assert d.grid_width + d.margin_x == d.width
    assert d.frame.h + d.grid_height + d.margin_y == d.height


def test_control_frame_region():
    d = Display(1000, 800, 3, 3, frame_h=100)
    assert d.frame.bounds() == (0, 0, 1000, 100)
    assert d.tile(0, 0).y == 100
    assert d.tile(2, 0).y == 100 + 2 * d.tile_h


def test_tile_offsets_tiled_wall():
    d = Display(900, 900, 3, 3)
    ts = d.tiles
    assert len(ts) == 9
    for t in ts:
        assert t.bounds() == (t.col * 300, t.row * 300, 300, 300)
    assert [t.bounds() for t in ts] == [
        (c * 300, r * 300, 300, 300) for r in range(3) for c in range(3)]


def test_matrix_size_rejected():
    for cols, rows in [(2, 3), (3, 2), (5, 4), (4, 3), (3, 5), (4, 4)]:
        if (cols, rows) == (4, 4):
            continue
        with pytest.raises(TileError):
            Display(100, 100, cols, rows)
    Display(100, 100, 4, 4)  # 4x4 is the other legal size


def test_resolution_validation():
    with pytest.raises(TileError):
        Display(0, 100, 3, 3)
    with pytest.raises(TileError):
        Display(100, -5, 3, 3)
    with pytest.raises(TileError):
        Display(100, 100, 3, 3, frame_h=-1)
    with pytest.raises(TileError):
        Display(100, 100, 3, 3, frame_h=100)   # frame >= height
    with pytest.raises(TileError):
        Display(2, 100, 3, 3)                   # tile_w = 0
    with pytest.raises(TileError):
        Display(100, 2, 3, 3)                   # tile_h = 0


def test_controls():
    d = Display(100, 100, 3, 3, 40,
                controls=["bpm", ("amp", "slider")])
    assert d.frame.controls == [
        {"name": "bpm", "kind": "param"},
        {"name": "amp", "kind": "slider"}]
    d.frame.add_control("gain")
    assert d.frame.controls[2] == {"name": "gain", "kind": "param"}


def test_link_group_submatrix():
    d = Display(1000, 1000, 4, 4, frame_h=100)
    g = d.link("wall", 1, 1, 2, 2)
    assert g.width == 2 * d.tile_w
    assert g.height == 2 * d.tile_h
    assert g.x == d.tile(1, 1).x
    assert g.y == d.tile(1, 1).y
    assert len(g.tiles) == 4
    assert g.bounds() == (g.x, g.y, g.width, g.height)


def test_link_out_of_bounds_rejected():
    d = Display(100, 100, 3, 3)
    with pytest.raises(TileError):
        d.link("a", 2, 0, 2, 1)    # row 2 + span 2 > 3
    with pytest.raises(TileError):
        d.link("b", 0, 2, 1, 2)    # col 2 + span 2 > 3
    with pytest.raises(TileError):
        d.link("c", -1, 0, 1, 1)
    with pytest.raises(TileError):
        d.link("d", 0, 0, 0, 1)    # span must be >= 1x1


def test_link_overlap_and_dup_rejected():
    d = Display(100, 100, 4, 4)
    d.link("a", 0, 0, 2, 2)
    with pytest.raises(TileError):
        d.link("b", 1, 1, 2, 2)    # overlaps "a" at (1,1)
    with pytest.raises(TileError):
        d.link("a", 3, 3, 1, 1)    # duplicate name
    d.link("c", 3, 3, 1, 1)        # diagonal corner: no overlap


def test_link_full_matrix():
    d = Display(100, 100, 3, 3, 10)
    g = d.link("all", 0, 0, 3, 3)
    assert (g.width, g.height) == (d.grid_width, d.grid_height)
    assert (g.x, g.y) == (0, 10)


def test_tile_lookup_out_of_range():
    d = Display(100, 100, 3, 3)
    with pytest.raises(TileError):
        d.tile(3, 0)
    with pytest.raises(TileError):
        d.tile(0, -1)
    assert d.tile(0, 2).row == 0 and d.tile(0, 2).col == 2


def test_summary_shape():
    d = Display(1000, 800, 4, 4, 64, controls=["bpm"])
    d.link("wall", 0, 0, 2, 2)
    s = d.summary()
    assert s["cols"] == 4 and s["rows"] == 4
    assert s["frame"] == {"w": 1000, "h": 64,
                          "controls": [{"name": "bpm", "kind": "param"}]}
    assert s["tile_w"] == 250
    assert s["tile_h"] == (800 - 64) // 4
    assert len(s["tiles"]) == 16
    assert s["groups"]["wall"]["span"] == [2, 2]
    assert s["groups"]["wall"]["w"] == 2 * d.tile_w
