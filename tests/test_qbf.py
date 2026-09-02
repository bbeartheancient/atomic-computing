"""QBF (Quantum Blob Format): the "middle" container for goal 6.

Pinned contracts:

  1. the header is self-describing (magic "QBF\0", v1, 64 bytes) and
     write -> open round-trips every blob byte-exactly
  2. there is NO 50 MB tier cap: a 62 MB blob crosses the .mv2 MV001
     boundary and still round-trips (QBF sizes/offsets are u64)
  3. there is no search index to poison: a >2.2 KB payload -- the
     size that bricks a .mv2 shard's lex index (MV004) -- is
     unremarkable here
  4. the per-blob sha256 detects a corrupted payload byte on open
  5. the H(4) gate is OPTIONAL and exact: W=[+ + + +], Z=[+ - + -],
     Y=[+ + - -], X=[+ - - +]; M.M = 4*I, so the inverse is M/4
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomic import QbfError, QbfFile, h4_gate, h4_inverse


def test_write_open_roundtrip(tmp_path):
    f = QbfFile.create(tmp_path / "a.qbf")
    f.put("hello", b"world")
    f.put_json("cfg", {"a": 1, "b": [1, 2, 3]})
    size = f.write()
    raw = f.path.read_bytes()
    assert raw[:4] == b"QBF\x00"
    assert raw[4] == 1
    assert size == len(raw)
    g = QbfFile.open(f.path)
    assert g.get("hello") == b"world"
    assert g.get_json("cfg") == {"a": 1, "b": [1, 2, 3]}
    assert g.names() == ["hello", "cfg"]


def test_no_mv2_limits(tmp_path):
    poison = bytes(range(251)) * 13       # 3263 B: over the ~2.2 KB poison line
    big = bytes(range(251)) * 250_000     # 62.75 MB: over the .mv2 50 MB tier
    f = QbfFile.create(tmp_path / "big.qbf")
    f.put("poison", poison)
    f.put("wall", big)
    f.write()
    g = QbfFile.open(f.path)
    assert g.get("poison") == poison
    assert g.get("wall") == big
    assert g.info("wall")["size"] == len(big)


def test_checksum_detects_corruption(tmp_path):
    f = QbfFile.create(tmp_path / "c.qbf")
    f.put("doc", b"abcdef")
    f.write()
    buf = bytearray(f.path.read_bytes())
    off = struct.unpack_from("<Q", bytes(buf), 24)[0]
    buf[off] ^= 0xFF                        # flip one payload byte
    f.path.write_bytes(bytes(buf))
    try:
        QbfFile.open(f.path)
        assert False, "a corrupted payload must raise"
    except QbfError:
        pass


def test_missing_blob_raises(tmp_path):
    f = QbfFile.create(tmp_path / "m.qbf")
    f.put("a", b"1")
    f.write()
    try:
        QbfFile.open(f.path).get("nope")
        assert False
    except QbfError:
        pass


def test_h4_gate_math():
    w, z, y, x = h4_gate((1, 2, 3, 4))
    assert (w, z, y, x) == (10.0, -2.0, -4.0, 0.0)
    back = h4_inverse((w, z, y, x))
    assert all(abs(a - b) < 1e-9 for a, b in zip(back, (1, 2, 3, 4)))


def test_h4_blob_roundtrip(tmp_path):
    groups = [(1.0, 2.0, 3.0, 4.0), (0.5, -1.0, 2.5, 0.0),
              (-3.0, 0.25, 1.5, -0.75)]
    f = QbfFile.create(tmp_path / "h4.qbf")
    f.put_h4("wzx", groups)
    f.write()
    g = QbfFile.open(f.path)
    assert g.info("wzx")["type"] == 2      # the H4 type flag is recorded
    back = g.get_h4("wzx")
    assert len(back) == 3
    assert all(abs(a - b) < 1e-6
               for ga, gb in zip(groups, back) for a, b in zip(ga, gb))
