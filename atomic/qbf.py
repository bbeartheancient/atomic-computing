"""QBF: the Quantum Blob Format -- goal 6, the portable trace store.

The "middle" container (the operator's iter-8 choice): it takes the
one good idea from each sibling format and drops their failure modes.

  .mv2  -- a real named-blob mechanism, but every shard is capped at
            a 50 MB tier (MV001) and one >~2.2 KB put poisons the
            whole shard's lex index (MV004, pinned live on this box).
  .tqbf -- no size cap (mmap'd, GB scale), but shaped like a
            model-weight file (a V4 model-metadata header, quantized
            tensors), Rust-only, and the H(4) Hadamard gate baked
            into its FOA tier instead of optional.

QBF keeps the blob as the unit of storage, with 64-bit offsets and
sizes and no tier, WAL or search index -- so neither the 50 MB wall
nor the lex poison can exist -- and makes the H(4) gate OPTIONAL: a
per-blob flag applied to a 4-channel numeric group, never forced
onto the data. Self-describing single file, no sidecars, pure Python.

Layout (all integers little-endian):

  header (64 bytes)
    0   u32  magic         0x00464251 (little-endian: disk bytes "QBF\0")
    4   u8   version       1
    5   u8   flags         0
    6   u16  reserved
    8   u32  num_blobs
    12  u32  index_offset  64
    16  u64  index_size
    24  u64  data_offset
    32  u64  data_size
    40  24 B reserved

  blob table (at index_offset): u32 count, then per blob
    u16 name_len | name bytes | u8 type | u8 flags |
    u64 offset (absolute) | u64 size | [32 B sha256 if checksum]

  blob data (at data_offset): the concatenated payloads

Blob types: RAW=0 arbitrary bytes; JSON=1 a UTF-8 JSON document;
H4=2 a 4-channel numeric group stored in the gated W/Z/Y/X domain.
Flags: 0x01 a sha256 follows the entry; 0x02 the H(4) gate was
applied to the payload.

The H(4) gate (the CORE keystone) splits a related 4-stream group
into 4 orthogonal streams: W=[+ + + +] amplitude/consensus (the
dominant row order is pinned by CORE), Z=[+ - + -], Y=[+ + - -],
X=[+ - - +]. The 4x4 Sylvester-Hadamard matrix is self-inverse up
to a factor of 4 (M.M = 4*I), so de-gating is the same matrix
scaled by 1/4. A H4 blob holds N groups of 4 float32 LE
(16 B/group); N = size/16.
"""

import hashlib
import json
import struct
from pathlib import Path

__all__ = ["QbfFile", "QbfError", "RAW", "JSON", "H4",
           "FLAG_CHECKSUM", "FLAG_H4",
           "h4_gate", "h4_inverse", "h4_encode", "h4_decode"]

MAGIC = 0x00464251      # packed little-endian -> file bytes 51 42 46 00 == "QBF\0"
VERSION = 1
HEADER_SIZE = 64
INDEX_OFFSET = 64

RAW, JSON, H4 = 0, 1, 2
FLAG_CHECKSUM = 0x01
FLAG_H4 = 0x02


class QbfError(ValueError):
    """A .qbf file (or one of its blobs) that cannot be written,
    parsed, or verified."""


# -- the H(4) gate: the CORE keystone, self-inverse up to a factor of 4

def h4_gate(t):
    """Gate one 4-tuple of related streams -> (W, Z, Y, X)."""
    a, b, c, d = (float(v) for v in t)
    return (a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d)


def h4_inverse(t):
    """De-gate (W, Z, Y, X) -> the original 4-tuple (M.M = 4*I)."""
    w, z, y, x = (float(v) for v in t)
    return ((w + z + y + x) / 4.0, (w - z + y - x) / 4.0,
            (w + z - y - x) / 4.0, (w - z - y + x) / 4.0)


def h4_encode(groups):
    """Gate a stream group (an iterable of 4-tuples) -> bytes
    (float32 LE, W Z Y X per group)."""
    pack = struct.Struct("<4f").pack
    return b"".join(pack(*h4_gate(g)) for g in groups)


def h4_decode(payload):
    """De-gate a h4_encode payload -> the original 4-tuples."""
    if len(payload) % 16:
        raise QbfError("H4 payload is not a multiple of a 16-byte group")
    unpack = struct.Struct("<4f").unpack
    mv = memoryview(payload)
    return [h4_inverse(unpack(mv[i * 16:(i + 1) * 16]))
            for i in range(len(payload) // 16)]


class QbfFile:
    """A self-describing container of named, lossless blobs.

    In-memory model: blobs live in a dict plus an insertion-order
    list; write() serializes header + table + data in two passes
    (the table is sized first so the absolute offsets it stores are
    correct) and rewrites the whole file. open() parses and
    verifies every checksummed blob -- a corrupted byte raises
    QbfError.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._blobs = {}
        self._order = []

    @classmethod
    def create(cls, path):
        """A fresh, empty QbfFile bound to path (not yet written)."""
        return cls(path)

    @classmethod
    def open(cls, path):
        """Parse (and verify) an existing .qbf file."""
        f = cls(path)
        f._load()
        return f

    # -- write side ------------------------------------------------------------

    def put(self, name, data, blob_type=RAW, checksum=True):
        if isinstance(data, str):
            data = data.encode("utf-8")
        if blob_type not in (RAW, JSON, H4):
            raise QbfError("unknown blob type %r" % (blob_type,))
        if len(name.encode("utf-8")) > 65535:
            raise QbfError("blob name too long for a QBF u16 length")
        flags = FLAG_CHECKSUM if checksum else 0
        if blob_type == H4:
            flags |= FLAG_H4
        self._blobs[name] = {
            "type": blob_type, "flags": flags, "data": bytes(data),
            "sha": hashlib.sha256(data).digest() if checksum else None,
        }
        if name not in self._order:
            self._order.append(name)
        return name

    def put_json(self, name, obj, checksum=True):
        """Store a JSON-serializable object as a lossless JSON blob."""
        return self.put(name, json.dumps(obj, sort_keys=True).encode("utf-8"),
                        JSON, checksum)

    def put_h4(self, name, groups, checksum=True):
        """Gate a 4-channel stream group and store it as an H4 blob."""
        return self.put(name, h4_encode(groups), H4, checksum)

    # -- read side ---------------------------------------------------------------

    def get(self, name):
        try:
            return self._blobs[name]["data"]
        except KeyError:
            raise QbfError("no such blob in %s: %r" % (self.path.name, name))

    def get_json(self, name):
        return json.loads(self.get(name))

    def get_h4(self, name):
        rec = self._blobs[name]
        if rec["type"] != H4:
            raise QbfError("%r is not a H4-gated blob" % name)
        return h4_decode(rec["data"])

    def info(self, name):
        rec = self._blobs[name]
        return {"type": rec["type"], "flags": rec["flags"],
                "size": len(rec["data"])}

    def names(self):
        return list(self._order)

    def __contains__(self, name):
        return name in self._blobs

    def __len__(self):
        return len(self._order)

    def __repr__(self):
        return "QbfFile(%r, %d blobs)" % (str(self.path), len(self._order))

    # -- serialize / parse ----------------------------------------------------------

    def write(self, path=None):
        """Serialize header + blob table + blob data; returns file size."""
        path = self.path if path is None else Path(path)
        count = len(self._order)
        index_size = 4                       # the u32 count
        plan = []
        for name in self._order:
            rec = self._blobs[name]
            name_b = name.encode("utf-8")
            sha = b"" if rec["sha"] is None else rec["sha"]
            index_size += 2 + len(name_b) + 2 + 16 + len(sha)
            plan.append((name_b, rec, sha))
        data_offset = INDEX_OFFSET + index_size
        data_size = sum(len(r["data"]) for r in self._blobs.values())
        header = bytearray(HEADER_SIZE)
        header[0:4] = struct.pack("<I", MAGIC)
        header[4] = VERSION
        header[8:12] = struct.pack("<I", count)
        header[12:16] = struct.pack("<I", INDEX_OFFSET)
        header[16:24] = struct.pack("<Q", index_size)
        header[24:32] = struct.pack("<Q", data_offset)
        header[32:40] = struct.pack("<Q", data_size)
        table = bytearray()
        table += struct.pack("<I", count)
        blob_data = []
        off = data_offset
        for name_b, rec, sha in plan:
            table += struct.pack("<H", len(name_b))
            table += name_b
            table += struct.pack("<BB", rec["type"], rec["flags"])
            table += struct.pack("<Q", off)
            table += struct.pack("<Q", len(rec["data"]))
            table += sha
            off += len(rec["data"])
            blob_data.append(rec["data"])
        path.write_bytes(bytes(header) + bytes(table) + b"".join(blob_data))
        return len(header) + len(table) + data_size

    def _load(self):
        if not self.path.exists():
            raise QbfError("no such .qbf file: %s" % self.path)
        buf = self.path.read_bytes()
        if len(buf) < HEADER_SIZE:
            raise QbfError("file too small to be a QBF container")
        if struct.unpack_from("<I", buf, 0)[0] != MAGIC:
            raise QbfError("bad magic (not a QBF file)")
        if buf[4] != VERSION:
            raise QbfError("unsupported QBF version %d" % buf[4])
        count, = struct.unpack_from("<I", buf, 8)
        index_offset, = struct.unpack_from("<I", buf, 12)
        index_size, = struct.unpack_from("<Q", buf, 16)
        data_offset, = struct.unpack_from("<Q", buf, 24)
        data_size, = struct.unpack_from("<Q", buf, 32)
        if index_offset != INDEX_OFFSET:
            raise QbfError("corrupt QBF header (index_offset)")
        if index_offset + index_size > data_offset or \
                data_offset + data_size > len(buf):
            raise QbfError("corrupt QBF region offsets")
        table = bytes(buf[index_offset:index_offset + index_size])
        if struct.unpack_from("<I", table, 0)[0] != count:
            raise QbfError("blob table count mismatch")
        pos = 4
        seen = set()
        for _ in range(count):
            (name_len,) = struct.unpack_from("<H", table, pos)
            pos += 2
            if pos + name_len + 20 > len(table):
                raise QbfError("corrupt QBF blob table (entry overrun)")
            name = table[pos:pos + name_len].decode("utf-8")
            pos += name_len
            blob_type, flags = struct.unpack_from("<BB", table, pos)
            pos += 2
            offset, size = struct.unpack_from("<QQ", table, pos)
            pos += 16
            sha = None
            if flags & FLAG_CHECKSUM:
                if pos + 32 > len(table):
                    raise QbfError("corrupt QBF blob table (sha overrun)")
                sha = bytes(table[pos:pos + 32])
                pos += 32
            if name in seen:
                raise QbfError("duplicate blob name %r" % name)
            seen.add(name)
            data = bytes(buf[offset:offset + size])
            if len(data) != size:
                raise QbfError("blob %r extends past end of file" % name)
            if sha is not None and hashlib.sha256(data).digest() != sha:
                raise QbfError("blob %r failed sha256 verification" % name)
            self._blobs[name] = {"type": blob_type, "flags": flags,
                                  "data": data, "sha": sha}
            self._order.append(name)
        if pos != len(table):
            raise QbfError("trailing bytes in the QBF blob table")
