"""iter28 tests: Jellyfin/HDHomeRun integration.

Tests:
  - JFinChannel M3U line format
  - JFinExporter lifecycle (no ffmpeg: size-check + graceful failure)
  - JFinM3U write/read round-trip
  - JFinScheduler channel registration + frame push
  - jfin_live_export atom registered + tick with/without gate
  - M3U HDHomeRun discovery stub
  - Default channels factory
  - M3U playlist URL generation
"""

import os
import sys
import tempfile
import shutil

import pytest


class TestJFinChannel:
    def test_m3u_line_format(self):
        from atomic.jellyfin import JFinChannel
        ch = JFinChannel(
            id="atomic-01", name="ATOMIC-1",
            m3u_url="http://localhost:8080/livetv/atomic-01/live.m3u8",
            group="ATOMIC", number=1
        )
        line = ch.m3u_line()
        assert 'tvg-id="atomic-01"' in line
        assert 'tvg-name="ATOMIC-1"' in line
        assert 'group-title="ATOMIC"' in line
        assert "ATOMIC-1\n" in line
        assert "http://localhost:8080" in line

    def test_channel_repr(self):
        from atomic.jellyfin import JFinChannel
        ch = JFinChannel("c1", "CH-1", "http://x/y.m3u8")
        assert "atomic-01" in repr(ch) or "c1" in repr(ch)

    def test_make_default_channels(self):
        from atomic.jellyfin import make_default_channels
        channels = make_default_channels(n=3)
        assert len(channels) == 3
        assert channels[0].number == 1
        assert channels[1].number == 2
        assert channels[2].number == 3
        assert channels[0].id == "atomic-01"
        assert channels[2].id == "atomic-03"


class TestJFinExporter:
    def test_exporter_init_with_ffmpeg(self):
        from atomic.jellyfin import JFinExporter, JFinChannel
        ch = JFinChannel("test-ch", "TEST", "http://x/y.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_test_")
        try:
            ex = JFinExporter(ch, hls_dir=tmp, width=64, height=64)
            assert ex.running is True
            assert ex.channel.id == "test-ch"
            ex.stop()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_frame_size_mismatch(self):
        from atomic.jellyfin import JFinExporter, JFinChannel, JFinError
        ch = JFinChannel("test-ch", "TEST", "http://x/y.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_test_")
        try:
            ex = JFinExporter(ch, hls_dir=tmp, width=8, height=8)
            with pytest.raises(JFinError, match="frame size mismatch"):
                ex.push(b"\x00" * 100, width=8, height=8)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_playlist_path(self):
        from atomic.jellyfin import JFinExporter, JFinChannel
        ch = JFinChannel("ch1", "C1", "http://x/y")
        tmp = tempfile.mkdtemp(prefix="jfin_test_")
        try:
            ex = JFinExporter(ch, hls_dir=tmp, width=64, height=64)
            assert "live.m3u8" in ex.playlist_path()
            assert tmp in ex.playlist_path()
            ex.stop()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_push_one_frame(self):
        from atomic.jellyfin import JFinExporter, JFinChannel
        ch = JFinChannel("push-ch", "PUSH", "http://x/y.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_push_test_")
        try:
            ex = JFinExporter(ch, hls_dir=tmp, width=4, height=4)
            ok = ex.push(b"\x80\x40\x20\x10" * 4 * 4, width=4, height=4)
            assert ok is True
            assert ex.frame_count == 1
            ex.stop()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestJFinM3U:
    def test_write_read_roundtrip(self):
        from atomic.jellyfin import JFinM3U, JFinChannel
        ch1 = JFinChannel("a1", "ATOMIC-1", "http://a1.m3u8", group="TV", number=1)
        ch2 = JFinChannel("a2", "ATOMIC-2", "http://a2.m3u8", group="TV", number=2)
        tmp = tempfile.mkdtemp(prefix="jfin_m3u_test_")
        try:
            m3u = JFinM3U(livetv_dir=tmp)
            m3u.add_channel(ch1)
            m3u.add_channel(ch2)
            path = m3u.write("atomic.m3u")
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "#EXTM3U" in content
            assert "ATOMIC-1" in content
            assert "ATOMIC-2" in content
            assert "http://a1.m3u8" in content
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_write_all_one_per_channel(self):
        from atomic.jellyfin import JFinM3U, JFinChannel
        channels = [
            JFinChannel("c1", "CH1", "http://c1.m3u8"),
            JFinChannel("c2", "CH2", "http://c2.m3u8"),
        ]
        tmp = tempfile.mkdtemp(prefix="jfin_m3u_test2_")
        try:
            m3u = JFinM3U(livetv_dir=tmp)
            for ch in channels:
                m3u.add_channel(ch)
            paths = m3u.write_all()
            assert len(paths) == 2
            assert all(os.path.exists(p) for p in paths)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_discover_hdhr_returns_list(self):
        from atomic.jellyfin import JFinM3U
        # UDP broadcast may return empty list on no devices (fine)
        result = JFinM3U.discover_hdhr()
        assert isinstance(result, list)


class TestJFinScheduler:
    def test_register_and_push(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel, JFinExporter
        sched = JFinScheduler()
        ch = JFinChannel("sch1", "SCHED-1", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_sched_test_")
        try:
            ex = sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
            assert ch.id in sched.channels
            assert ch.id in sched.exporters
            sched.assign_program(ch.id, "test_program")
            assert sched.mappings[ch.id] == "test_program"
            # ffmpeg IS available, so push returns True on success
            ok = sched.push_frame(ch.id, b"\x00" * 8 * 8 * 4, width=8, height=8)
            assert ok is True
            sched.stop_channel(ch.id)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rotate_round_robin(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        for i in range(3):
            ch = JFinChannel(f"rr{i}", f"RR{i}", "http://x.m3u8")
            tmp = tempfile.mkdtemp(prefix=f"jfin_rr_{i}_")
            sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
        programs = ["p0", "p1", "p2", "p3"]
        try:
            sched.rotate(programs, mode="round_robin")
            m1 = dict(sched.mappings)
            sched.rotate(programs, mode="round_robin")
            m2 = dict(sched.mappings)
            # round robin advances; assignments should differ
            assert m1 != m2
            # every assignment is a valid program
            for v in m1.values():
                assert v in programs
        finally:
            sched.stop_all()

    def test_rotate_random(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        ch = JFinChannel("rn1", "RN1", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_rn_test_")
        try:
            sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
            programs = ["alpha", "bravo", "charlie", "delta"]
            sched.rotate(programs, mode="random")
            assert sched.mappings["rn1"] in programs
            sched.rotate(programs, mode="random")
            # with seed=0 first call; both rotations should be valid programs
            assert sched.mappings["rn1"] in programs
        finally:
            sched.stop_all()

    def test_rotate_h4_consensus(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        for i in range(2):
            ch = JFinChannel(f"h4{i}", f"H4{i}", "http://x.m3u8")
            tmp = tempfile.mkdtemp(prefix=f"jfin_h4_{i}_")
            sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
        programs = ["x", "y", "z", "w"]
        try:
            sched.rotate(programs, mode="h4_consensus")
            for v in sched.mappings.values():
                assert v in programs
        finally:
            sched.stop_all()

    def test_rotate_empty_inputs(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        ch = JFinChannel("em1", "EM1", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_em_test_")
        try:
            sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
            sched.rotate([], mode="round_robin")
            assert "em1" not in sched.mappings
        finally:
            sched.stop_all()

    def test_stop_channel(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        ch = JFinChannel("sc2", "SC2", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_sched_test2_")
        try:
            ex = sched.register_channel(ch, hls_dir=tmp)
            sched.stop_channel(ch.id)
            assert ch.id not in sched.exporters
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_stop_all(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        for i in range(2):
            ch = JFinChannel(f"sc{i}", f"SC{i}", "http://x.m3u8")
            tmp = tempfile.mkdtemp(prefix="jfin_sched_test3_")
            sched.register_channel(ch, hls_dir=tmp)
        sched.stop_all()
        assert len(sched.exporters) == 0

    def test_stats(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        ch = JFinChannel("st1", "ST1", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_stats_test_")
        try:
            sched.register_channel(ch, hls_dir=tmp)
            stats = sched.stats()
            assert "st1" in stats
            assert "running" in stats["st1"]
            assert "frame_count" in stats["st1"]
            assert "playlist" in stats["st1"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestJFinLiveExportAtom:
    def test_atom_registered(self):
        from atomic import ATOMS
        assert "jfin_live_export" in ATOMS
        a = ATOMS["jfin_live_export"]
        assert a.category == "sink"
        assert "in" in a.inputs
        assert "trig" in a.inputs
        assert "active" in a.outputs
        assert "frames_pushed" in a.outputs

    def test_atom_tick_no_ffmpeg(self):
        from atomic import Program, Block, Wire, Engine
        from atomic.gates import register_jfin_scheduler
        from atomic.jellyfin import JFinScheduler, JFinChannel

        p = Program("jle_test", blocks=[
            Block("c", "const", {"value": 1.0}),
            Block("jle", "jfin_live_export", {"scheduler_key": "ch1", "width": 8, "height": 8}),
        ], wires=[Wire("c.cv", "jle.in")])

        sched = JFinScheduler()
        register_jfin_scheduler(sched)

        tmp = tempfile.mkdtemp(prefix="jfin_atom_test_")
        try:
            ch = JFinChannel("ch1", "CH1", "http://x.m3u8")
            sched.register_channel(ch, hls_dir=tmp, width=8, height=8)
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], patch.get("wires", []))
            eng.run(2)
            jle_id = next((n.id for n in eng.nodes
                          if n.primitive == "jfin_live_export"), None)
            assert jle_id is not None
            assert eng.bus.get(f"{jle_id}.active") is not None
            assert eng.bus.get(f"{jle_id}.frames_pushed") is not None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            register_jfin_scheduler(None)
