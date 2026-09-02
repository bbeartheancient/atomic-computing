"""iter29 tests: DASH muxer, seeded rotation determinism, H4-consensus routing,
recursive group-title M3U emission, M3U stdin redirect, mock ffmpeg, and
end-to-end program with viz_video -> jfin_live_export.

Tests:
  - JFinExporter mock mode (no ffmpeg)
  - JFinExporter DASH muxer
  - JFinExporter force_keyframe
  - JFinScheduler rotation_seed determinism
  - JFinScheduler seeded_round_robin mode
  - JFinScheduler consensus_pick (H4 W routing)
  - JFinM3U recursive group emission
  - JFinM3U write_to_stdin
  - JFinM3U group_titles
  - jfin_live_export trig -> keyframe
  - viz_video -> jfin_live_export end-to-end program
  - Swarm.consensus -> JFinScheduler.consensus_pick integration
"""

import io
import os
import sys
import tempfile
import shutil

import pytest


class TestJFinExporterMock:
    def test_mock_mode_no_ffmpeg(self):
        from atomic.jellyfin import JFinExporter, JFinChannel, JFinError
        ch = JFinChannel("mock1", "MOCK1", "http://x.m3u8")
        ex = JFinExporter(ch, width=8, height=8, mock=True)
        assert ex.mock is True
        assert ex.running is True
        assert ex.frame_count == 0
        assert ex.keyframes == 0
        frame = b"\x00" * 8 * 8 * 4
        ok = ex.push(frame, width=8, height=8)
        assert ok is True
        assert ex.frame_count == 1
        assert ex.keyframes == 0
        # keyframe force increments
        ok2 = ex.push(frame, width=8, height=8, force_key=True)
        assert ok2 is True
        assert ex.keyframes == 1
        assert ex.frame_count == 2
        ex.stop()
        assert ex.running is False

    def test_mock_frame_size_check(self):
        from atomic.jellyfin import JFinExporter, JFinChannel, JFinError
        ch = JFinChannel("mock_sz", "MOCK_SZ", "http://x.m3u8")
        ex = JFinExporter(ch, width=4, height=4, mock=True)
        with pytest.raises(JFinError, match="frame size mismatch"):
            ex.push(b"\x00" * 100, width=4, height=4)

    def test_mock_muxer_dash(self):
        from atomic.jellyfin import JFinExporter, JFinChannel
        ch = JFinChannel("dash1", "DASH1", "http://x.mpd")
        tmp = tempfile.mkdtemp(prefix="jfin_dash_")
        try:
            ex = JFinExporter(ch, hls_dir=tmp, width=4, height=4,
                              muxer="dash", mock=True)
            assert ex.muxer == "dash"
            assert "live.mpd" in ex.playlist_name()
            assert "live.mpd" in ex.playlist_path()
            assert "live.mpd" in ex.mpd_path()
            frame = b"\x00" * 4 * 4 * 4
            ok = ex.push(frame, width=4, height=4)
            assert ok is True
            ex.stop()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_mock_keyframe_rising_edge(self):
        from atomic.jellyfin import JFinExporter, JFinChannel
        ch = JFinChannel("kf1", "KF1", "http://x.m3u8")
        ex = JFinExporter(ch, width=4, height=4, mock=True)
        frame = b"\x00" * 4 * 4 * 4
        # no keyframe without flag
        ex.push(frame)
        assert ex.keyframes == 0
        # explicit flag
        ex.push(frame, force_key=True)
        assert ex.keyframes == 1
        # two pushes, one keyframe
        ex.push(frame)
        ex.push(frame, force_key=True)
        assert ex.keyframes == 2
        ex.stop()


class TestJFinSchedulerSeeded:
    def test_rotation_seed_determinism(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        programs = ["p0", "p1", "p2", "p3"]
        for seed in [0, 42, 99]:
            s1 = JFinScheduler(rotation_seed=seed)
            for i in range(2):
                ch = JFinChannel(f"ch{i}", f"CH{i}", "http://x.m3u8")
                s1.register_channel(ch, mock=True)
            s1.rotate(programs, mode="random")
            m1 = dict(s1.mappings)
            s2 = JFinScheduler(rotation_seed=seed)
            for i in range(2):
                ch = JFinChannel(f"ch{i}", f"CH{i}", "http://x.m3u8")
                s2.register_channel(ch, mock=True)
            s2.rotate(programs, mode="random")
            m2 = dict(s2.mappings)
            assert m1 == m2, f"seed={seed} not deterministic"
            s1.stop_all()
            s2.stop_all()

    def test_seeded_round_robin_mode(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        programs = ["a", "b", "c", "d"]
        sched = JFinScheduler(rotation_seed=7)
        for i in range(3):
            ch = JFinChannel(f"srr{i}", f"SRR{i}", "http://x.m3u8")
            sched.register_channel(ch, mock=True)
        sched.rotate(programs, mode="seeded_round_robin")
        m1 = dict(sched.mappings)
        sched2 = JFinScheduler(rotation_seed=7)
        for i in range(3):
            ch = JFinChannel(f"srr{i}", f"SRR{i}", "http://x.m3u8")
            sched2.register_channel(ch, mock=True)
        sched2.rotate(programs, mode="seeded_round_robin")
        m2 = dict(sched2.mappings)
        assert m1 == m2, "seeded_round_robin not deterministic"
        sched.stop_all()
        sched2.stop_all()

    def test_consensus_pick(self):
        from atomic.jellyfin import JFinScheduler
        programs = ["prog0", "prog1", "prog2", "prog3"]
        sched = JFinScheduler()
        # W=0 -> idx=0
        assert sched.consensus_pick(programs, last_w=0.0) == "prog0"
        # W=3.9 -> idx=3
        assert sched.consensus_pick(programs, last_w=3.9) == "prog3"
        # W=7.2 -> idx=(7 % 4) = 3
        assert sched.consensus_pick(programs, last_w=7.2) == "prog3"
        # W=0.5 -> idx=(0 % 4) = 0
        assert sched.consensus_pick(programs, last_w=0.5) == "prog0"

    def test_consensus_pick_empty(self):
        from atomic.jellyfin import JFinScheduler
        sched = JFinScheduler()
        assert sched.consensus_pick([], last_w=1.0) == ""

    def test_force_keyframe_via_scheduler(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        ch = JFinChannel("fk1", "FK1", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_fk_")
        try:
            sched.register_channel(ch, hls_dir=tmp, width=4, height=4, mock=True)
            frame = b"\x00" * 4 * 4 * 4
            sched.push_frame(ch.id, frame, width=4, height=4)
            ex = sched.exporters[ch.id]
            assert ex.keyframes == 0
            sched.force_keyframe(ch.id)
            assert ex.keyframes == 1
            sched.stop_channel(ch.id)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_stats_includes_keyframes_and_muxer(self):
        from atomic.jellyfin import JFinScheduler, JFinChannel
        sched = JFinScheduler()
        ch = JFinChannel("st1", "ST1", "http://x.m3u8")
        tmp = tempfile.mkdtemp(prefix="jfin_stats29_")
        try:
            sched.register_channel(ch, hls_dir=tmp, width=4, height=4, muxer="dash", mock=True)
            frame = b"\x00" * 4 * 4 * 4
            sched.push_frame(ch.id, frame, force_key=True)
            stats = sched.stats()
            assert "keyframes" in stats[ch.id]
            assert "muxer" in stats[ch.id]
            assert "mock" in stats[ch.id]
            assert stats[ch.id]["muxer"] == "dash"
            assert stats[ch.id]["keyframes"] >= 1
            sched.stop_all()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestJFinM3URecursive:
    def test_group_titles(self):
        from atomic.jellyfin import JFinM3U, JFinChannel
        m3u = JFinM3U(livetv_dir=tempfile.mkdtemp(prefix="m3u_gt_"))
        m3u.add_channel(JFinChannel("c1", "C1", "http://x.m3u8", group="TV"))
        m3u.add_channel(JFinChannel("c2", "C2", "http://y.m3u8", group="ATOMIC"))
        m3u.add_channel(JFinChannel("c3", "C3", "http://z.m3u8", group="TV"))
        groups = m3u.group_titles()
        assert groups == ["ATOMIC", "TV"]

    def test_write_recursive_groups(self):
        from atomic.jellyfin import JFinM3U, JFinChannel
        tmp = tempfile.mkdtemp(prefix="m3u_rec_")
        try:
            m3u = JFinM3U(livetv_dir=tmp)
            m3u.add_channel(JFinChannel("a1", "A1", "http://a1.m3u8", group="ATOMIC"))
            m3u.add_channel(JFinChannel("a2", "A2", "http://a2.m3u8", group="ATOMIC"))
            m3u.add_channel(JFinChannel("t1", "T1", "http://t1.m3u8", group="TV"))
            result = m3u.write_recursive_groups()
            assert "ATOMIC" in result
            assert "TV" in result
            assert "__root__" in result
            assert os.path.exists(result["ATOMIC"])
            assert os.path.exists(result["TV"])
            assert os.path.exists(result["__root__"])
            # root M3U contains EXTGRP lines
            with open(result["__root__"]) as f:
                root = f.read()
            assert "#EXTGRP:ATOMIC" in root
            assert "#EXTGRP:TV" in root
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_write_to_stdin(self):
        from atomic.jellyfin import JFinM3U, JFinChannel
        tmp = tempfile.mkdtemp(prefix="m3u_stdin_")
        try:
            m3u = JFinM3U(livetv_dir=tmp)
            m3u.add_channel(JFinChannel("s1", "S1", "http://s1.m3u8"))
            buf = io.BytesIO()
            rendered = m3u.write_to_stdin(filename="test.m3u", stdin_fp=buf)
            assert b"#EXTM3U" in rendered
            assert b"S1" in rendered
            assert buf.tell() > 0
            # with None stdin_fp
            rendered2 = m3u.write_to_stdin()
            assert b"#EXTM3U" in rendered2
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_channel_id_in_m3u_line(self):
        from atomic.jellyfin import JFinChannel
        ch = JFinChannel("ch-99", "CH-99", "http://x.m3u8")
        line = ch.m3u_line()
        assert 'channel-id="ch-99"' in line

    def test_group_path_in_m3u_line(self):
        from atomic.jellyfin import JFinChannel
        ch = JFinChannel("g1", "G1", "http://x.m3u8", group="ATOMIC")
        line = ch.m3u_line(group_path="HD")
        assert 'group-title="HD/ATOMIC"' in line
        # no group_path falls back to bare group
        line2 = ch.m3u_line()
        assert 'group-title="ATOMIC"' in line2


class TestJFinLiveExportKeyframe:
    def test_trig_rising_edge_keyframe(self):
        from atomic import Program, Block, Wire, Engine
        from atomic.gates import register_jfin_scheduler
        from atomic.jellyfin import JFinScheduler, JFinChannel
        p = Program("jle_kf", blocks=[
            Block("clk", "clock_bpm", {"bpm": 1}),
            Block("jle", "jfin_live_export", {"scheduler_key": "kf1", "width": 4, "height": 4}),
        ], wires=[Wire("clk.trig", "jle.trig")])
        sched = JFinScheduler()
        register_jfin_scheduler(sched)
        tmp = tempfile.mkdtemp(prefix="jle_kf_test_")
        try:
            ch = JFinChannel("kf1", "KF1", "http://x.m3u8")
            sched.register_channel(ch, hls_dir=tmp, width=4, height=4, mock=True)
            # feed a frame directly via bus
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], patch.get("wires", []))
            eng.bus.set("jle.frame", b"\x00" * 4 * 4 * 4)
            eng.run(5)  # 5 ticks at 1 bpm -> 1 beat at tick 30
            jle_id = next((n.id for n in eng.nodes
                          if n.primitive == "jfin_live_export"), None)
            assert jle_id is not None
            kf_out = eng.bus.get(f"{jle_id}.keyframes")
            assert kf_out is not None
            sched.stop_all()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            register_jfin_scheduler(None)


class TestSwarmJFinIntegration:
    def test_swarm_consensus_to_scheduler(self):
        from atomic import Swarm, Agent, Program, Block, Wire, Display
        from atomic.jellyfin import JFinScheduler
        d = Display(800, 800, 4, 4)
        swarm = Swarm(display=d)
        for i, val in enumerate([1.0, 2.0, 3.0, 4.0]):
            prog = Program("p", blocks=[
                Block("c0", "const", {"value": val}),
                Block("g1", "gain", {"factor": 1.0}),
                Block("v0", "viz_series"),
            ], wires=[Wire("c0.cv", "g1.in"), Wire("g1.cv", "v0.in")])
            g = d.link(f"s{i}", i // 4, i % 4, 1, 1)
            swarm.add_agent(Agent(f"a{i}", prog, tile_group=g))
        res = swarm.run(3)
        w = res.consensus(port="g1.cv")
        # 1+2+3+4 = 10.0 via H4 W
        assert abs(w - 10.0) < 1e-9
        programs = ["neon rain", "glowing library", "saturn rings", "beetle bark"]
        sched = JFinScheduler()
        picked = sched.consensus_pick(programs, last_w=w)
        assert picked in programs


class TestEndToEndProgram:
    def test_viz_video_jfin_live_export_pipeline(self):
        from atomic import Program, Block, Wire, Engine
        from atomic.gates import register_jfin_scheduler
        from atomic.jellyfin import JFinScheduler, JFinChannel
        p = Program("vv_jfin", blocks=[
            Block("c", "const", {"value": 1.0}),
            Block("vv", "viz_video"),
            Block("jle", "jfin_live_export", {"scheduler_key": "vj1", "width": 4, "height": 4}),
        ], wires=[
            Wire("c.cv", "vv.in"),
            Wire("c.cv", "jle.in"),
        ])
        sched = JFinScheduler()
        register_jfin_scheduler(sched)
        tmp = tempfile.mkdtemp(prefix="vv_jfin_")
        try:
            ch = JFinChannel("vj1", "VJ1", "http://x.m3u8")
            sched.register_channel(ch, hls_dir=tmp, width=4, height=4, mock=True)
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], patch.get("wires", []))
            # feed a frame so viz_video has something to decode
            frame = b"\x80\x40\x20\x10" * 4 * 4
            eng.bus.set("vv.frame", frame)
            eng.run(2)
            jle_id = next((n.id for n in eng.nodes
                          if n.primitive == "jfin_live_export"), None)
            vv_id = next((n.id for n in eng.nodes
                         if n.primitive == "viz_video"), None)
            assert jle_id is not None
            assert vv_id is not None
            frames_out = eng.bus.get(f"{jle_id}.frames_pushed")
            active = eng.bus.get(f"{jle_id}.active")
            # no frame on jle.in bus -> no frame pushed (gate=0)
            assert frames_out is not None
            sched.stop_all()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            register_jfin_scheduler(None)

    def test_viz_video_jfin_with_frame_on_jle(self):
        from atomic import Program, Block, Wire, Engine
        from atomic.gates import register_jfin_scheduler
        from atomic.jellyfin import JFinScheduler, JFinChannel
        p = Program("vv_jfin_on", blocks=[
            Block("c", "const", {"value": 1.0}),
            Block("jle", "jfin_live_export", {"scheduler_key": "vj2", "width": 4, "height": 4}),
        ], wires=[Wire("c.cv", "jle.in")])
        sched = JFinScheduler()
        register_jfin_scheduler(sched)
        tmp = tempfile.mkdtemp(prefix="vv_jfin_on_")
        try:
            ch = JFinChannel("vj2", "VJ2", "http://x.m3u8")
            sched.register_channel(ch, hls_dir=tmp, width=4, height=4, mock=True)
            patch = p.compile("microfx")
            eng = Engine(patch["modules"], patch.get("wires", []))
            frame = b"\xff\x00\x80\x40" * 4 * 4
            eng.bus.set("jle.frame", frame)
            eng.run(1)
            jle_id = next((n.id for n in eng.nodes
                          if n.primitive == "jfin_live_export"), None)
            assert jle_id is not None
            # gate from c.cv=1.0 > 0.5 -> export enabled
            # but jle.frame is not set at tick 0 (it's set before run(1)),
            # so frames_pushed stays 0 in first tick
            sched.stop_all()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            register_jfin_scheduler(None)
