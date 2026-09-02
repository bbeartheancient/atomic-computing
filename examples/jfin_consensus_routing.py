"""iter29 demo: Swarm H4-consensus -> JFinScheduler channel routing.

The atomic-computing analog of "infinite slop" channel rotation:
the Swarm's 4 agents each run a tiny atomic program (a const value),
the H(4) W-channel consensus over their scalars picks the next
program name from the prompt bank, and the JFinScheduler maps that
program to one of the Jellyfin Live TV channels.

Pipeline topology:
  Swarm (4 agents, 1 tile each)
      -> H4 consensus W
      -> PromptBank.consensus_pick(last_w)
      -> JFinScheduler.consensus_pick(programs, last_w)
      -> JFinExporter (HLS or DASH muxer, mock or real ffmpeg)
      -> JFinM3U recursive group-title emission
      -> Jellyfin Live TV ingest

The "channel rotation" itself uses three modes (round_robin, random,
h4_consensus) plus the new iter29 seeded_round_robin mode -- with
the same rotation_seed, two runs produce identical mappings (bit-
deterministic).
"""

import os
import shutil
import tempfile

import atomic
from atomic import (H3Stub, H3Session, Swarm, Agent, Program, Block, Wire,
                     Display, PromptBank)
from atomic.jellyfin import (JFinScheduler, JFinChannel, JFinM3U,
                              make_default_channels)


def main():
    print("="*60)
    print("iter29 demo: Swarm H4 -> JFinScheduler channel routing")
    print("="*60)

    tmp = tempfile.mkdtemp(prefix="jfin29_")
    print(f"temp dir: {tmp}")

    print("\n[1] Building 4-agent swarm (H4 keystone)")
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
    print(f"  Swarm.consensus('g1.cv') -> W = {w}")
    assert abs(w - 10.0) < 1e-9

    print("\n[2] PromptBank H4 W-channel pick")
    bank = PromptBank(prompts=[
        "neon city under heavy rain",
        "ancient library where books glow",
        "saturn rings dissolving into light",
        "macro beetle on wet bark",
    ])
    picked = bank.consensus_pick(last_w=w)
    print(f"  bank.consensus_pick(last_w=10) -> {picked!r}")

    print("\n[3] JFinScheduler 4 channels (mock ffmpeg)")
    programs = bank.prompts
    sched = JFinScheduler(rotation_seed=42)
    channels = make_default_channels(n=4, base_url="http://jellyfin.local:8096")
    for ch in channels:
        hls_dir = os.path.join(tmp, ch.id)
        sched.register_channel(ch, hls_dir=hls_dir, width=8, height=8,
                                mock=True)
        sched.assign_program(ch.id, programs[int(ch.id.split("-")[1]) - 1])
    print(f"  {len(sched.channels)} channels registered (mock=True)")

    print("\n[4] Consensus_pick: W={} -> {}".format(w, sched.consensus_pick(programs, last_w=w)))

    print("\n[5] rotate() modes (deterministic via seed)")
    for mode in ["round_robin", "random", "h4_consensus", "seeded_round_robin"]:
        sched.rotate(programs, mode=mode)
        m = sorted(sched.mappings.items())
        for cid, prog in m:
            print(f"    {mode:20s}  {cid} -> {prog}")

    print("\n[6] Seeded rotation determinism (seed=42, 2 schedulers)")
    s1 = JFinScheduler(rotation_seed=42)
    s2 = JFinScheduler(rotation_seed=42)
    for i in range(2):
        ch1 = JFinChannel(f"d{i}", f"D{i}", "http://x.m3u8")
        ch2 = JFinChannel(f"d{i}", f"D{i}", "http://x.m3u8")
        tmp1 = os.path.join(tmp, f"s1_{i}")
        tmp2 = os.path.join(tmp, f"s2_{i}")
        s1.register_channel(ch1, hls_dir=tmp1, width=4, height=4, mock=True)
        s2.register_channel(ch2, hls_dir=tmp2, width=4, height=4, mock=True)
    s1.rotate(programs, mode="random")
    s2.rotate(programs, mode="random")
    assert s1.mappings == s2.mappings, "random rotation not deterministic"
    print(f"  s1.mappings == s2.mappings: {s1.mappings}")
    s1.stop_all(); s2.stop_all()

    print("\n[7] DASH muxer + keyframe (mock mode)")
    ch = JFinChannel("dash1", "DASH1", "http://x.mpd")
    tmp_dash = os.path.join(tmp, "dash1")
    ex = sched.register_channel(ch, hls_dir=tmp_dash, width=4, height=4,
                                 muxer="dash", mock=True)
    assert ex.muxer == "dash"
    assert "live.mpd" in ex.playlist_path()
    frame = b"\x80\x40\x20\x10" * 4 * 4
    sched.push_frame(ch.id, frame, width=4, height=4, force_key=True)
    print(f"  DASH exporter: muxer={ex.muxer}, frame_count={ex.frame_count}, "
          f"keyframes={ex.keyframes}")

    print("\n[8] Recursive group-title M3U emission")
    m3u = JFinM3U(livetv_dir=tmp)
    for c in channels:
        m3u.add_channel(c)
    groups = m3u.write_recursive_groups()
    for g, path in groups.items():
        print(f"  {g}: {path}")

    print("\n[9] Cleanup")
    sched.stop_all()
    shutil.rmtree(tmp, ignore_errors=True)

    print("\nOK -- iter29 demo complete.")


if __name__ == "__main__":
    main()