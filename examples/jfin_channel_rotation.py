"""iter28 demo: Jellyfin/HDHomeRun channel rotation.

Builds a fleet of 4 H3 sessions (each its own infinite-livestream),
registers 4 Jellyfin Live TV channels, rotates program-to-channel
mappings (round-robin, random, h4_consensus), and writes M3U tuner
files to a temp directory (substitute /etc/jellyfin/livetv in prod).

Pipeline topology:
  H3 fleet (GPU1, stubbed) -> JFinScheduler -> ffmpeg HLS muxer
                                          -> M3U tuner emission
                                          -> Jellyfin Live TV

This example runs end-to-end. With real GPU1 H3 in production, swap
H3Stub for H3Client(endpoint='http://gpu1.local:8765') and the
hls_dir to /var/lib/jellyfin/livetv.
"""

import os
import shutil
import sys
import tempfile

import atomic
from atomic import (H3Stub, H3Session, JFinScheduler, JFinChannel,
                     JFinM3U, make_default_channels)


def main():
    print("="*60)
    print("Jellyfin/HDHomeRun channel rotation demo (iter 28)")
    print("="*60)

    tmp = tempfile.mkdtemp(prefix="jfin_demo_")
    print(f"temp dir: {tmp}")

    print("\n[1] Building H3 fleet (4 stubbed infinite-livestream sessions)")
    h3 = H3Stub(width=64, height=64, n_frames=2)
    programs = [
        "neon city under heavy rain",
        "ancient library where books glow",
        "saturn rings dissolving into light",
        "macro beetle on wet bark",
    ]
    sessions = []
    for i, p in enumerate(programs):
        s = H3Session(h3, prompts=[p], frames_per_prompt=2)
        s.tick()
        sessions.append(s)
        print(f"  session {i+1}: '{p}' -> 1 frame emitted")

    print("\n[2] Registering Jellyfin Live TV channels")
    sched = JFinScheduler()
    channels = make_default_channels(n=4, base_url="http://jellyfin.local:8096")
    for ch in channels:
        hls_dir = os.path.join(tmp, ch.id)
        sched.register_channel(ch, hls_dir=hls_dir, width=64, height=64)
        sched.assign_program(ch.id, programs[int(ch.id.split("-")[1]) - 1])
    print(f"  {len(sched.channels)} channels registered")

    print("\n[3] Pushing 8 frames (one session per channel per tick)")
    for tick in range(8):
        for i, s in enumerate(sessions):
            f = s.tick()
            if f is not None:
                cid = f"atomic-{i+1:02d}"
                ok = sched.push_frame(cid, f.rgba, width=64, height=64)
                print(f"  tick {tick+1}: ch={cid} prompt='{f.prompt}' pushed={ok}")

    print("\n[4] Rotating channel mappings (round_robin)")
    sched.rotate(programs, mode="round_robin")
    for cid, prog in sched.mappings.items():
        print(f"  {cid} -> {prog}")

    print("\n[5] Rotating channel mappings (random)")
    sched.rotate(programs, mode="random")
    for cid, prog in sched.mappings.items():
        print(f"  {cid} -> {prog}")

    print("\n[6] Rotating channel mappings (h4_consensus)")
    sched.rotate(programs, mode="h4_consensus")
    for cid, prog in sched.mappings.items():
        print(f"  {cid} -> {prog}")

    print("\n[7] Writing M3U tuner emission to ", tmp)
    m3u = JFinM3U(livetv_dir=tmp)
    for ch in channels:
        m3u.add_channel(ch)
    single = m3u.write("atomic-all.m3u")
    print(f"  aggregate: {single}")
    per_channel = m3u.write_all()
    for p in per_channel:
        print(f"  per-channel: {p}")

    print("\n[8] Stopping all channels (cleanup)")
    sched.stop_all()
    shutil.rmtree(tmp, ignore_errors=True)

    print("\nOK -- iter28 demo complete.")


if __name__ == "__main__":
    main()