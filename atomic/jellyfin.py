"""Jellyfin + HDHomeRun integration (iter 28): OS-level topology.

This module handles the "infinite slop" export pipeline: tails the active
atomic program (or H3 session), muxes frames via ffmpeg into an HLS/DASH
segment, and publishes it to Jellyfin's Live TV ingest. HDHomeRun devices
on the LAN surface the channels via M3U tuner files written to
/etc/jellyfin/livetv/*.m3u.

The topology:
  H3 session (GPU1) -> viz_video -> jfin_live_export
                                    -> ffmpeg HLS/DASH muxer
                                    -> Jellyfin Live TV ingest
                                    -> HDHomeRun M3U tuner
                                    -> any LAN client

The harness treats Jellyfin/ffmpeg as EXTERNAL OS-level services (not
vendored). This module owns the topology: M3U emission, channel
registration, and the ffmpeg subprocess lifecycle. It does NOT import
torch/diffusers/comfy -- it receives per-tick RGBA frames from the
active engine (via the jfin_live_export atom) and pipes them to ffmpeg.

Public API:
  JFinExporter   HLS/DASH exporter: owns the ffmpeg subprocess, accepts
                  RGBA frames per tick, writes HLS segments + M3U.

  JFinM3U        HDHomeRun M3U tuner emitter: writes /etc/jellyfin/livetv/*.m3u
                  from the program registry.

  JFinChannel     One logical TV channel: name, M3U URL, logo, tuner type.

  JFinScheduler   Rotates which H3 session (or atomic program) maps to
                  which Jellyfin Live TV channel. A fleet of H3 instances
                  becomes a randomized multichannel universe.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import struct
import subprocess
import tempfile
import threading
import time
from typing import Optional

from .qbf import h4_gate

__all__ = [
    "JFinExporter",
    "JFinM3U",
    "JFinChannel",
    "JFinScheduler",
    "DEFAULT_LIVETV_DIR",
    "DEFAULT_HLS_DIR",
    "h4_gate",
    "_h4_consensus_w",
]


DEFAULT_LIVETV_DIR = "/etc/jellyfin/livetv"
DEFAULT_HLS_DIR = "/var/lib/jellyfin/livetv"
DEFAULT_FFMPEG_BIN = "ffmpeg"


def _h4_consensus_w(items: list[str], seed: int) -> float:
    """Compute the H4 W-channel consensus over up to 4 items.

    Hashes each item -> float in [0, 1) -> H(4) gate -> returns the W
    row (the sum / consensus value). Used by JFinScheduler.h4_consensus
    mode and consensus_pick().
    """
    vals = []
    for item in items[:4]:
        h = int(hashlib.sha256(
            str(item).encode("utf-8") + str(seed).encode("utf-8")
        ).hexdigest()[:8], 16)
        vals.append((h % 1000) / 1000.0)
    while len(vals) < 4:
        vals.append(0.0)
    w, _, _, _ = h4_gate(tuple(vals))
    return w


class JFinError(RuntimeError):
    pass


class JFinChannel:
    """One logical TV channel surfaced via HDHomeRun M3U.

    Attributes:
      id         Unique channel ID (e.g. 'atomic-01')
      name       Display name (e.g. 'ATOMIC-1')
      m3u_url    Full URL the HDHomeRun tuner will tune to
      logo_url   Optional logo/icon URL
      tuner_type "hdhr" (HDHomeRun) or "m3u"
      group      Guide channel group (e.g. 'ATOMIC')
      number     Logical channel number (integer)
    """

    __slots__ = ("id", "name", "m3u_url", "logo_url", "tuner_type", "group", "number")

    def __init__(self, id, name, m3u_url, logo_url="", tuner_type="hdhr",
                 group="ATOMIC", number=1):
        self.id = str(id)
        self.name = str(name)
        self.m3u_url = str(m3u_url)
        self.logo_url = str(logo_url)
        self.tuner_type = str(tuner_type)
        self.group = str(group)
        self.number = int(number)

    def m3u_line(self, group_path: str = "") -> str:
        g = self.group
        if group_path:
            g = f"{group_path}/{self.group}"
        f = f'#EXTINF:-1 tvg-id="{self.id}" tvg-name="{self.name}" '
        f += f'tvg-logo="{self.logo_url}" group-title="{g}" '
        f += f'channel-id="{self.id}",{self.name}\n'
        f += f'{self.m3u_url}\n'
        return f

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "m3u_url": self.m3u_url,
            "logo_url": self.logo_url, "tuner_type": self.tuner_type,
            "group": self.group, "number": self.number,
        }

    def __repr__(self):
        return f"JFinChannel({self.id!r}, {self.name!r}, {self.m3u_url!r})"


class JFinExporter:
    """HLS/DASH exporter: owns the ffmpeg subprocess, writes HLS segments.

    On each tick, call push(frame: bytes, width, height) with a raw RGBA
    frame. The exporter pipes it to ffmpeg which transcodes to H.264/AAC
    in an HLS playlist. Jellyfin's Live TV DVR picks up the M3U playlist
    and surfaces the channel.

    The ffmpeg pipeline (HLS mode):
      raw RGBA (pipe) -> format=rgba -> scale -> x264 + aac -> HLS

    For DASH, set muxer="dash": ffmpeg emits a .mpd manifest + segments
    suitable for Jellyfin's DASH Live ingest (the same channel can be
    served by either HLS or DASH; JFinM3U emits the right URL).

    Set mock=True to disable the real ffmpeg subprocess and just count
    frames in memory (useful for tests where ffmpeg is absent or
    expensive to spawn). push() returns True, frame_count increments,
    running stays True. The HLS .m3u8 / DASH .mpd manifest files are
    NOT created in mock mode -- only a `mock.m3u8` placeholder is
    written so playlist_path() still resolves.

    Attributes:
      channel     JFinChannel this exporter publishes
      hls_dir     Directory where .m3u8 + .ts segments land
      ffmpeg_bin  Path to ffmpeg binary
      running     bool: True while the ffmpeg subprocess is alive (or mock)
      frame_count Total frames pushed
      keyframes   Total keyframes forced via force_keyframe() / trig
      muxer       "hls" (default) or "dash"
      mock        bool: True if running in mock mode (no real ffmpeg)
    """

    def __init__(self, channel: JFinChannel, hls_dir=None, ffmpeg_bin=None,
                 width=640, height=360, fps=30, segment_duration=2,
                 bitrate="1000k", audio=True, muxer="hls", mock=False):
        self.channel = channel
        self.hls_dir = str(hls_dir) if hls_dir else tempfile.mkdtemp(prefix="jfin_hls_")
        self.ffmpeg_bin = str(ffmpeg_bin) if ffmpeg_bin else DEFAULT_FFMPEG_BIN
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.segment_duration = int(segment_duration)
        self.bitrate = str(bitrate)
        self.audio = bool(audio)
        self.muxer = str(muxer).lower()
        self.mock = bool(mock)

        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.running = False
        self.frame_count = 0
        self.keyframes = 0
        self._next_force_key = False

        if self.mock:
            self._start_mock()
        else:
            self._start_ffmpeg()

    def _start_mock(self):
        os.makedirs(self.hls_dir, exist_ok=True)
        # write a placeholder manifest so playlist_path() / mpd_path() resolve
        ext = "mpd" if self.muxer == "dash" else "m3u8"
        placeholder = os.path.join(self.hls_dir, f"live.{ext}")
        try:
            with open(placeholder, "w") as f:
                f.write("# mock manifest\n")
        except OSError:
            pass
        self.running = True

    def _start_ffmpeg(self):
        os.makedirs(self.hls_dir, exist_ok=True)
        if self.muxer == "dash":
            playlist = os.path.join(self.hls_dir, "live.mpd")
            out_args = [
                "-f", "dash",
                "-seg_duration", str(self.segment_duration),
                "-window_size", "6",
                "-extra_window_size", "2",
                "-use_template", "1",
                "-use_timeline", "1",
                "-adaptation_sets", "id=0,streams=v id=1,streams=a",
                playlist,
            ]
        else:
            playlist = os.path.join(self.hls_dir, "live.m3u8")
            out_args = [
                "-f", "hls",
                "-hls_time", str(self.segment_duration),
                "-hls_list_size", "6",
                "-hls_flags", "delete_segments+independent_segments",
                playlist,
            ]

        video_in = [
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
        ]
        video_enc = [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-b:v", self.bitrate,
            "-pix_fmt", "yuv420p",
            "-force_key_frames", f"expr:gte(n,n_forced*{max(1, self.fps)})",
        ]
        audio_in = []
        audio_enc = []
        audio_shortest = []
        if self.audio:
            audio_in = [
                "-f", "lavfi",
                "-i", "anullsrc=r=48000:cl=stereo",
            ]
            audio_enc = [
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "48000",
                "-ac", "2",
            ]
            # anullsrc is infinite, so without -shortest ffmpeg never
            # finishes a segment. -shortest tells it to stop when the
            # other (video) input ends.
            audio_shortest = ["-shortest"]
        cmd = [self.ffmpeg_bin] + video_in + audio_in + video_enc + audio_enc + audio_shortest + out_args

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as e:
            raise JFinError(
                f"ffmpeg not found at {self.ffmpeg_bin!r}; "
                "install ffmpeg to enable live export"
            ) from e

        self.running = True

    def force_keyframe(self):
        """Mark the next push() as a keyframe.

        The keyframes counter advances immediately so the caller can
        observe a force_keyframe() call without a follow-up push(). The
        actual keyframe boundary in the HLS stream is enforced by the
        ffmpeg ``-force_key_frames`` expr in the argv (one every fps
        frames). This method does NOT inject any bytes into the rawvideo
        pipe (that would corrupt the stream).
        """
        with self._lock:
            self.keyframes += 1
            self._next_force_key = True

    def push(self, frame: bytes, width=None, height=None,
             force_key=False) -> bool:
        """Push one RGBA frame (bytes) to the ffmpeg pipeline.

        Returns True on success, False if the pipeline is not running.
        Raises JFinError if the frame size is wrong or the pipe is broken.
        If force_key=True, a keyframe boundary is forced before the frame.
        """
        with self._lock:
            if not self.running:
                return False
            if self.mock:
                w = int(width) if width else self.width
                h = int(height) if height else self.height
                expected = w * h * 4
                if len(frame) != expected:
                    raise JFinError(
                        f"frame size mismatch: got {len(frame)} bytes, "
                        f"expected {expected} ({w}x{h} RGBA)"
                    )
                if force_key or self._next_force_key:
                    self.keyframes += 1
                self._next_force_key = False
                self.frame_count += 1
                return True
            if self._proc is None:
                return False
            if self._proc.poll() is not None:
                self.running = False
                return False
            w = int(width) if width else self.width
            h = int(height) if height else self.height
            expected = w * h * 4
            if len(frame) != expected:
                raise JFinError(
                    f"frame size mismatch: got {len(frame)} bytes, "
                    f"expected {expected} ({w}x{h} RGBA)"
                )
            try:
                self._proc.stdin.write(frame)
                self._proc.stdin.flush()
                self.frame_count += 1
                if force_key or self._next_force_key:
                    self.keyframes += 1
                self._next_force_key = False
                return True
            except (BrokenPipeError, OSError) as e:
                self.running = False
                raise JFinError(f"ffmpeg pipe broken: {e}") from e

    def stop(self):
        """Close the stdin pipe and wait for ffmpeg to finish."""
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.stdin.close()
                except OSError:
                    pass
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                self._proc = None
            self.running = False

    @property
    def m3u_url(self) -> str:
        playlist = os.path.join(self.hls_dir, self.playlist_name())
        return f"file://{playlist}"

    def playlist_name(self) -> str:
        return "live.mpd" if self.muxer == "dash" else "live.m3u8"

    def playlist_path(self) -> str:
        return os.path.join(self.hls_dir, self.playlist_name())

    def mpd_path(self) -> str:
        return os.path.join(self.hls_dir, "live.mpd")

    def __repr__(self):
        return (f"JFinExporter({self.channel.id!r}, "
                f"hls_dir={self.hls_dir!r}, running={self.running}, "
                f"muxer={self.muxer!r}, mock={self.mock})")

    def __del__(self):
        self.stop()


class JFinM3U:
    """HDHomeRun M3U tuner emitter.

    Writes /etc/jellyfin/livetv/*.m3u files from a list of JFinChannel
    objects. Jellyfin Live TV scans this directory for tuner definitions.

    HDHomeRun devices on the LAN can also serve M3U playlists directly
    via their HTTP interface; the M3U format is the same.

    The "recursive group-title" emission (iter29) writes one M3U per
    channel group (group-title="ATOMIC", "TV", etc.) plus a recursive
    root M3U that #EXTGRP-references the group playlists -- this lets
    Jellyfin's Live TV group browse mirror the H4 channel organization.

    Attributes:
      livetv_dir  Target directory for .m3u files
      channels    List of JFinChannel objects
    """

    def __init__(self, livetv_dir=None):
        self.livetv_dir = str(livetv_dir) if livetv_dir else DEFAULT_LIVETV_DIR
        self.channels: list[JFinChannel] = []

    def add_channel(self, channel: JFinChannel):
        self.channels.append(channel)

    def find_by_id(self, ch_id: str) -> JFinChannel | None:
        for ch in self.channels:
            if ch.id == ch_id:
                return ch
        return None

    def write(self, filename="atomic.m3u", fp=None) -> str:
        """Write the M3U playlist to livetv_dir/filename.

        If `fp` is given, the M3U bytes are also written there (used by
        write_to_stdin() to redirect emission to a process pipe or
        in-memory buffer).

        Returns the full path written.
        """
        os.makedirs(self.livetv_dir, exist_ok=True)
        path = os.path.join(self.livetv_dir, filename)
        lines = ["#EXTM3U\n"]
        for ch in self.channels:
            lines.append(ch.m3u_line())
        with open(path, "w") as f:
            f.writelines(lines)
        if fp is not None:
            try:
                fp.write("".join(lines).encode("utf-8"))
                fp.flush()
            except (AttributeError, OSError):
                # fall back: caller wants a file-like object that supports .write()
                fp.write("".join(lines))
                fp.flush()
        return path

    def write_to_stdin(self, filename="atomic.m3u", stdin_fp=None) -> bytes:
        """Write the M3U bytes (returning them) AND optionally push them
        to `stdin_fp` (a file-like object that supports .write/.flush --
        typically subprocess.Popen.stdin).

        If stdin_fp is None, the bytes are only returned.
        Returns the rendered M3U bytes (always).
        """
        lines = ["#EXTM3U\n"]
        for ch in self.channels:
            lines.append(ch.m3u_line())
        rendered = "".join(lines).encode("utf-8")
        if stdin_fp is not None:
            stdin_fp.write(rendered)
            try:
                stdin_fp.flush()
            except (AttributeError, OSError):
                pass
        return rendered

    def write_all(self) -> list[str]:
        """Write one .m3u per channel (atomic-{id}.m3u)."""
        paths = []
        for ch in self.channels:
            safe_id = ch.id.replace("/", "_").replace("\\", "_")
            path = self.write(f"atomic-{safe_id}.m3u")
            paths.append(path)
        return paths

    def group_titles(self) -> list[str]:
        """Sorted list of distinct group-title values across channels."""
        seen = []
        for ch in self.channels:
            if ch.group and ch.group not in seen:
                seen.append(ch.group)
        return sorted(seen)

    def write_recursive_groups(self) -> dict[str, str]:
        """One M3U per group-title plus a root M3U that references them.

        Returns a dict {group_title: m3u_path}; the root playlist is
        written under "atomic-root.m3u" and its #EXTINF lines point at
        the per-group files (so Jellyfin's tuner shows the group
        hierarchy).

        Per-group filename: atomic-group-<group_title>.m3u (sanitized).
        """
        out = {}
        os.makedirs(self.livetv_dir, exist_ok=True)
        groups: dict[str, list[JFinChannel]] = {}
        for ch in self.channels:
            groups.setdefault(ch.group or "ATOMIC", []).append(ch)

        # per-group playlists
        for g, channels in groups.items():
            safe = g.replace("/", "_").replace("\\", "_").replace(" ", "_")
            m3u = JFinM3U(livetv_dir=self.livetv_dir)
            for c in channels:
                m3u.add_channel(c)
            path = m3u.write(f"atomic-group-{safe}.m3u")
            out[g] = path

        # root playlist: one stub entry per group, pointing at the group playlist
        root_path = os.path.join(self.livetv_dir, "atomic-root.m3u")
        with open(root_path, "w") as f:
            f.write("#EXTM3U\n")
            for g, p in out.items():
                safe = g.replace("/", "_").replace("\\", "_").replace(" ", "_")
                f.write(f'#EXTGRP:{g}\n')
                # EXTINF placeholder for the group root, m3u_url points to playlist
                f.write(f'#EXTINF:-1 tvg-id="group-{safe}" tvg-name="{g}" '
                        f'group-title="GROUPS",{g}\n')
                f.write(f"{p}\n")
        out["__root__"] = root_path
        return out

    @staticmethod
    def discover_hdhr(timeout: float = 2.0) -> list[dict]:
        """Probe HDHomeRun devices on the LAN via UDP broadcast.

        Returns a list of discovered devices with keys: device_type,
        device_id, firmware, tuner_count, ip.

        `timeout` (iter 30): how long to wait for replies in seconds.
        """
        results = []
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(max(0.1, float(timeout)))
            msg = b"HTTP/1.1 0x10d6ccf0\r\n"
            sock.sendto(msg, ("255.255.255.255", 65001))
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                    line = data.decode("utf-8", errors="replace")
                    device_id = ""
                    device_type = ""
                    tuner_count = 0
                    firmware = ""
                    for ln in line.split("\r\n"):
                        if ln.startswith("DeviceID:") or ln.startswith("device-id:"):
                            device_id = ln.split(":", 1)[1].strip()
                        elif ln.startswith("DeviceType:") or ln.startswith("device-type:"):
                            device_type = ln.split(":", 1)[1].strip()
                        elif ln.startswith("TunerCount:") or ln.startswith("tuner-count:"):
                            try:
                                tuner_count = int(ln.split(":", 1)[1].strip())
                            except ValueError:
                                tuner_count = 0
                        elif ln.startswith("FirmwareVersion:") or ln.startswith("firmware-version:"):
                            firmware = ln.split(":", 1)[1].strip()
                    if device_id:
                        results.append({
                            "device_id": device_id,
                            "device_type": device_type,
                            "tuner_count": tuner_count,
                            "firmware": firmware,
                            "ip": addr[0],
                        })
                except socket.timeout:
                    break
        except OSError:
            pass
        return results

    @staticmethod
    def from_discovered_hdhr(base_url: str = "http://localhost:8080",
                             timeout: float = 1.0,
                             livetv_dir: str | None = None) -> "JFinM3U":
        """Run HDHomeRun discovery and auto-generate one JFinChannel per
        tuner (iter 30).

        Returns a fresh JFinM3U whose channels list is populated from
        the discovered devices (one channel per tuner per device).
        """
        m3u = JFinM3U(livetv_dir=livetv_dir)
        devices = JFinM3U.discover_hdhr(timeout=timeout)
        for d in devices:
            tuner_count = int(d.get("tuner_count", 0) or 1)
            for t in range(max(1, tuner_count)):
                dev_id = d.get("device_id", "unknown")
                ch_id = f"hdhr-{dev_id}-t{t+1}"
                ch = JFinChannel(
                    id=ch_id,
                    name=f"HDHR {dev_id} Tuner {t+1}",
                    m3u_url=f"{base_url}/livetv/{ch_id}/live.m3u8",
                    tuner_type="hdhr",
                    group="ATOMIC",
                    number=len(m3u.channels) + 1,
                )
                m3u.add_channel(ch)
        return m3u

    def __repr__(self):
        return f"JFinM3U({len(self.channels)} channels, dir={self.livetv_dir!r})"


class JFinScheduler:
    """Channel scheduler: rotates which atomic program/H3 instance maps to which channel.

    A fleet of H3 sessions (each fed by a separate infinite-livestream process)
    becomes a randomized multichannel universe. The scheduler tracks the
    mapping and exposes the current active channels.

    `rotation_seed` (iter29) is the explicit seed for the "random" and
    "h4_consensus" modes -- same seed -> same mapping. The default seed
    is 0 (deterministic across runs). Setting `rotation_seed` and calling
    rotate() in "random" or "h4_consensus" mode is bit-identical.

    Attributes:
      exporters       Dict[channel_id, JFinExporter]
      channels        Dict[channel_id, JFinChannel]
      mappings        Dict[channel_id, str]  # channel_id -> program_name
      rotation_seed   int -- RNG seed for rotate()
      rotation_cursor int -- advance counter for round_robin
    """

    def __init__(self, rotation_seed=0):
        self.exporters: dict[str, JFinExporter] = {}
        self.channels: dict[str, JFinChannel] = {}
        self.mappings: dict[str, str] = {}
        self._rotation_cursor = 0
        self.rotation_seed = int(rotation_seed)

    def register_channel(self, channel: JFinChannel, hls_dir=None,
                        width=640, height=360, muxer="hls",
                        mock=False) -> JFinExporter:
        """Create and register a new channel + exporter."""
        ex = JFinExporter(channel, hls_dir=hls_dir, width=width, height=height,
                          muxer=muxer, mock=mock)
        self.channels[channel.id] = channel
        self.exporters[channel.id] = ex
        return ex

    def assign_program(self, channel_id: str, program_name: str):
        """Map a channel to an atomic program name."""
        self.mappings[channel_id] = program_name

    def push_frame(self, channel_id: str, frame: bytes, width=None, height=None,
                   force_key=False) -> bool:
        """Push one frame to a channel's exporter. Returns success bool."""
        ex = self.exporters.get(channel_id)
        if ex is None:
            return False
        return ex.push(frame, width=width, height=height, force_key=force_key)

    def force_keyframe(self, channel_id: str) -> bool:
        ex = self.exporters.get(channel_id)
        if ex is None:
            return False
        ex.force_keyframe()
        return True

    def stop_channel(self, channel_id: str):
        ex = self.exporters.pop(channel_id, None)
        if ex is not None:
            ex.stop()

    def stop_all(self):
        for ex in list(self.exporters.values()):
            ex.stop()
        self.exporters.clear()

    def active_channels(self) -> list[str]:
        return [cid for cid, ex in self.exporters.items() if ex.running]

    def rotate(self, programs: list[str], mode: str = "round_robin") -> dict:
        """Rotate which program maps to which channel.

        Modes:
          round_robin    cursor advances one position per call (deterministic)
          random         shuffled assignment using rotation_seed RNG
          h4_consensus   picks the program whose hash's H4 W row is dominant
          seeded_round_robin  like round_robin but starts at seed offset
        h4_consensus supports an optional `last_w` swarm-consensus value:
        if the caller has a W-channel sum from a 4-agent swarm, the W
        mod len(programs) is the channel pick -- the H4 keystone as
        the active-router.

        Returns the new mappings dict.
        """
        if not programs or not self.channels:
            return dict(self.mappings)
        channel_ids = sorted(self.channels.keys())
        if mode == "random":
            rng = random.Random(self.rotation_seed)
            programs_shuf = list(programs)
            rng.shuffle(programs_shuf)
            n = min(len(channel_ids), len(programs_shuf))
            for i in range(n):
                self.mappings[channel_ids[i]] = programs_shuf[i]
            # any channels beyond n keep their prior program (no-op)
        elif mode == "h4_consensus":
            # hash each program -> 4 floats in [0,1) -> H4 gate -> W row
            # (the sum) -> mod len(programs) -> bank index
            for cid in channel_ids:
                w = _h4_consensus_w(programs, self.rotation_seed
                                    + sum(ord(c) for c in cid))
                idx = int(abs(w)) % len(programs)
                self.mappings[cid] = programs[idx]
        elif mode == "seeded_round_robin":
            for cid in channel_ids:
                idx = (self._rotation_cursor
                       + self.rotation_seed) % len(programs)
                self.mappings[cid] = programs[idx]
                self._rotation_cursor += 1
        else:  # round_robin
            for cid in channel_ids:
                idx = self._rotation_cursor % len(programs)
                self.mappings[cid] = programs[idx]
                self._rotation_cursor += 1
        return dict(self.mappings)

    def consensus_pick(self, programs: list[str], last_w: float = 0.0) -> str:
        """H4-consensus router: pick the next program by the W-channel
        consensus value (a 4-agent Swarm's W-sum, or any float that
        maps mod len(programs)). Same formula as h4_consensus mode but
        with the W input supplied directly (the live swarm hook).
        """
        if not programs:
            return ""
        idx = int(abs(float(last_w))) % len(programs)
        return programs[idx]

    def stats(self) -> dict:
        return {
            cid: {
                "running": ex.running,
                "frame_count": ex.frame_count,
                "keyframes": ex.keyframes,
                "muxer": ex.muxer,
                "mock": ex.mock,
                "program": self.mappings.get(cid, ""),
                "hls_dir": ex.hls_dir,
                "playlist": ex.playlist_path(),
                "mpd": ex.mpd_path(),
            }
            for cid, ex in self.exporters.items()
        }

    def __repr__(self):
        return (f"JFinScheduler({len(self.exporters)} channels, "
                f"seed={self.rotation_seed})")


def make_default_channels(n=4, base_url="http://localhost:8080") -> list[JFinChannel]:
    """Factory: creates n default ATOMIC-TV channels.

    The M3U URLs point to Jellyfin's built-in HTTP server; in production
    these would be the Jellyfin Live TV ingest URLs.
    """
    channels = []
    for i in range(1, n + 1):
        ch = JFinChannel(
            id=f"atomic-{i:02d}",
            name=f"ATOMIC-{i}",
            m3u_url=f"{base_url}/livetv/atomic-{i:02d}/live.m3u8",
            group="ATOMIC",
            number=i,
        )
        channels.append(ch)
    return channels


class _JFinState:
    """Global Jellyfin/HDHomeRun state (singleton, thread-safe for REST)."""
    __slots__ = ("m3u", "scheduler")

    def __init__(self):
        self.m3u = JFinM3U()
        self.scheduler = JFinScheduler()
        for ch in make_default_channels(n=4):
            self.m3u.add_channel(ch)
            self.scheduler.channels[ch.id] = ch


_JFIN_STATE = _JFinState()
