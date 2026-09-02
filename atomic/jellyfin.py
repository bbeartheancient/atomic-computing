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

__all__ = [
    "JFinExporter",
    "JFinM3U",
    "JFinChannel",
    "JFinScheduler",
    "DEFAULT_LIVETV_DIR",
    "DEFAULT_HLS_DIR",
]


DEFAULT_LIVETV_DIR = "/etc/jellyfin/livetv"
DEFAULT_HLS_DIR = "/var/lib/jellyfin/livetv"
DEFAULT_FFMPEG_BIN = "ffmpeg"


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

    def m3u_line(self) -> str:
        f = f'#EXTINF:-1 tvg-id="{self.id}" tvg-name="{self.name}" '
        f += f'tvg-logo="{self.logo_url}" group-title="{self.group}" '
        f += f'channel-id="{self.id}",{self.name}\n'
        f += f'{self.m3u_url}\n'
        return f

    def __repr__(self):
        return f"JFinChannel({self.id!r}, {self.name!r}, {self.m3u_url!r})"


class JFinExporter:
    """HLS/DASH exporter: owns the ffmpeg subprocess, writes HLS segments.

    On each tick, call push(frame: bytes, width, height) with a raw RGBA
    frame. The exporter pipes it to ffmpeg which transcodes to H.264/AAC
    in an HLS playlist. Jellyfin's Live TV DVR picks up the M3U playlist
    and surfaces the channel.

    The ffmpeg pipeline:
      raw RGBA (pipe) -> format=rgba -> scale -> x264 + aac -> HLS

    Attributes:
      channel     JFinChannel this exporter publishes
      hls_dir     Directory where .m3u8 + .ts segments land
      ffmpeg_bin  Path to ffmpeg binary
      running     bool: True while the ffmpeg subprocess is alive
      frame_count Total frames pushed
    """

    def __init__(self, channel: JFinChannel, hls_dir=None, ffmpeg_bin=None,
                 width=640, height=360, fps=30, segment_duration=2,
                 bitrate="1000k", audio=True):
        self.channel = channel
        self.hls_dir = str(hls_dir) if hls_dir else tempfile.mkdtemp(prefix="jfin_hls_")
        self.ffmpeg_bin = str(ffmpeg_bin) if ffmpeg_bin else DEFAULT_FFMPEG_BIN
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.segment_duration = int(segment_duration)
        self.bitrate = str(bitrate)
        self.audio = bool(audio)

        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.running = False
        self.frame_count = 0

        self._start_ffmpeg()

    def _start_ffmpeg(self):
        os.makedirs(self.hls_dir, exist_ok=True)
        playlist = os.path.join(self.hls_dir, "live.m3u8")

        cmd = [
            self.ffmpeg_bin,
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",  # stdin pipe
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-b:v", self.bitrate,
            "-pix_fmt", "yuv420p",
            "-f", "hls",
            "-hls_time", str(self.segment_duration),
            "-hls_list_size", "6",
            "-hls_flags", "delete_segments",
            "-hls_dir", self.hls_dir,
            playlist,
        ]

        if self.audio:
            cmd[5:5] = [
                "-f", "lavfi",
                "-i", "anullsrc=r=48000:cl=stereo",
                "-shortest",
            ]

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

    def push(self, frame: bytes, width=None, height=None) -> bool:
        """Push one RGBA frame (bytes) to the ffmpeg pipeline.

        Returns True on success, False if the pipeline is not running.
        Raises JFinError if the frame size is wrong or the pipe is broken.
        """
        with self._lock:
            if not self.running or self._proc is None:
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
                self._proc.wait(timeout=5.0)
                self._proc = None
            self.running = False

    @property
    def m3u_url(self) -> str:
        playlist = os.path.join(self.hls_dir, "live.m3u8")
        return f"file://{playlist}"

    def playlist_path(self) -> str:
        return os.path.join(self.hls_dir, "live.m3u8")

    def __repr__(self):
        return (f"JFinExporter({self.channel.id!r}, "
                f"hls_dir={self.hls_dir!r}, running={self.running})")

    def __del__(self):
        self.stop()


class JFinM3U:
    """HDHomeRun M3U tuner emitter.

    Writes /etc/jellyfin/livetv/*.m3u files from a list of JFinChannel
    objects. Jellyfin Live TV scans this directory for tuner definitions.

    HDHomeRun devices on the LAN can also serve M3U playlists directly
    via their HTTP interface; the M3U format is the same.

    Attributes:
      livetv_dir  Target directory for .m3u files
      channels    List of JFinChannel objects
    """

    def __init__(self, livetv_dir=None):
        self.livetv_dir = str(livetv_dir) if livetv_dir else DEFAULT_LIVETV_DIR
        self.channels: list[JFinChannel] = []

    def add_channel(self, channel: JFinChannel):
        self.channels.append(channel)

    def write(self, filename="atomic.m3u") -> str:
        """Write the M3U playlist to livetv_dir/filename.

        Returns the full path written.
        """
        os.makedirs(self.livetv_dir, exist_ok=True)
        path = os.path.join(self.livetv_dir, filename)
        lines = ["#EXTM3U\n"]
        for ch in self.channels:
            lines.append(ch.m3u_line())
        with open(path, "w") as f:
            f.writelines(lines)
        return path

    def write_all(self) -> list[str]:
        """Write one .m3u per channel (atomic-{id}.m3u)."""
        paths = []
        for ch in self.channels:
            safe_id = ch.id.replace("/", "_").replace("\\", "_")
            path = self.write(f"atomic-{safe_id}.m3u")
            paths.append(path)
        return paths

    @staticmethod
    def discover_hdhr() -> list[dict]:
        """Probe HDHomeRun devices on the LAN via UDP broadcast.

        Returns a list of discovered devices with keys: device_type,
        device_id, firmware, tuner_count, ip.
        """
        results = []
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(2.0)
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
                            tuner_count = int(ln.split(":", 1)[1].strip())
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

    def __repr__(self):
        return f"JFinM3U({len(self.channels)} channels, dir={self.livetv_dir!r})"


class JFinScheduler:
    """Channel scheduler: rotates which atomic program/H3 instance maps to which channel.

    A fleet of H3 sessions (each fed by a separate infinite-livestream process)
    becomes a randomized multichannel universe. The scheduler tracks the
    mapping and exposes the current active channels.

    Attributes:
      exporters   Dict[channel_id, JFinExporter]
      channels    Dict[channel_id, JFinChannel]
      mappings    Dict[channel_id, str]  # channel_id -> program_name
    """

    def __init__(self):
        self.exporters: dict[str, JFinExporter] = {}
        self.channels: dict[str, JFinChannel] = {}
        self.mappings: dict[str, str] = {}
        self._rotation_cursor = 0
        self._rotation_seed = 0

    def register_channel(self, channel: JFinChannel, hls_dir=None,
                        width=640, height=360) -> JFinExporter:
        """Create and register a new channel + exporter."""
        ex = JFinExporter(channel, hls_dir=hls_dir, width=width, height=height)
        self.channels[channel.id] = channel
        self.exporters[channel.id] = ex
        return ex

    def assign_program(self, channel_id: str, program_name: str):
        """Map a channel to an atomic program name."""
        self.mappings[channel_id] = program_name

    def push_frame(self, channel_id: str, frame: bytes, width=None, height=None) -> bool:
        """Push one frame to a channel's exporter. Returns success bool."""
        ex = self.exporters.get(channel_id)
        if ex is None:
            return False
        return ex.push(frame, width=width, height=height)

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
          round_robin  cursor advances one position per call
          random       shuffled assignment using seeded RNG
          h4_consensus picks the program whose hash's H4 W row is dominant

        Returns the new mappings dict.
        """
        if not programs or not self.channels:
            return dict(self.mappings)
        channel_ids = sorted(self.channels.keys())
        if mode == "random":
            rng = random.Random(self._rotation_seed)
            self._rotation_seed += 1
            programs_shuf = list(programs)
            rng.shuffle(programs_shuf)
            n = min(len(channel_ids), len(programs_shuf))
            for i in range(n):
                self.mappings[channel_ids[i]] = programs_shuf[i]
        elif mode == "h4_consensus":
            for cid in channel_ids:
                idx = (self._rotation_cursor
                       + sum(ord(c) for c in cid)) % len(programs)
                self.mappings[cid] = programs[idx]
                self._rotation_cursor += 1
        else:  # round_robin
            for cid in channel_ids:
                idx = self._rotation_cursor % len(programs)
                self.mappings[cid] = programs[idx]
                self._rotation_cursor += 1
        return dict(self.mappings)

    def stats(self) -> dict:
        return {
            cid: {
                "running": ex.running,
                "frame_count": ex.frame_count,
                "program": self.mappings.get(cid, ""),
                "hls_dir": ex.hls_dir,
                "playlist": ex.playlist_path(),
            }
            for cid, ex in self.exporters.items()
        }

    def __repr__(self):
        return f"JFinScheduler({len(self.exporters)} channels)"


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
