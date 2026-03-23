"""Shared rendering infrastructure: decode, process, encode helpers.

Single-pass architecture: all phases are applied per-frame in phase order,
with direct H.264 output. No intermediate files are created.
"""

import json
import math
import re
import subprocess
import time

import cv2
import numpy as np
from collections.abc import Callable

from video_effects.effect_registry import group_by_phase
from video_effects.effects import (
    ZoomEffect, BlurEffect, ColorEffect,
    WhipEffect, VignetteEffect, SpeedRampEffect,
)
from video_effects.effects.base import EffectContext
from video_effects.schemas.effects import EffectType, VideoInfo

_HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}  # HLG, PQ/HDR10
_HDR_PRIMARIES = {"bt2020"}


def _is_hdr(video_info: VideoInfo) -> bool:
    return (video_info.color_transfer in _HDR_TRANSFERS
            or video_info.color_primaries in _HDR_PRIMARIES)


EFFECT_PROCESSORS = {
    EffectType.VIGNETTE: VignetteEffect,
    EffectType.ZOOM: ZoomEffect,
    EffectType.BLUR: BlurEffect,
    EffectType.COLOR_CHANGE: ColorEffect,
    EffectType.WHIP: WhipEffect,
    EffectType.SPEED_RAMP: SpeedRampEffect,
}


def _build_merged_intervals(processors: list, video_info: VideoInfo) -> list[tuple[int, int]]:
    """Build sorted, merged list of (start_frame, end_frame) intervals across all processors."""
    fps = video_info.fps
    intervals = []
    for processor in processors:
        for start_time, end_time in processor.get_active_ranges():
            s = int(start_time * fps)
            e = math.ceil(end_time * fps) + 1
            intervals.append((s, e))

    if not intervals:
        return []

    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _is_in_active_interval(frame_index: int, intervals: list[tuple[int, int]]) -> bool:
    """Check if a frame index falls within any active interval."""
    for s, e in intervals:
        if s <= frame_index < e:
            return True
        if frame_index < s:
            break
    return False


class _FrameDecoder:
    """Decode frames via ffmpeg pipe to rgb24."""

    def __init__(self, input_path: str, w: int, h: int, total_frames: int,
                 is_hdr: bool = False):
        self._frame_size = w * h * 3
        self._shape = (h, w, 3)
        if is_hdr:
            cmd = [
                "ffmpeg", "-i", input_path,
                "-frames:v", str(total_frames),
                "-vf", (
                    "zscale=t=linear:npl=100,format=gbrpf32le,"
                    "zscale=p=bt709,"
                    "tonemap=tonemap=hable:desat=0,"
                    "zscale=t=bt709:m=bt709:r=tv,"
                    "format=rgb24"
                ),
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1",
            ]
        else:
            cmd = [
                "ffmpeg", "-i", input_path,
                "-frames:v", str(total_frames),
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1",
            ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        data = self._proc.stdout.read(self._frame_size)
        if len(data) != self._frame_size:
            return False, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape(self._shape).copy()
        return True, frame

    def release(self):
        self._proc.stdout.close()
        self._proc.terminate()
        self._proc.wait()


def _probe_decoded_size(input_path: str) -> tuple[int, int]:
    """Get actual frame dimensions after ffmpeg autorotate."""
    cmd = [
        "ffmpeg", "-i", input_path,
        "-frames:v", "1",
        "-vf", "showinfo",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r"s:(\d+)x(\d+)", result.stderr)
    if match:
        return int(match.group(1)), int(match.group(2))
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-print_format", "json", input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _process_single_pass(
    input_path: str,
    output_path: str,
    processors: list,
    video_info: VideoInfo,
    active_intervals: list[tuple[int, int]],
    *,
    decoded_width: int | None = None,
    decoded_height: int | None = None,
    heartbeat_fn: Callable[[str], None] | None = None,
) -> None:
    """Decode all frames, apply effects in phase order, encode to H.264.

    Uses ffmpeg pipe for both decode and encode with rgb24 as the intermediate
    format, matching the proven color-correct pipeline from zoom_bounce.py.
    """
    if decoded_width and decoded_height:
        w, h = decoded_width, decoded_height
    else:
        w, h = _probe_decoded_size(input_path)
    total_frames = video_info.total_frames
    is_hdr = _is_hdr(video_info)

    if is_hdr:
        print(f"[vfx]   HDR detected (transfer={video_info.color_transfer}, "
              f"primaries={video_info.color_primaries}) — tone mapping to SDR")

    decoder = _FrameDecoder(input_path, w, h, total_frames, is_hdr=is_hdr)

    active_frame_count = sum(e - s for s, e in active_intervals)
    print(f"[vfx]   resolution: {w}x{h}, total frames: {total_frames}, "
          f"active: {active_frame_count} ({active_frame_count * 100 // max(total_frames, 1)}%)")
    for s, e in active_intervals:
        print(f"[vfx]     active: frames {s}-{e} "
              f"({s / video_info.fps:.1f}s - {e / video_info.fps:.1f}s)")

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(video_info.fps),
            "-i", "pipe:0",
            "-vf", "scale=in_range=full:in_color_matrix=bt709:out_range=tv:out_color_matrix=bt709",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16",
            "-pix_fmt", "yuv420p",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-color_range", "tv",
            "-movflags", "+faststart",
            "-an",
            output_path,
        ],
        stdin=subprocess.PIPE,
    )

    expected_bytes = w * h * 3
    start_time = time.time()
    last_log = start_time

    try:
        for frame_index in range(total_frames):
            ret, frame = decoder.read()
            if not ret:
                break

            timestamp = frame_index / video_info.fps

            if _is_in_active_interval(frame_index, active_intervals):
                context = EffectContext(
                    video_info=video_info,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    total_frames=total_frames,
                )
                for processor in processors:
                    frame = processor.apply_frame(frame, timestamp, context)

            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)

            raw = frame.tobytes()
            assert len(raw) == expected_bytes, (
                f"Frame {frame_index}: expected {expected_bytes} bytes, got {len(raw)}"
            )
            ffmpeg_proc.stdin.write(raw)

            now = time.time()
            if now - last_log >= 2.0:
                elapsed = now - start_time
                fps_rate = (frame_index + 1) / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_index - 1) / fps_rate if fps_rate > 0 else 0
                print(
                    f"[vfx]   {frame_index + 1}/{total_frames} frames "
                    f"| {fps_rate:.1f} fps | ETA {eta:.0f}s"
                )
                last_log = now
                if heartbeat_fn:
                    heartbeat_fn(f"frame {frame_index + 1}/{total_frames}")
    finally:
        decoder.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed (exit {ffmpeg_proc.returncode})")

    elapsed = time.time() - start_time
    print(f"[vfx]   single-pass done: {total_frames} frames in {elapsed:.1f}s")
