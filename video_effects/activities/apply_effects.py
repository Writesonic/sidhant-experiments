"""Activity: execute effects in phase order using the effect processor pipeline.

Single-pass architecture: all phases are applied per-frame in phase order,
with direct H.264 output. No intermediate files are created.
"""

import math
import os
import subprocess
import time

import cv2
import numpy as np
from temporalio import activity

from video_effects.effect_registry import group_by_phase
from video_effects.effects import ZoomEffect, BlurEffect, ColorEffect, SubtitleEffect
from video_effects.effects.base import EffectContext
from video_effects.schemas.effects import EffectCue, EffectType, VideoInfo

# Map effect types to processor classes
EFFECT_PROCESSORS = {
    EffectType.ZOOM: ZoomEffect,
    EffectType.BLUR: BlurEffect,
    EffectType.COLOR_CHANGE: ColorEffect,
    EffectType.SUBTITLE: SubtitleEffect,
}


@activity.defn(name="vfx_apply_effects")
def apply_effects(input_data: dict) -> dict:
    """Apply effects to video in phase order."""
    video_path = input_data["video_path"]
    output_dir = input_data["output_dir"]
    effects = [EffectCue(**e) for e in input_data["effects"]]
    video_info = VideoInfo(**input_data["video_info"])

    os.makedirs(output_dir, exist_ok=True)

    if not effects:
        print("[vfx] No effects to apply, passing through")
        return {"processed_video": video_path, "phases_executed": 0}

    phase_groups = group_by_phase(effects)
    print(f"[vfx] {len(phase_groups)} phases, {len(effects)} total effects")
    for phase_num, phase_effects in phase_groups.items():
        types = ", ".join(e.effect_type.value for e in phase_effects)
        print(f"[vfx]   phase {phase_num}: {len(phase_effects)} effects ({types})")

    # Initialize all processors upfront in phase order
    processors = []
    for phase_num, phase_effects in sorted(phase_groups.items()):
        effect_type = phase_effects[0].effect_type
        processor_cls = EFFECT_PROCESSORS.get(effect_type)
        if processor_cls is None:
            print(f"[vfx] ⚠ No processor for {effect_type}, skipping phase {phase_num}")
            continue
        processor = processor_cls()
        processor.setup(video_info, phase_effects)
        processors.append(processor)

    if not processors:
        print("[vfx] No valid processors, passing through")
        return {"processed_video": video_path, "phases_executed": 0}

    # Build merged active intervals across all processors
    all_intervals = _build_merged_intervals(processors, video_info)

    # Single pass: decode frames, apply all effects in phase order, encode to H.264
    output_path = os.path.join(output_dir, "processed.mp4")
    _process_single_pass(video_path, output_path, processors, video_info, all_intervals)

    print(f"[vfx] All {len(processors)} phases complete")
    return {"processed_video": output_path, "phases_executed": len(processors)}


def _build_merged_intervals(processors: list, video_info: VideoInfo) -> list[tuple[int, int]]:
    """Build sorted, merged list of (start_frame, end_frame) intervals across all processors."""
    fps = video_info.fps
    intervals = []
    for processor in processors:
        for cue in processor._cues:
            s = int(cue.start_time * fps)
            e = math.ceil(cue.end_time * fps) + 1
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


def _process_single_pass(
    input_path: str,
    output_path: str,
    processors: list,
    video_info: VideoInfo,
    active_intervals: list[tuple[int, int]],
) -> None:
    """Decode all frames, apply effects in phase order, encode to H.264."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    # Read actual dimensions from the video capture, not video_info,
    # to guarantee the raw pipe dimensions match what OpenCV decodes.
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = video_info.total_frames

    active_frame_count = sum(e - s for s, e in active_intervals)
    print(f"[vfx]   resolution: {w}x{h}, total frames: {total_frames}, "
          f"active: {active_frame_count} ({active_frame_count * 100 // max(total_frames, 1)}%)")
    for s, e in active_intervals:
        print(f"[vfx]     active: frames {s}-{e} "
              f"({s / video_info.fps:.1f}s - {e / video_info.fps:.1f}s)")

    # Pipe raw BGR frames to ffmpeg → H.264 output
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(video_info.fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            output_path,
        ],
        stdin=subprocess.PIPE,
    )

    expected_bytes = w * h * 3  # BGR24 = 3 bytes per pixel
    start_time = time.time()
    last_log = start_time

    try:
        for frame_index in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_index / video_info.fps

            # Apply all active effects in phase order
            if _is_in_active_interval(frame_index, active_intervals):
                context = EffectContext(
                    video_info=video_info,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    total_frames=total_frames,
                )
                for processor in processors:
                    frame = processor.apply_frame(frame, timestamp, context)

            # Ensure frame is contiguous BGR24 at the expected size
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
                activity.heartbeat(f"frame {frame_index + 1}/{total_frames}")
    finally:
        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed (exit {ffmpeg_proc.returncode})")

    elapsed = time.time() - start_time
    print(f"[vfx]   single-pass done: {total_frames} frames in {elapsed:.1f}s")
