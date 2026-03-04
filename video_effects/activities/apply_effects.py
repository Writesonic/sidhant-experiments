"""Activity: execute effects in phase order using the effect processor pipeline."""

import math
import os
import subprocess
import time

import cv2
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

    current_video = video_path
    phases_executed = 0

    for phase_num, phase_effects in phase_groups.items():
        effect_type = phase_effects[0].effect_type
        processor_cls = EFFECT_PROCESSORS.get(effect_type)
        if processor_cls is None:
            print(f"[vfx] ⚠ No processor for {effect_type}, skipping phase {phase_num}")
            continue

        print(f"[vfx] ── Phase {phase_num}: {effect_type.value} ({len(phase_effects)} cues) ──")

        processor = processor_cls()
        processor.setup(video_info, phase_effects)

        current_video = _process_video_with_effect(
            current_video, output_dir, f"phase_{phase_num}",
            processor, video_info, phase_num,
        )
        phases_executed += 1

    print(f"[vfx] All {phases_executed} phases complete")
    return {
        "processed_video": current_video,
        "phases_executed": phases_executed,
    }


def _build_active_intervals(processor, video_info: VideoInfo) -> list[tuple[int, int]]:
    """Build sorted list of (start_frame, end_frame) intervals where effects are active.

    Merges overlapping intervals.
    """
    fps = video_info.fps
    intervals = []
    for cue in processor._cues:
        s = int(cue.start_time * fps)
        e = math.ceil(cue.end_time * fps) + 1  # inclusive end
        intervals.append((s, e))

    if not intervals:
        return []

    # Merge overlapping
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# Consistent FFV1 encoding args so all segments are concat-compatible.
# -level 3 -slices 4: modern FFV1 with multi-threaded slice encoding
# -context 0: no cross-frame context (each frame independently decodable)
# -g 1: every frame is a keyframe, enabling frame-accurate stream copy
# -pix_fmt bgr0: forces consistent pixel format across all segments.
#   Without this, transcode passthrough inherits source pix_fmt (e.g. yuv420p10le)
#   while active segments piped as BGR24 produce bgr0 → bytestream mismatch on concat.
_FFV1_ENC_ARGS = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "4", "-context", "0", "-pix_fmt", "bgr0"]


def _probe_video_codec(path: str) -> str:
    """Detect the video codec of a file using ffprobe."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0", path,
        ],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def _ffmpeg_extract_segment(
    input_path: str, output_path: str,
    start_frame: int, end_frame: int, video_info: VideoInfo,
    input_codec: str = "",
) -> None:
    """Extract a frame range with ffmpeg using fast seek + exact frame count.

    Two modes based on input codec:
    - FFV1 input: true stream copy (-c:v copy). Frame-accurate because every
      FFV1 frame is a keyframe (-g 1). Near-instant, no decode/encode.
    - Non-FFV1 input: transcode to FFV1. -ss before -i for fast keyframe seek.

    Both use -frames:v for exact output frame count.
    """
    stream_copy = input_codec == "ffv1"
    n_frames = end_frame - start_frame

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]

    if start_frame > 0:
        start_time = start_frame / video_info.fps
        cmd += ["-ss", f"{start_time:.6f}"]

    cmd += ["-i", input_path, "-frames:v", str(n_frames)]

    if stream_copy:
        cmd += ["-c:v", "copy"]
    else:
        cmd += _FFV1_ENC_ARGS

    cmd += ["-an", output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extract failed (exit {result.returncode}): {result.stderr}")


def _process_active_segment(
    input_path: str, output_path: str,
    start_frame: int, end_frame: int,
    processor, video_info: VideoInfo, phase_num: int,
) -> None:
    """Decode, apply effects, and pipe processed frames to ffmpeg for encoding.

    Uses ffmpeg for FFV1 encoding (not OpenCV) so all segments share
    the same encoder parameters and can be concatenated with stream copy.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    w, h = video_info.width, video_info.height

    # Pipe raw BGR frames to ffmpeg for consistent FFV1 encoding
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(video_info.fps),
            "-i", "pipe:0",
            *_FFV1_ENC_ARGS, "-an",
            output_path,
        ],
        stdin=subprocess.PIPE,
    )

    n_frames = end_frame - start_frame
    start_time = time.time()
    last_log = start_time

    try:
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_index = start_frame + i
            timestamp = frame_index / video_info.fps
            context = EffectContext(
                video_info=video_info,
                frame_index=frame_index,
                timestamp=timestamp,
                total_frames=video_info.total_frames,
            )
            processed = processor.apply_frame(frame, timestamp, context)
            ffmpeg_proc.stdin.write(processed.tobytes())

            now = time.time()
            if now - last_log >= 2.0:
                elapsed = now - start_time
                fps_rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_frames - i - 1) / fps_rate if fps_rate > 0 else 0
                print(
                    f"[vfx]   phase {phase_num}: "
                    f"{i + 1}/{n_frames} effect frames "
                    f"| {fps_rate:.1f} fps | ETA {eta:.0f}s"
                )
                last_log = now
                activity.heartbeat(f"phase {phase_num}: effect frame {i + 1}/{n_frames}")
    finally:
        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()


def _ffmpeg_concat(segment_paths: list[str], output_path: str) -> None:
    """Concatenate FFV1 segments using ffmpeg concat demuxer with stream copy."""
    list_path = output_path + ".concat.txt"
    with open(list_path, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed (exit {result.returncode}): {result.stderr}")
    finally:
        os.remove(list_path)


def _process_video_with_effect(
    input_path: str,
    output_dir: str,
    phase_label: str,
    processor,
    video_info: VideoInfo,
    phase_num: int,
) -> str:
    """Process video by splitting into segments: ffmpeg for passthrough, OpenCV for effects.

    Passthrough segments use ffmpeg stream copy (FFV1 input) or fast transcode,
    avoiding the slow per-frame Python/OpenCV loop for frames without effects.
    """
    lossless_path = os.path.join(output_dir, f"{phase_label}.avi")
    active_intervals = _build_active_intervals(processor, video_info)
    total_frames = video_info.total_frames
    fps = video_info.fps

    # Probe codec once — drives stream-copy vs transcode for all passthrough segments
    input_codec = _probe_video_codec(input_path)
    passthrough_method = "stream copy" if input_codec == "ffv1" else "ffmpeg transcode"

    active_frame_count = sum(e - s for s, e in active_intervals)
    pass_frame_count = total_frames - active_frame_count

    print(f"[vfx]   reading from: {input_path}")
    print(f"[vfx]   writing to:   {lossless_path}")
    print(f"[vfx]   input codec: {input_codec}, passthrough method: {passthrough_method}")
    print(f"[vfx]   total frames: {total_frames}, active: {active_frame_count} ({active_frame_count * 100 // max(total_frames, 1)}%)")
    if active_intervals:
        for s, e in active_intervals:
            print(f"[vfx]     active: frames {s}-{e} ({s / fps:.1f}s - {e / fps:.1f}s)")

    # If no active intervals, just copy/transcode the whole file
    if not active_intervals:
        _ffmpeg_extract_segment(input_path, lossless_path, 0, total_frames, video_info, input_codec)
        print(f"[vfx]   phase {phase_num} done: no active effects, passthrough via {passthrough_method}")
        return lossless_path

    # Build ordered segment list: (start_frame, end_frame, is_active)
    segments: list[tuple[int, int, bool]] = []
    prev_end = 0
    for s, e in active_intervals:
        if s > prev_end:
            segments.append((prev_end, s, False))
        segments.append((s, e, True))
        prev_end = e
    if prev_end < total_frames:
        segments.append((prev_end, total_frames, False))

    print(f"[vfx]   {len(segments)} segments, passthrough via {passthrough_method}")

    start_time = time.time()
    seg_paths: list[str] = []

    for i, (start, end, is_active) in enumerate(segments):
        seg_path = os.path.join(output_dir, f"{phase_label}_seg{i:03d}.avi")
        n_frames = end - start

        if is_active:
            print(f"[vfx]   seg {i}: frames {start}-{end} ({n_frames} frames) — processing effect")
            _process_active_segment(
                input_path, seg_path, start, end,
                processor, video_info, phase_num,
            )
        else:
            print(f"[vfx]   seg {i}: frames {start}-{end} ({n_frames} frames) — {passthrough_method}")
            _ffmpeg_extract_segment(input_path, seg_path, start, end, video_info, input_codec)

        seg_paths.append(seg_path)
        activity.heartbeat(f"phase {phase_num}: segment {i + 1}/{len(segments)}")

    # Concatenate all segments
    if len(seg_paths) == 1:
        os.rename(seg_paths[0], lossless_path)
    else:
        print(f"[vfx]   concatenating {len(seg_paths)} segments...")
        _ffmpeg_concat(seg_paths, lossless_path)
        for p in seg_paths:
            os.remove(p)

    elapsed = time.time() - start_time
    print(
        f"[vfx]   phase {phase_num} done: {total_frames} frames in {elapsed:.1f}s "
        f"({active_frame_count} processed, {pass_frame_count} passthrough via {passthrough_method})"
    )
    return lossless_path
