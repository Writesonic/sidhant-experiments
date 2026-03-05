"""Activities: extract video info and audio track."""

import json
import os
import subprocess

from temporalio import activity

from video_effects.schemas.effects import VideoInfo


@activity.defn(name="vfx_get_video_info")
def get_video_info(video_path: str) -> dict:
    """Extract video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    duration = float(probe["format"].get("duration", 0))
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    info = VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        codec=video_stream.get("codec_name", ""),
        total_frames=int(duration * fps),
        audio_codec=audio_stream.get("codec_name", "") if audio_stream else "",
        has_audio=audio_stream is not None,
        color_space=video_stream.get("color_space", ""),
        color_transfer=video_stream.get("color_transfer", ""),
        color_primaries=video_stream.get("color_primaries", ""),
        pix_fmt=video_stream.get("pix_fmt", ""),
    )
    return info.model_dump()


@activity.defn(name="vfx_extract_audio")
def extract_audio(input_data: dict) -> dict:
    """Extract audio track from video: one copy for transcription, one preserved original.

    Input: {"video_path": str, "output_dir": str}
    Output: {"audio_path": str, "original_audio_path": str}
    """
    video_path = input_data["video_path"]
    output_dir = input_data["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Transcription copy: 16kHz mono WAV (Whisper input format)
    audio_path = os.path.join(output_dir, "audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Preservation copy: stream-copy original audio (no re-encoding)
    original_audio_path = os.path.join(output_dir, "original_audio.aac")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-c:a", "copy",
        original_audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # Fallback: if stream copy fails (e.g. unsupported container), re-encode to AAC
        original_audio_path = os.path.join(output_dir, "original_audio.m4a")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-c:a", "aac", "-b:a", "192k",
            original_audio_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    return {"audio_path": audio_path, "original_audio_path": original_audio_path}
