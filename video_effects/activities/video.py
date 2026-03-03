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
    )
    return info.model_dump()


@activity.defn(name="vfx_extract_audio")
def extract_audio(input_data: dict) -> dict:
    """Extract audio track from video to WAV file.

    Input: {"video_path": str, "output_dir": str}
    Output: {"audio_path": str}
    """
    video_path = input_data["video_path"]
    output_dir = input_data["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    audio_path = os.path.join(output_dir, "audio.wav")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    return {"audio_path": audio_path}
