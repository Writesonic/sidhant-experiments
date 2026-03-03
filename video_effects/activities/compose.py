"""Activity: final composition — mux processed video with original audio."""

import logging
import os
import subprocess

from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn(name="vfx_compose_final")
def compose_final(input_data: dict) -> dict:
    """Mux processed video with original audio to produce final output.

    Input: {
        "processed_video": str,
        "audio_path": str,
        "output_path": str,
        "has_audio": bool,
    }
    Output: {"output_video": str}
    """
    processed_video = input_data["processed_video"]
    audio_path = input_data.get("audio_path", "")
    output_path = input_data["output_path"]
    has_audio = input_data.get("has_audio", True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if has_audio and audio_path and os.path.exists(audio_path):
        # Mux video + audio
        cmd = [
            "ffmpeg", "-y",
            "-i", processed_video,
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path,
        ]
    else:
        # No audio — just re-encode video
        cmd = [
            "ffmpeg", "-y",
            "-i", processed_video,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-an",
            output_path,
        ]

    logger.info(f"Composing final video: {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg composition failed: {result.stderr}")

    logger.info(f"Final video written to {output_path}")
    return {"output_video": output_path}
