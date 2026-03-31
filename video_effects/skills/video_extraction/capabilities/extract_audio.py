import os
import subprocess

from video_effects.core import BaseCapability
from video_effects.skills.video_extraction.schemas import ExtractAudioRequest, ExtractAudioResponse


class ExtractAudioCapability(BaseCapability[ExtractAudioRequest, ExtractAudioResponse]):
    async def execute(self, request: ExtractAudioRequest) -> ExtractAudioResponse:
        os.makedirs(request.output_dir, exist_ok=True)

        # Transcription copy: 16kHz mono WAV (Whisper input format)
        audio_path = os.path.join(request.output_dir, "audio.wav")
        cmd = [
            "ffmpeg", "-y", "-i", request.video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            audio_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # Preservation copy: stream-copy original audio (no re-encoding)
        original_audio_path = os.path.join(request.output_dir, "original_audio.aac")
        cmd = [
            "ffmpeg", "-y", "-i", request.video_path,
            "-vn", "-c:a", "copy",
            original_audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            original_audio_path = os.path.join(request.output_dir, "original_audio.m4a")
            cmd = [
                "ffmpeg", "-y", "-i", request.video_path,
                "-vn", "-c:a", "aac", "-b:a", "192k",
                original_audio_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

        return ExtractAudioResponse(
            audio_path=audio_path,
            original_audio_path=original_audio_path,
        )
