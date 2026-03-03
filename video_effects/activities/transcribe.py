"""Activity: transcribe audio to text with timestamps."""

import json
import logging
import subprocess

from temporalio import activity

from video_effects.config import settings

logger = logging.getLogger(__name__)


@activity.defn(name="vfx_transcribe_audio")
async def transcribe_audio(input_data: dict) -> dict:
    """Transcribe audio file to text with word-level timestamps.

    Input: {"audio_path": str}
    Output: {"transcript": str, "segments": list[dict]}

    Uses ElevenLabs API if available, falls back to local Whisper.
    """
    audio_path = input_data["audio_path"]

    if settings.ELEVENLABS_API_KEY:
        return await _transcribe_elevenlabs(audio_path)
    else:
        return _transcribe_whisper(audio_path)


async def _transcribe_elevenlabs(audio_path: str) -> dict:
    """Transcribe via ElevenLabs Speech-to-Text API."""
    import httpx

    async with httpx.AsyncClient(timeout=120) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": settings.ELEVENLABS_API_KEY},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"model_id": "scribe_v1"},
            )
            response.raise_for_status()

    result = response.json()

    # Build segments from ElevenLabs response
    segments = []
    full_text = result.get("text", "")

    for word_info in result.get("words", []):
        segments.append({
            "text": word_info.get("text", ""),
            "start": word_info.get("start", 0),
            "end": word_info.get("end", 0),
            "type": word_info.get("type", "word"),
        })

    return {
        "transcript": full_text,
        "segments": segments,
    }


def _transcribe_whisper(audio_path: str) -> dict:
    """Transcribe via local Whisper (fallback)."""
    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "Neither ELEVENLABS_API_KEY is set nor whisper is installed. "
            "Install whisper: pip install openai-whisper"
        )

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)

    segments = []
    for seg in result.get("segments", []):
        for word in seg.get("words", []):
            segments.append({
                "text": word["word"].strip(),
                "start": word["start"],
                "end": word["end"],
                "type": "word",
            })

    return {
        "transcript": result.get("text", ""),
        "segments": segments,
    }
