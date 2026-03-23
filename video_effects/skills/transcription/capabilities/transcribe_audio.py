from video_effects.config import settings
from video_effects.core import BaseCapability
from video_effects.skills.transcription.schemas import TranscribeAudioRequest, TranscribeAudioResponse


class TranscribeAudioCapability(BaseCapability[TranscribeAudioRequest, TranscribeAudioResponse]):
    async def _transcribe_elevenlabs(self, audio_path: str) -> dict:
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
        segments = []
        full_text = result.get("text", "")
        for word_info in result.get("words", []):
            segments.append({
                "text": word_info.get("text", ""),
                "start": word_info.get("start", 0),
                "end": word_info.get("end", 0),
                "type": word_info.get("type", "word"),
            })
        return {"transcript": full_text, "segments": segments}

    def _transcribe_whisper(self, audio_path: str) -> dict:
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
        return {"transcript": result.get("text", ""), "segments": segments}

    async def execute(self, request: TranscribeAudioRequest) -> TranscribeAudioResponse:
        if settings.ELEVENLABS_API_KEY:
            result = await self._transcribe_elevenlabs(request.audio_path)
        else:
            result = self._transcribe_whisper(request.audio_path)

        return TranscribeAudioResponse(**result)
