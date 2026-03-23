from video_effects.infrastructure import run_capability_sync
from video_effects.skills.registry import register_activity
from video_effects.skills.transcription.capabilities.transcribe_audio import TranscribeAudioCapability
from video_effects.skills.transcription.schemas import TranscribeAudioRequest


@register_activity(name="vfx_transcribe_audio", description="Transcribe audio to text with timestamps")
def vfx_transcribe_audio(input_data: dict) -> dict:
    request = TranscribeAudioRequest(**input_data)
    response = run_capability_sync(TranscribeAudioCapability, request)
    return response.model_dump()
