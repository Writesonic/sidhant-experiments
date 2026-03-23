from video_effects.infrastructure import run_capability_sync
from video_effects.skills.registry import register_activity
from video_effects.skills.video_extraction.capabilities.extract_audio import ExtractAudioCapability
from video_effects.skills.video_extraction.capabilities.get_video_info import GetVideoInfoCapability
from video_effects.skills.video_extraction.schemas import (
    ExtractAudioRequest,
    GetVideoInfoRequest,
)


@register_activity(name="vfx_get_video_info", description="Extract video metadata via ffprobe")
def vfx_get_video_info(video_path: str) -> dict:
    response = run_capability_sync(GetVideoInfoCapability, GetVideoInfoRequest(video_path=video_path))
    return response.video_info.model_dump()


@register_activity(name="vfx_extract_audio", description="Extract audio track from video")
def vfx_extract_audio(input_data: dict) -> dict:
    request = ExtractAudioRequest(**input_data)
    response = run_capability_sync(ExtractAudioCapability, request)
    return response.model_dump()
