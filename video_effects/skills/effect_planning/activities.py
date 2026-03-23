from video_effects.infrastructure import run_capability_sync
from video_effects.skills.registry import register_activity
from video_effects.skills.effect_planning.capabilities.parse_effect_cues import ParseEffectCuesCapability
from video_effects.skills.effect_planning.capabilities.validate_timeline import ValidateTimelineCapability
from video_effects.skills.effect_planning.schemas import (
    ParseEffectCuesRequest,
    ValidateTimelineRequest,
)


@register_activity(name="vfx_parse_effect_cues", description="Parse effect cues from transcript using LLM")
def vfx_parse_effect_cues(input_data: dict) -> dict:
    request = ParseEffectCuesRequest(**input_data)
    response = run_capability_sync(ParseEffectCuesCapability, request)
    return response.model_dump()


@register_activity(name="vfx_validate_timeline", description="Validate and resolve conflicts in effect timeline")
def vfx_validate_timeline(input_data: dict) -> dict:
    request = ValidateTimelineRequest(**input_data)
    response = run_capability_sync(ValidateTimelineCapability, request)
    return response.model_dump()
