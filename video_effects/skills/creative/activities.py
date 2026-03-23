from video_effects.infrastructure import run_capability_sync
from video_effects.skills.creative.capabilities.design_style import DesignStyleCapability
from video_effects.skills.creative.schemas import DesignStyleRequest
from video_effects.skills.registry import register_activity


@register_activity(name="vfx_design_style", description="Pick and customize style preset via LLM")
def vfx_design_style(input_data: dict) -> dict:
    request = DesignStyleRequest(**input_data)
    response = run_capability_sync(DesignStyleCapability, request)
    return response.model_dump()
