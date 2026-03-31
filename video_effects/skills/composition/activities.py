from video_effects.infrastructure import run_capability_sync
from video_effects.skills.composition.capabilities.compose_final import ComposeFinalCapability
from video_effects.skills.composition.capabilities.composite_motion_graphics import (
    CompositeMotionGraphicsCapability,
)
from video_effects.skills.composition.schemas import (
    ComposeFinalRequest,
    CompositeMotionGraphicsRequest,
)
from video_effects.skills.registry import register_activity


@register_activity(name="vfx_compose_final", description="Mux processed video with original audio")
def vfx_compose_final(input_data: dict) -> dict:
    request = ComposeFinalRequest(**input_data)
    response = run_capability_sync(ComposeFinalCapability, request)
    return response.model_dump()


@register_activity(name="vfx_composite_motion_graphics", description="Composite transparent overlay onto base video")
def vfx_composite_motion_graphics(input_data: dict) -> dict:
    request = CompositeMotionGraphicsRequest(**input_data)
    response = run_capability_sync(CompositeMotionGraphicsCapability, request)
    return response.model_dump()
