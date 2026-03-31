from video_effects.infrastructure import run_capability_sync
from video_effects.skills.registry import register_activity
from video_effects.skills.studio.capabilities.start_studio import StartStudioCapability
from video_effects.skills.studio.capabilities.stop_studio import StopStudioCapability
from video_effects.skills.studio.capabilities.update_studio_preview import UpdateStudioPreviewCapability
from video_effects.skills.studio.schemas import (
    StartStudioRequest,
    StopStudioRequest,
    UpdateStudioPreviewRequest,
)


@register_activity(name="vfx_start_studio", description="Start Remotion Studio for preview")
def vfx_start_studio(input_data: dict) -> dict:
    request = StartStudioRequest(**input_data)
    response = run_capability_sync(StartStudioCapability, request)
    return response.model_dump()


@register_activity(name="vfx_stop_studio", description="Stop Remotion Studio")
def vfx_stop_studio(input_data: dict) -> dict:
    request = StopStudioRequest(**input_data)
    response = run_capability_sync(StopStudioCapability, request)
    return response.model_dump()


@register_activity(name="vfx_update_studio_preview", description="Update Studio preview plan")
def vfx_update_studio_preview(input_data: dict) -> dict:
    request = UpdateStudioPreviewRequest(**input_data)
    response = run_capability_sync(UpdateStudioPreviewCapability, request)
    return response.model_dump()
