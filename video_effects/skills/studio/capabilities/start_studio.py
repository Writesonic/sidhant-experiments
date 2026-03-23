from video_effects.config import settings
from video_effects.core import BaseCapability
from video_effects.helpers.studio import start_studio, write_preview_assets
from video_effects.skills.studio.schemas import StartStudioRequest, StartStudioResponse


class StartStudioCapability(BaseCapability[StartStudioRequest, StartStudioResponse]):
    async def execute(self, request: StartStudioRequest) -> StartStudioResponse:
        write_preview_assets(
            mg_plan=request.mg_plan,
            base_video_path=request.base_video_path,
            face_data_path=request.face_data_path,
            zoom_state_path=request.zoom_state_path,
            video_info=request.video_info,
        )
        result = start_studio(port=getattr(settings, "REMOTION_STUDIO_PORT", 3100))
        return StartStudioResponse(**result)
