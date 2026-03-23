from video_effects.core import BaseCapability
from video_effects.helpers.studio import stop_studio
from video_effects.skills.studio.schemas import StopStudioRequest, StopStudioResponse


class StopStudioCapability(BaseCapability[StopStudioRequest, StopStudioResponse]):
    async def execute(self, request: StopStudioRequest) -> StopStudioResponse:
        stop_studio(request.pid)
        return StopStudioResponse(stopped=True)
