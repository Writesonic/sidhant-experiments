from video_effects.core import BaseCapability
from video_effects.helpers.studio import update_preview_plan
from video_effects.skills.studio.schemas import UpdateStudioPreviewRequest, UpdateStudioPreviewResponse


class UpdateStudioPreviewCapability(BaseCapability[UpdateStudioPreviewRequest, UpdateStudioPreviewResponse]):
    async def execute(self, request: UpdateStudioPreviewRequest) -> UpdateStudioPreviewResponse:
        update_preview_plan(request.mg_plan, request.video_info)
        return UpdateStudioPreviewResponse(updated=True)
