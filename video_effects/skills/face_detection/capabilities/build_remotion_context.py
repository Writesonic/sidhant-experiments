from video_effects.core import BaseCapability
from video_effects.skills.face_detection.schemas import BuildRemotionContextRequest, BuildRemotionContextResponse


class BuildRemotionContextCapability(BaseCapability[BuildRemotionContextRequest, BuildRemotionContextResponse]):
    async def execute(self, request: BuildRemotionContextRequest) -> BuildRemotionContextResponse:
        from video_effects.activities.remotion import build_remotion_context as _legacy_build
        result = _legacy_build(request.model_dump())
        return BuildRemotionContextResponse(context=result)
