import os
import shutil

from video_effects.core import BaseCapability
from video_effects.helpers.remotion import composite_overlay
from video_effects.skills.composition.schemas import (
    CompositeMotionGraphicsRequest,
    CompositeMotionGraphicsResponse,
)


class CompositeMotionGraphicsCapability(
    BaseCapability[CompositeMotionGraphicsRequest, CompositeMotionGraphicsResponse]
):
    async def execute(self, request: CompositeMotionGraphicsRequest) -> CompositeMotionGraphicsResponse:
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        temp_dir = request.temp_dir or os.path.dirname(request.output_path)
        same_file = os.path.abspath(request.base_video) == os.path.abspath(request.output_path)
        actual_output = os.path.join(temp_dir, "mg_composited.mp4") if same_file else request.output_path
        self.heartbeat_sync("Compositing motion graphics overlay")
        composite_overlay(
            base_video=request.base_video,
            overlay_video=request.overlay_video,
            output_path=actual_output,
        )
        if same_file:
            shutil.move(actual_output, request.output_path)
        return CompositeMotionGraphicsResponse(output_video=request.output_path)
