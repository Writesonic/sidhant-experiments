import os

from video_effects.core import BaseCapability
from video_effects.skills.rendering.schemas import SetupProcessorsRequest, SetupProcessorsResponse
from video_effects.skills.rendering.capabilities._pipeline import EFFECT_PROCESSORS, group_by_phase
from video_effects.schemas.effects import EffectCue, VideoInfo


class SetupProcessorsCapability(BaseCapability[SetupProcessorsRequest, SetupProcessorsResponse]):
    async def execute(self, request):
        effects = [EffectCue(**e) for e in request.effects]
        video_info = VideoInfo(**request.video_info)
        os.makedirs(request.cache_dir, exist_ok=True)
        phase_groups = group_by_phase(effects)
        setup_summary = []
        for phase_num, phase_effects in sorted(phase_groups.items()):
            effect_type = phase_effects[0].effect_type
            processor_cls = EFFECT_PROCESSORS.get(effect_type)
            if processor_cls is None:
                continue
            processor = processor_cls()
            processor.setup(video_info, phase_effects, cache_dir=request.cache_dir, video_path=request.video_path)
            setup_summary.append({"phase": phase_num, "effect_type": effect_type.value})
            self.heartbeat_sync(f"setup phase {phase_num}")
        return SetupProcessorsResponse(setup_summary=setup_summary, processors_ready=True)
