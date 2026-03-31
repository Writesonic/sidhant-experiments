import os

from video_effects.core import BaseCapability
from video_effects.skills.rendering.schemas import ApplyEffectsRequest, ApplyEffectsResponse
from video_effects.skills.rendering.capabilities._pipeline import (
    EFFECT_PROCESSORS, _build_merged_intervals, _process_single_pass, group_by_phase,
)
from video_effects.schemas.effects import EffectCue, VideoInfo


class ApplyEffectsCapability(BaseCapability[ApplyEffectsRequest, ApplyEffectsResponse]):
    async def execute(self, request):
        effects = [EffectCue(**e) for e in request.effects]
        video_info = VideoInfo(**request.video_info)
        os.makedirs(request.output_dir, exist_ok=True)
        if not effects:
            return ApplyEffectsResponse(processed_video=request.video_path, phases_executed=0)
        phase_groups = group_by_phase(effects)
        processors = []
        for phase_num, phase_effects in sorted(phase_groups.items()):
            effect_type = phase_effects[0].effect_type
            processor_cls = EFFECT_PROCESSORS.get(effect_type)
            if processor_cls is None:
                continue
            processor = processor_cls()
            processor.setup(video_info, phase_effects)
            processors.append(processor)
        if not processors:
            return ApplyEffectsResponse(processed_video=request.video_path, phases_executed=0)
        all_intervals = _build_merged_intervals(processors, video_info)
        output_path = os.path.join(request.output_dir, "processed.mp4")
        _process_single_pass(request.video_path, output_path, processors, video_info, all_intervals,
                             heartbeat_fn=self.heartbeat_sync)
        return ApplyEffectsResponse(processed_video=output_path, phases_executed=len(processors))
