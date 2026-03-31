from video_effects.core import BaseCapability
from video_effects.skills.rendering.schemas import PrepareRenderRequest, PrepareRenderResponse
from video_effects.skills.rendering.capabilities._pipeline import (
    EFFECT_PROCESSORS, _build_merged_intervals, _is_hdr, _probe_decoded_size, group_by_phase,
)
from video_effects.schemas.effects import EffectCue, VideoInfo


class PrepareRenderCapability(BaseCapability[PrepareRenderRequest, PrepareRenderResponse]):
    async def execute(self, request):
        effects = [EffectCue(**e) for e in request.effects]
        video_info = VideoInfo(**request.video_info)
        if not effects:
            return PrepareRenderResponse()
        decoded_width, decoded_height = _probe_decoded_size(request.video_path)
        hdr = _is_hdr(video_info)
        phase_groups = group_by_phase(effects)
        phase_summary = []
        processors = []
        for phase_num, phase_effects in sorted(phase_groups.items()):
            effect_type = phase_effects[0].effect_type
            processor_cls = EFFECT_PROCESSORS.get(effect_type)
            if processor_cls is None:
                continue
            processor = processor_cls()
            processor.set_cues(phase_effects)
            processors.append(processor)
            phase_summary.append({"phase": phase_num, "effect_type": effect_type.value, "count": len(phase_effects)})
        active_intervals = _build_merged_intervals(processors, video_info)
        active_frame_count = sum(e - s for s, e in active_intervals)
        return PrepareRenderResponse(
            decoded_width=decoded_width, decoded_height=decoded_height, is_hdr=hdr,
            phase_summary=phase_summary, active_intervals=active_intervals,
            active_frame_count=active_frame_count, total_phases=len(processors), has_effects=len(processors) > 0,
        )
