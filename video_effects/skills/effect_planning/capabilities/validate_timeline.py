from video_effects.core import BaseCapability
from video_effects.effect_registry import EFFECT_PHASES
from video_effects.helpers.effects import validate_zoom_pairs
from video_effects.schemas.effects import EffectCue, EffectType, ValidatedTimeline
from video_effects.skills.effect_planning.schemas import ValidateTimelineRequest, ValidateTimelineResponse


class ValidateTimelineCapability(BaseCapability[ValidateTimelineRequest, ValidateTimelineResponse]):
    async def execute(self, request: ValidateTimelineRequest) -> ValidateTimelineResponse:
        raw_effects = [EffectCue(**e) for e in request.effects]
        duration = request.duration
        for effect in raw_effects:
            effect.start_time = max(0, effect.start_time)
            if duration > 0:
                effect.end_time = min(duration, effect.end_time)
            if effect.end_time <= effect.start_time:
                effect.end_time = effect.start_time + 1.0
        effects = [e for e in raw_effects if e.confidence >= 0.3]
        removed_low_conf = len(raw_effects) - len(effects)
        if removed_low_conf:
            self.logger.info(f"Removed {removed_low_conf} low-confidence effects")
        conflicts_resolved = 0
        resolved = []
        by_type: dict[EffectType, list[EffectCue]] = {}
        for e in effects:
            by_type.setdefault(e.effect_type, []).append(e)
        for effect_type, type_effects in by_type.items():
            type_effects.sort(key=lambda e: e.start_time)
            kept: list[EffectCue] = []
            for effect in type_effects:
                overlapping = [
                    k for k in kept
                    if k.start_time < effect.end_time and effect.start_time < k.end_time
                ]
                if overlapping:
                    for overlap in overlapping:
                        if effect.confidence > overlap.confidence:
                            kept.remove(overlap)
                            kept.append(effect)
                            conflicts_resolved += 1
                        else:
                            conflicts_resolved += 1
                else:
                    kept.append(effect)
            resolved.extend(kept)
        resolved = validate_zoom_pairs(resolved, self.logger)
        resolved.sort(key=lambda e: (EFFECT_PHASES.get(e.effect_type, 99), e.start_time))
        if conflicts_resolved:
            self.logger.info(f"Resolved {conflicts_resolved} timeline conflicts")
        timeline = ValidatedTimeline(effects=resolved, conflicts_resolved=conflicts_resolved, total_duration=duration)
        return ValidateTimelineResponse(**timeline.model_dump())
