import logging

from video_effects.core import BaseCapability
from video_effects.effect_registry import EFFECT_PHASES
from video_effects.schemas.effects import EffectCue, EffectType, ValidatedTimeline
from video_effects.skills.effect_planning.schemas import ValidateTimelineRequest, ValidateTimelineResponse


def _validate_zoom_pairs(effects: list[EffectCue], logger: logging.Logger) -> list[EffectCue]:
    zoom_cues = [
        e for e in effects
        if e.effect_type == EffectType.ZOOM and e.zoom_params is not None
    ]
    non_zoom = [e for e in effects if e.effect_type != EffectType.ZOOM or e.zoom_params is None]

    zoom_cues.sort(key=lambda e: e.start_time)

    kept = []
    zoomed_in = False
    last_zoom = 1.5
    dropped = 0

    for cue in zoom_cues:
        action = cue.zoom_params.action
        if action == "out":
            if not zoomed_in:
                logger.warning(
                    f"Dropping orphaned zoom-out at t={cue.start_time:.1f}s (no prior zoom-in)"
                )
                dropped += 1
                continue
            cue.zoom_params.zoom_level = last_zoom
            zoomed_in = False
        elif action == "in":
            if zoomed_in:
                logger.warning(
                    f"Dropping duplicate zoom-in at t={cue.start_time:.1f}s (already zoomed in)"
                )
                dropped += 1
                continue
            zoomed_in = True
            last_zoom = cue.zoom_params.zoom_level
        kept.append(cue)

    if dropped:
        logger.info(f"Zoom pair validation: dropped {dropped} invalid cues")

    return non_zoom + kept


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
        resolved = _validate_zoom_pairs(resolved, self.logger)
        resolved.sort(key=lambda e: (EFFECT_PHASES.get(e.effect_type, 99), e.start_time))
        if conflicts_resolved:
            self.logger.info(f"Resolved {conflicts_resolved} timeline conflicts")
        timeline = ValidatedTimeline(effects=resolved, conflicts_resolved=conflicts_resolved, total_duration=duration)
        return ValidateTimelineResponse(**timeline.model_dump())
