import logging

from video_effects.schemas.effects import EffectCue, EffectType


def validate_zoom_pairs(effects: list[EffectCue], logger: logging.Logger) -> list[EffectCue]:
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
