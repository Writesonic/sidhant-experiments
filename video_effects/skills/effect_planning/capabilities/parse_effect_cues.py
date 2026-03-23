from video_effects.core import BaseCapability
from video_effects.helpers.llm import call_structured, load_prompt
from video_effects.prompts.schema import ParsedEffectCues
from video_effects.schemas.styles import get_style
from video_effects.skills.effect_planning.schemas import ParseEffectCuesRequest, ParseEffectCuesResponse


def _build_effects_style_guide(style_config: dict | None, style_preset_name: str = "") -> str:
    if not style_preset_name and not style_config:
        return ""
    preset = get_style(style_preset_name) if style_preset_name else None
    if not preset or preset.name == "default":
        if not style_config:
            return ""
        if not style_preset_name:
            return ""
    lines = ["## Style Context\n"]
    lines.append(f"**Style: {preset.display_name}** — {preset.description}\n")
    lines.append(f"**Effect density**: {preset.density_label} — target {preset.density_range[0]}-{preset.density_range[1]} effects per 60 seconds.")
    if preset.color_grading_preset:
        lines.append(
            f"**Base color grading**: {preset.color_grading_preset} at {preset.color_grading_intensity:.0%} "
            f"is already applied to the full video. Only add color_change effects for tonal shifts that differ from this base."
        )
    else:
        lines.append("**No base color grading** — feel free to add color_change effects where appropriate.")
    if preset.preferred_animations:
        energy = "high" if "bounce" in preset.preferred_animations or "pop" in preset.preferred_animations else "low"
        if energy == "high":
            lines.append("**Energy**: High — use snap/overshoot easing, more frequent zooms, dramatic zoom levels.")
        else:
            lines.append("**Energy**: Calm — use smooth easing, subtle zoom levels (1.2-1.3), fewer effects.")
    if preset.preferred_effects:
        lines.append(f"**Preferred effects**: Use {', '.join(preset.preferred_effects)} when the content warrants it.")
    if preset.avoided_effects:
        lines.append(f"**Avoid**: Do NOT use {', '.join(preset.avoided_effects)} unless explicitly requested.")
    lines.append("")
    return "\n".join(lines)


class ParseEffectCuesCapability(BaseCapability[ParseEffectCuesRequest, ParseEffectCuesResponse]):
    async def execute(self, request: ParseEffectCuesRequest) -> ParseEffectCuesResponse:
        prompt_name = "parse_effect_cues_dev.md" if request.dev_mode else "parse_effect_cues.md"
        system_prompt = load_prompt(prompt_name)
        style_guide = _build_effects_style_guide(request.style_config, request.style_preset_name)
        system_prompt = system_prompt.replace("{STYLE_GUIDE}", style_guide)

        lines = []
        if request.feedback:
            lines.append("## IMPORTANT: Previous attempt was rejected by the user")
            lines.append(f"Feedback: {request.feedback}")
            lines.append("Please adjust your effect choices based on this feedback.\n")
        lines.append(f"Video duration: {request.duration:.1f} seconds\n")
        lines.append("## Timestamped Transcript\n")
        for seg in request.segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            if seg.get("type") == "word":
                lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")
        lines.append(f"\n## Full Transcript\n\n{request.transcript}")
        user_message = "\n".join(lines)

        self.logger.info(f"Sending transcript to LLM for effect cue parsing ({len(request.transcript)} chars)")
        result = call_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=ParsedEffectCues,
            max_tokens=8192,
        )
        effects = result.get("effects", [])
        reasoning = result.get("reasoning", "")
        self.logger.info(f"LLM found {len(effects)} effect cues: {reasoning}")
        return ParseEffectCuesResponse(effects=effects, reasoning=reasoning)
