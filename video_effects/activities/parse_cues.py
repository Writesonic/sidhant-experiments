"""Activity: LLM-powered effect cue extraction from transcript."""

import logging

from temporalio import activity

from video_effects.helpers.llm import call_structured, load_prompt
from video_effects.prompts.schema import ParsedEffectCues

logger = logging.getLogger(__name__)


@activity.defn(name="vfx_parse_effect_cues")
async def parse_effect_cues(input_data: dict) -> dict:
    """Parse effect cues from transcript using LLM.

    Input: {"transcript": str, "segments": list[dict], "duration": float}
    Output: {"effects": list[dict], "reasoning": str}
    """
    transcript = input_data["transcript"]
    segments = input_data["segments"]
    duration = input_data.get("duration", 0)

    system_prompt = load_prompt("parse_effect_cues.md")

    # Build user message with timestamped transcript
    lines = []
    lines.append(f"Video duration: {duration:.1f} seconds\n")
    lines.append("## Timestamped Transcript\n")

    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        if seg.get("type") == "word":
            lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")

    lines.append(f"\n## Full Transcript\n\n{transcript}")

    user_message = "\n".join(lines)

    logger.info(f"Sending transcript to LLM for effect cue parsing ({len(transcript)} chars)")

    result = call_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=ParsedEffectCues,
    )

    effects = result.get("effects", [])
    reasoning = result.get("reasoning", "")

    logger.info(f"LLM found {len(effects)} effect cues: {reasoning}")

    return {
        "effects": effects,
        "reasoning": reasoning,
    }
