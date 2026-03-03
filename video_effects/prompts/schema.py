"""Pydantic schema for LLM structured output (effect cue parsing)."""

from pydantic import BaseModel, Field

from video_effects.schemas.effects import EffectCue


class ParsedEffectCues(BaseModel):
    """Output schema for the LLM effect cue parser."""

    effects: list[EffectCue] = Field(
        default_factory=list,
        description="List of detected effect cues from the transcript",
    )
    reasoning: str = Field(
        "",
        description="Brief explanation of why these effects were chosen",
    )
