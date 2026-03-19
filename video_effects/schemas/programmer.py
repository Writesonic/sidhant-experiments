"""Schemas for the programmer code-generation workflow."""

from pydantic import BaseModel, Field


class ProgrammerComponentSpec(BaseModel):
    """Free-form specification for a single visual component."""

    id: str = Field(description="Unique identifier for this component")
    title: str = Field(description="Short display title")
    description: str = Field(description="What to build visually")
    rationale: str = Field(description="Why this matters for the video content")
    visual_approach: str = Field(description="How to build it (SVG paths, div layout, animation style)")
    data: dict = Field(description="Props/data the component needs to render — include ALL values")
    score: int = Field(ge=0, le=100, description="Self-assessed impact score (0-100)")
    start_time: float = Field(description="When to show (seconds)")
    end_time: float = Field(description="When to hide (seconds)")
    bounds: dict = Field(description="Normalized screen region {x, y, w, h} where each value is 0-1")
    anchor: str = Field("static", description="Face-aware anchor mode")


class ProgrammerPlanResponse(BaseModel):
    """LLM response from the brainstorm activity."""

    reasoning: str = Field(description="Creative analysis of the transcript")
    components: list[ProgrammerComponentSpec] = Field(
        description="List of proposed visual components — MUST contain at least one component"
    )


class TemplatePlacement(BaseModel):
    """Where and when to show a single library template."""

    template_id: str = Field(description="Library template ID")
    start_time: float = Field(description="When to show (seconds)")
    end_time: float = Field(description="When to hide (seconds)")
    bounds: dict = Field(description="Normalized {x, y, w, h}")
    props: dict = Field(description="Filled props matching template's PropSpec")
    anchor: str = Field("static", description="Anchor mode")
    z_index: int = Field(10, description="Z-index tier")
    rationale: str = Field(description="Why this moment/content")


class TemplatePlacementResponse(BaseModel):
    """LLM response for context-aware template placement."""

    placements: list[TemplatePlacement]
