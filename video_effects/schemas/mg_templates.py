"""Template registry for motion graphics components.

Each template declares its metadata (props, spatial hints, duration range)
and points to a creative-guidance markdown file. The planner assembles its
system prompt dynamically from only the templates listed in
IMPLEMENTED_TEMPLATES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Spec models
# ---------------------------------------------------------------------------

class PropSpec(BaseModel):
    """Describes a single prop that a template accepts."""
    name: str
    type: Literal["str", "int", "float", "bool", "literal", "list", "json"]
    required: bool = True
    default: Any = None
    description: str = ""
    choices: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None


class SpatialHint(BaseModel):
    """Rough guidance on where this template typically lives on screen."""
    typical_y_range: tuple[float, float] = (0.0, 1.0)
    typical_x_range: tuple[float, float] = (0.0, 1.0)
    edge_aligned: bool = False


class MGTemplateSpec(BaseModel):
    """Full metadata for a single motion-graphics template."""
    name: str
    display_name: str
    description: str
    props: list[PropSpec]
    duration_range: tuple[float, float] = (1.0, 10.0)
    spatial: SpatialHint = Field(default_factory=SpatialHint)
    guidance_file: str = ""


# ---------------------------------------------------------------------------
# Template specs
# ---------------------------------------------------------------------------

ANIMATED_TITLE = MGTemplateSpec(
    name="animated_title",
    display_name="Animated Title",
    description="Animated text overlay. Use for section headers, key statements, or emphasis.",
    props=[
        PropSpec(
            name="text",
            type="str",
            required=True,
            description="The text to display",
        ),
        PropSpec(
            name="style",
            type="literal",
            required=False,
            default="fade",
            description="Animation style",
            choices=["fade", "slide-in", "typewriter", "bounce"],
        ),
        PropSpec(
            name="fontSize",
            type="int",
            required=False,
            default=64,
            description="Font size in pixels",
            min_value=24,
            max_value=96,
        ),
        PropSpec(
            name="color",
            type="str",
            required=False,
            default="#FFFFFF",
            description="CSS hex color",
        ),
        PropSpec(
            name="fontWeight",
            type="str",
            required=False,
            default="700",
            description="Font weight (400-900)",
        ),
    ],
    duration_range=(2.0, 5.0),
    spatial=SpatialHint(
        typical_y_range=(0.05, 0.25),
        typical_x_range=(0.1, 0.9),
    ),
    guidance_file="animated_title.md",
)


LOWER_THIRD = MGTemplateSpec(
    name="lower_third",
    display_name="Lower Third",
    description="Name/title card with accent bar. Use to introduce the speaker or label topics.",
    props=[
        PropSpec(
            name="name",
            type="str",
            required=True,
            description="Primary text (speaker name or topic label)",
        ),
        PropSpec(
            name="title",
            type="str",
            required=False,
            description="Secondary text (job title, subtitle)",
        ),
        PropSpec(
            name="accentColor",
            type="str",
            required=False,
            default="#FFD700",
            description="CSS hex color for the accent bar",
        ),
        PropSpec(
            name="style",
            type="literal",
            required=False,
            default="slide",
            description="Animation style",
            choices=["slide", "fade"],
        ),
        PropSpec(
            name="fontSize",
            type="int",
            required=False,
            default=36,
            description="Font size in pixels",
            min_value=20,
            max_value=56,
        ),
        PropSpec(
            name="color",
            type="str",
            required=False,
            default="#FFFFFF",
            description="CSS hex color for text",
        ),
    ],
    duration_range=(3.0, 6.0),
    spatial=SpatialHint(
        typical_y_range=(0.75, 0.88),
        typical_x_range=(0.03, 0.45),
        edge_aligned=True,
    ),
    guidance_file="lower_third.md",
)


LISTICLE = MGTemplateSpec(
    name="listicle",
    display_name="Listicle",
    description="Staggered list of items that appear one by one. Use when the speaker lists things, compares items, or presents steps.",
    props=[
        PropSpec(
            name="items",
            type="list",
            required=True,
            description="List of text items to display",
            max_value=5,
        ),
        PropSpec(
            name="style",
            type="literal",
            required=False,
            default="pop",
            description="Animation style for each item",
            choices=["pop", "slide"],
        ),
        PropSpec(
            name="listStyle",
            type="literal",
            required=False,
            default="numbered",
            description="List marker style",
            choices=["numbered", "bullet", "none"],
        ),
        PropSpec(
            name="staggerDelay",
            type="int",
            required=False,
            default=10,
            description="Frames between each item reveal",
            min_value=5,
            max_value=30,
        ),
        PropSpec(
            name="fontSize",
            type="int",
            required=False,
            default=32,
            description="Font size in pixels",
            min_value=18,
            max_value=48,
        ),
        PropSpec(
            name="color",
            type="str",
            required=False,
            default="#FFFFFF",
            description="CSS hex color for text",
        ),
        PropSpec(
            name="accentColor",
            type="str",
            required=False,
            default="#FFD700",
            description="CSS hex color for markers/bullets",
        ),
    ],
    duration_range=(3.0, 8.0),
    spatial=SpatialHint(
        typical_y_range=(0.15, 0.75),
        typical_x_range=(0.1, 0.6),
    ),
    guidance_file="listicle.md",
)


DATA_ANIMATION = MGTemplateSpec(
    name="data_animation",
    display_name="Data Animation",
    description="Animated numbers, stats, or bar charts. Use when the speaker mentions specific metrics or data points.",
    props=[
        PropSpec(
            name="style",
            type="literal",
            required=True,
            description="Visualization sub-style",
            choices=["counter", "stat-callout", "bar"],
        ),
        PropSpec(
            name="value",
            type="float",
            required=True,
            description="Primary numeric value to animate to",
        ),
        PropSpec(
            name="label",
            type="str",
            required=True,
            description="Label describing the value",
        ),
        PropSpec(
            name="startValue",
            type="float",
            required=False,
            default=0,
            description="Starting value for counter animation",
        ),
        PropSpec(
            name="suffix",
            type="str",
            required=False,
            description="Text after the number (e.g. '%', 'M', 'users')",
        ),
        PropSpec(
            name="prefix",
            type="str",
            required=False,
            description="Text before the number (e.g. '$', '#')",
        ),
        PropSpec(
            name="delta",
            type="float",
            required=False,
            description="Change indicator (positive = up arrow, negative = down arrow)",
        ),
        PropSpec(
            name="items",
            type="json",
            required=False,
            description="Bar chart items: [{label: string, value: number}] (for bar style)",
        ),
        PropSpec(
            name="fontSize",
            type="int",
            required=False,
            default=48,
            description="Font size in pixels for the main number",
            min_value=24,
            max_value=96,
        ),
        PropSpec(
            name="color",
            type="str",
            required=False,
            default="#FFFFFF",
            description="CSS hex color for text",
        ),
        PropSpec(
            name="accentColor",
            type="str",
            required=False,
            default="#FFD700",
            description="CSS hex color for bars and accents",
        ),
    ],
    duration_range=(2.0, 6.0),
    spatial=SpatialHint(
        typical_y_range=(0.15, 0.75),
        typical_x_range=(0.1, 0.6),
    ),
    guidance_file="data_animation.md",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MG_TEMPLATE_REGISTRY: dict[str, MGTemplateSpec] = {
    "animated_title": ANIMATED_TITLE,
    "lower_third": LOWER_THIRD,
    "listicle": LISTICLE,
    "data_animation": DATA_ANIMATION,
}

IMPLEMENTED_TEMPLATES: set[str] = {"animated_title", "lower_third", "listicle", "data_animation"}

_GUIDANCE_DIR = Path(__file__).resolve().parent.parent / "prompts" / "mg_guidance"


def get_available_templates() -> list[MGTemplateSpec]:
    """Return specs for all implemented templates."""
    return [
        MG_TEMPLATE_REGISTRY[name]
        for name in sorted(IMPLEMENTED_TEMPLATES)
        if name in MG_TEMPLATE_REGISTRY
    ]


def load_guidance(spec: MGTemplateSpec) -> str:
    """Load the creative-guidance markdown for a template.

    Returns empty string if the file doesn't exist.
    """
    if not spec.guidance_file:
        return ""
    path = _GUIDANCE_DIR / spec.guidance_file
    if path.exists():
        return path.read_text().strip()
    return ""
