"""Activities for the programmer code-generation workflow."""

import json
import logging
import re
from pathlib import Path

from temporalio import activity

from video_effects.config import settings
from video_effects.helpers.llm import call_structured, call_text, load_prompt
from video_effects.helpers.remotion import _get_remotion_dir
from video_effects.schemas.programmer import ProgrammerPlanResponse

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _build_style_guide(style_config: dict | None) -> str:
    """Build a minimal style guide section for the planner."""
    if not style_config:
        return ""
    palette = style_config.get("palette", [])
    lines = ["## Style Guide\n"]

    if isinstance(palette, list) and len(palette) >= 3:
        lines.append(f"**Color palette**: {palette[0]} (text), {palette[1]} (secondary), {palette[2]} (accent)")
    elif isinstance(palette, dict):
        parts = [f"{v} ({k})" for k, v in palette.items() if isinstance(v, str) and v.startswith("#")]
        if parts:
            lines.append(f"**Color palette**: {', '.join(parts)}")

    font = style_config.get("font_family", "")
    if font:
        lines.append(f"**Font**: {font}")

    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


def _build_spatial_user_message(input_data: dict) -> str:
    """Build the user message with full spatial context and transcript."""
    context = input_data["spatial_context"]
    transcript = input_data.get("transcript", "")
    segments = input_data.get("segments", [])

    lines = []
    video = context.get("video", {})
    lines.append("## Video Info")
    lines.append(f"- Duration: {video.get('duration', 0):.1f}s")
    lines.append(f"- Resolution: {video.get('width', '?')}x{video.get('height', '?')}")
    lines.append(f"- FPS: {video.get('fps', 30)}\n")

    # Face windows
    face_windows = context.get("face_windows", [])
    if face_windows:
        lines.append("## Face Position (time windows)")
        for fw in face_windows[:15]:
            fr = fw["face_region"]
            safe_labels = [s["label"] for s in fw.get("safe_regions", [])]
            lines.append(
                f"- [{fw['start_time']:.1f}s - {fw['end_time']:.1f}s] "
                f"face at ({fr['x']:.2f}, {fr['y']:.2f}, {fr['w']:.2f}x{fr['h']:.2f}) safe: {', '.join(safe_labels) or 'none'}"
            )
        lines.append("")

    # Existing OpenCV effects (to complement, not conflict)
    effects = input_data.get("effects", [])
    if effects:
        lines.append("## Existing Video Effects (do NOT overlap these)")
        for e in effects:
            lines.append(f"- [{e.get('start_time', 0):.1f}s - {e.get('end_time', 0):.1f}s] {e.get('effect_type', '?')}")
        lines.append("")

    # Timestamped transcript
    if segments:
        lines.append("## Timestamped Transcript")
        for seg in segments:
            if seg.get("type") == "word":
                lines.append(f"[{seg.get('start', 0):.2f}s] {seg.get('text', '')}")
        lines.append("")

    if transcript:
        lines.append(f"## Full Transcript\n\n{transcript[:3000]}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: Creative Brainstorm
# ---------------------------------------------------------------------------


@activity.defn(name="vfx_programmer_brainstorm")
def programmer_brainstorm(input_data: dict) -> dict:
    """Creative brainstorm: propose unconstrained visual components.

    Input: {
        "spatial_context": dict,
        "transcript": str,
        "segments": list[dict],
        "style_config": dict | None,
        "video_info": dict,
        "effects": list[dict],  # existing OpenCV effects
    }
    Output: {
        "reasoning": str,
        "components": list[dict],  # ProgrammerComponentSpec dicts
    }
    """
    style_config = input_data.get("style_config")

    base_prompt = load_prompt("programmer_brainstorm.md")
    style_guide = _build_style_guide(style_config)
    system_prompt = base_prompt.replace("{STYLE_GUIDE}", style_guide)

    user_message = _build_spatial_user_message(input_data)

    activity.heartbeat("Brainstorming visual components via LLM")

    raw = call_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=ProgrammerPlanResponse,
        model=settings.PROGRAMMER_LLM_MODEL,
        max_tokens=16384,
    )

    components = raw.get("components", [])
    logger.info("Brainstorm proposed %d component(s)", len(components))

    return {
        "reasoning": raw.get("reasoning", ""),
        "components": components,
    }


# ---------------------------------------------------------------------------
# Step 2: Self-Critique + Filter
# ---------------------------------------------------------------------------


@activity.defn(name="vfx_programmer_critique")
def programmer_critique(input_data: dict) -> dict:
    """Self-critique and filter brainstorm proposals.

    Input: {
        "proposals": list[dict],    # ProgrammerComponentSpec dicts from brainstorm
        "spatial_context": dict,
        "transcript": str,
        "video_info": dict,
        "max_specs": int,
    }
    Output: {
        "components": list[dict],  # filtered + scored ProgrammerComponentSpec dicts
    }
    """
    proposals = input_data["proposals"]
    max_specs = input_data.get("max_specs", 6)

    if not proposals:
        return {"components": []}

    base_prompt = load_prompt("programmer_critique.md")
    system_prompt = base_prompt.replace("{MAX_SPECS}", str(max_specs))

    # Build user message with proposals + context
    lines = [
        "## Proposals to Evaluate\n",
        f"```json\n{json.dumps(proposals, indent=2)}\n```\n",
        f"## Constraints",
        f"- Maximum components to keep: {max_specs}",
        f"- Subtitle zone (y >= 0.78) is reserved",
        f"- Minimum 2 second gap between components",
    ]

    # Add spatial context summary for overlap checking
    context = input_data.get("spatial_context", {})
    face_windows = context.get("face_windows", [])
    if face_windows:
        lines.append("\n## Face Windows (for overlap checking)")
        for fw in face_windows[:10]:
            fr = fw["face_region"]
            lines.append(
                f"- [{fw['start_time']:.1f}s - {fw['end_time']:.1f}s] "
                f"face at ({fr['x']:.2f}, {fr['y']:.2f}, {fr['w']:.2f}x{fr['h']:.2f})"
            )

    user_message = "\n".join(lines)

    activity.heartbeat("Critiquing proposals via LLM")

    raw = call_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=ProgrammerPlanResponse,
        model=settings.PROGRAMMER_LLM_MODEL,
        max_tokens=16384,
    )

    components = raw.get("components", [])

    # Sort by score descending, enforce max_specs cap
    components.sort(key=lambda c: c.get("score", 50), reverse=True)
    components = components[:max_specs]

    logger.info(
        "Critique kept %d of %d proposals (max %d)",
        len(components), len(proposals), max_specs,
    )

    return {"components": components}


# ---------------------------------------------------------------------------
# Step 3: Code Generation
# ---------------------------------------------------------------------------


def _build_programmer_codegen_prompt(api_reference: str) -> str:
    """Assemble the code generation system prompt with API ref and examples."""
    base = load_prompt("programmer_generate_code.md")

    # Load real component examples
    components_dir = _get_remotion_dir() / "src" / "components"
    examples = []
    for name in ("DataAnimation.tsx", "AnimatedTitle.tsx"):
        path = components_dir / name
        if path.exists():
            examples.append(f"### Example: {name}\n\n```tsx\n{path.read_text()}\n```")

    examples_section = "## Real Component Examples\n\n" + "\n\n".join(examples) if examples else ""

    return (
        base
        .replace("{API_REFERENCE}", f"## Allowed Imports (API Reference)\n\n{api_reference}")
        .replace("{EXAMPLES}", examples_section)
    )


@activity.defn(name="vfx_programmer_generate_code")
def programmer_generate_code(input_data: dict) -> dict:
    """Generate TSX source code for ONE programmer component.

    Input: {
        "spec": dict,                # ProgrammerComponentSpec
        "style_config": dict | None,
        "video_info": dict,
        "attempt": int,
        "previous_errors": list[str],
        "previous_code": str,
    }
    Output: {
        "component_id": str,
        "tsx_code": str,
        "export_name": str,
        "props": dict,
    }
    """
    spec = input_data["spec"]
    style_config = input_data.get("style_config")
    video_info = input_data.get("video_info", {})
    attempt = input_data.get("attempt", 1)
    previous_errors = input_data.get("previous_errors", [])
    previous_code = input_data.get("previous_code", "")

    api_reference = load_prompt("infographic_api_reference.md")
    system_prompt = _build_programmer_codegen_prompt(api_reference)

    # Build component ID and export name
    # Use same "Ig" prefix as _derive_export_name in infographic.py so the
    # registry rebuild (which rescans all .tsx files) derives matching names.
    component_id = spec["id"]
    export_name = "".join(word.capitalize() for word in component_id.replace("-", "_").split("_"))
    if export_name and export_name[0].isdigit():
        export_name = "Ig" + export_name

    lines = [
        "## Component Spec",
        f"- Component ID: `{component_id}`",
        f"- Export name: `{export_name}`",
        f"- Title: {spec['title']}",
        f"- Description: {spec['description']}",
        f"- Rationale: {spec['rationale']}",
        f"- Visual Approach: {spec['visual_approach']}",
        f"- Duration: {spec['end_time'] - spec['start_time']:.1f}s",
        f"- Video: {video_info.get('width', 1920)}x{video_info.get('height', 1080)} @ {video_info.get('fps', 30)}fps",
        "",
        "## Data (passed as props)",
        "```json",
        json.dumps(spec.get("data", {}), indent=2),
        "```",
    ]

    if style_config:
        palette = style_config.get("palette", [])
        if palette:
            lines.append(f"\n## Style")
            lines.append(f"Use useStyle() for colors, but for reference the palette is: {palette}")

    if attempt > 1 and previous_errors:
        lines.append(f"\n## RETRY (attempt {attempt})")
        lines.append("The previous code had these errors:")
        for err in previous_errors:
            lines.append(f"- {err}")
        if previous_code:
            lines.append(f"\n## Previous Code (fix these issues)\n\n```tsx\n{previous_code}\n```")

    user_message = "\n".join(lines)

    activity.heartbeat(f"Generating code for {component_id} (attempt {attempt})")

    raw_code = call_text(
        system_prompt=system_prompt,
        user_message=user_message,
        model=settings.PROGRAMMER_LLM_MODEL,
    )

    # Strip markdown fencing if present
    tsx_code = raw_code.strip()
    if tsx_code.startswith("```"):
        tsx_code = re.sub(r"^```\w*\n?", "", tsx_code)
        tsx_code = re.sub(r"\n?```$", "", tsx_code)

    props = dict(spec.get("data", {}))

    return {
        "component_id": component_id,
        "tsx_code": tsx_code,
        "export_name": export_name,
        "props": props,
    }
