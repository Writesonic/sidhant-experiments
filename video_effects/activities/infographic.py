"""Activities for the infographic code-generation workflow (A1-A4)."""

import json
import logging
import re
import subprocess
import threading
from pathlib import Path

from temporalio import activity

from video_effects.config import settings
from video_effects.helpers.llm import call_structured, call_text, load_prompt
from video_effects.helpers.remotion import render_still, _get_remotion_dir
from video_effects.schemas import template_library
from video_effects.schemas.infographic import (
    InfographicPlanResponse,
)

logger = logging.getLogger(__name__)

_REGISTRY_LOCK = threading.Lock()

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
        # Fallback: LLM returned a dict palette
        parts = [f"{v} ({k})" for k, v in palette.items() if isinstance(v, str) and v.startswith("#")]
        if parts:
            lines.append(f"**Color palette**: {', '.join(parts)}")

    font = style_config.get("font_family", "")
    if font:
        lines.append(f"**Font**: {font}")

    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


# ---------------------------------------------------------------------------
# A1: Plan infographics
# ---------------------------------------------------------------------------


@activity.defn(name="vfx_cleanup_generated")
def cleanup_generated(input_data: dict) -> dict:
    """Remove all previously generated infographic components.

    Input: {} (no params needed)
    Output: {"cleaned": int}
    """
    remotion_dir = _get_remotion_dir()
    generated_dir = remotion_dir / "src" / "components" / "generated"

    cleaned = 0
    if generated_dir.exists():
        for f in generated_dir.iterdir():
            if f.name != ".gitignore":
                f.unlink()
                cleaned += 1

    # Always write an empty registry so live imports don't break
    generated_dir.mkdir(parents=True, exist_ok=True)
    _rebuild_registry(generated_dir)

    logger.info("Cleaned %d generated files", cleaned)
    return {"cleaned": cleaned}


def _build_user_message(input_data: dict) -> str:
    """Build the shared user message for all planning activities."""
    context = input_data["spatial_context"]
    transcript = input_data.get("transcript", "")
    segments = input_data.get("segments", [])

    lines = []
    video = context.get("video", {})
    lines.append(f"## Video Info")
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


def _plan_category(category: str, prompt_filename: str, input_data: dict) -> dict:
    """Shared planner logic for all component categories.

    Loads the category-specific prompt, builds the user message,
    calls the LLM, and returns the planning result.
    """
    style_config = input_data.get("style_config")

    base_prompt = load_prompt(prompt_filename)
    style_guide = _build_style_guide(style_config)
    system_prompt = base_prompt.replace("{STYLE_GUIDE}", style_guide)

    user_message = _build_user_message(input_data)

    activity.heartbeat(f"Planning {category} via LLM")

    raw = call_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=InfographicPlanResponse,
        model=settings.INFOGRAPHIC_LLM_MODEL,
    )

    infographics = raw.get("infographics", [])
    logger.info("LLM planned %d %s component(s)", len(infographics), category)

    return {
        "infographics": infographics,
        "reasoning": raw.get("reasoning", ""),
    }


@activity.defn(name="vfx_plan_infographics")
def plan_infographics(input_data: dict) -> dict:
    """LLM analyzes transcript and decides WHAT infographics to create.

    Input: {
        "spatial_context": dict,
        "transcript": str,
        "segments": list[dict],
        "style_config": dict | None,
        "video_fps": int,
    }
    Output: {
        "infographics": list[dict],  # list of InfographicSpec dicts
        "reasoning": str,
    }
    """
    return _plan_category("infographics", "plan_infographics.md", input_data)


@activity.defn(name="vfx_plan_diagrams")
def plan_diagrams(input_data: dict) -> dict:
    """LLM analyzes transcript for diagram opportunities (flowcharts, mind maps, etc.)."""
    return _plan_category("diagrams", "plan_diagrams.md", input_data)


@activity.defn(name="vfx_plan_timelines")
def plan_timelines(input_data: dict) -> dict:
    """LLM analyzes transcript for timeline/journey opportunities."""
    return _plan_category("timelines", "plan_timelines.md", input_data)


@activity.defn(name="vfx_plan_quotes")
def plan_quotes(input_data: dict) -> dict:
    """LLM analyzes transcript for key quotes and callout opportunities."""
    return _plan_category("quotes", "plan_quotes.md", input_data)


@activity.defn(name="vfx_plan_code_blocks")
def plan_code_blocks(input_data: dict) -> dict:
    """LLM analyzes transcript for code snippet opportunities."""
    return _plan_category("code_blocks", "plan_code_blocks.md", input_data)


@activity.defn(name="vfx_plan_comparisons")
def plan_comparisons(input_data: dict) -> dict:
    """LLM analyzes transcript for comparison/versus opportunities."""
    return _plan_category("comparisons", "plan_comparisons.md", input_data)


# ---------------------------------------------------------------------------
# A2: Generate infographic code
# ---------------------------------------------------------------------------


def _build_codegen_prompt(api_reference: str) -> str:
    """Assemble the code generation system prompt with API ref and examples."""
    base = load_prompt("generate_infographic_code.md")

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


@activity.defn(name="vfx_generate_infographic_code")
def generate_infographic_code(input_data: dict) -> dict:
    """LLM generates TSX source code for ONE infographic.

    Input: {
        "spec": dict,                # InfographicSpec
        "style_config": dict | None,
        "video_info": dict,          # {width, height, fps, duration}
        "attempt": int,              # 1-based attempt number
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
    system_prompt = _build_codegen_prompt(api_reference)

    # Build user message
    component_id = spec["id"]
    # PascalCase export name from id — must be a valid JS identifier
    export_name = "".join(word.capitalize() for word in component_id.replace("-", "_").split("_"))
    if export_name and export_name[0].isdigit():
        export_name = "Ig" + export_name

    lines = [
        f"## Infographic Spec",
        f"- Component ID: `{component_id}`",
        f"- Export name: `{export_name}`",
        f"- Type: {spec['type']}",
        f"- Title: {spec['title']}",
        f"- Description: {spec['description']}",
        f"- Duration: {spec['end_time'] - spec['start_time']:.1f}s",
        f"- Video: {video_info.get('width', 1920)}x{video_info.get('height', 1080)} @ {video_info.get('fps', 30)}fps",
        f"",
        f"## Data to Visualize",
        f"```json",
        json.dumps(spec.get("data", {}), indent=2),
        f"```",
    ]

    if style_config:
        palette = style_config.get("palette", [])
        if palette:
            lines.append(f"\n## Style")
            lines.append(f"Use useStyle() for colors, but for reference the palette is: {palette}")

    if attempt > 1 and previous_errors:
        lines.append(f"\n## RETRY (attempt {attempt})")
        lines.append(f"The previous code had these errors:")
        for err in previous_errors:
            lines.append(f"- {err}")
        if previous_code:
            lines.append(f"\n## Previous Code (fix these issues)\n\n```tsx\n{previous_code}\n```")

    user_message = "\n".join(lines)

    activity.heartbeat(f"Generating code for {component_id} (attempt {attempt})")

    raw_code = call_text(
        system_prompt=system_prompt,
        user_message=user_message,
        model=settings.INFOGRAPHIC_LLM_MODEL,
    )

    # Strip markdown fencing if present
    tsx_code = raw_code.strip()
    if tsx_code.startswith("```"):
        # Remove opening fence
        tsx_code = re.sub(r"^```\w*\n?", "", tsx_code)
        # Remove closing fence
        tsx_code = re.sub(r"\n?```$", "", tsx_code)

    # Build props dict from spec data (what gets passed at render time)
    props = dict(spec.get("data", {}))

    return {
        "component_id": component_id,
        "tsx_code": tsx_code,
        "export_name": export_name,
        "props": props,
    }


# ---------------------------------------------------------------------------
# A3: Validate infographic
# ---------------------------------------------------------------------------


@activity.defn(name="vfx_validate_infographic")
def validate_infographic(input_data: dict) -> dict:
    """Write TSX to disk, type-check, and test render.

    Input: {
        "component_id": str,
        "tsx_code": str,
        "export_name": str,
        "props": dict | None,  # component data props for test render
    }
    Output: {
        "valid": bool,
        "errors": list[str],
        "preview_path": str,
    }
    """
    component_id = input_data["component_id"]
    tsx_code = input_data["tsx_code"]
    export_name = input_data["export_name"]
    component_props = input_data.get("props", {})

    remotion_dir = _get_remotion_dir()
    generated_dir = remotion_dir / "src" / "components" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Everything under the lock: write → rebuild registry → type-check → render.
    # This prevents concurrent validations from poisoning each other's registry.
    with _REGISTRY_LOCK:
        # Write component file
        component_path = generated_dir / f"{component_id}.tsx"
        component_path.write_text(tsx_code)

        # Rebuild registry BEFORE type-check so tsc sees a consistent state:
        # all existing .tsx files + the current component.
        _write_temp_registry(generated_dir, component_id, export_name)

        errors: list[str] = []

        # Step 1: TypeScript type-check
        activity.heartbeat(f"Type-checking {component_id}")
        try:
            tsc_result = subprocess.run(
                ["npx", "tsc", "--noEmit", "--pretty", "false"],
                cwd=str(remotion_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if tsc_result.returncode != 0:
                stderr = tsc_result.stdout + tsc_result.stderr
                # Only blame this component for errors in its own file
                component_file = f"{component_id}.tsx"
                for line in stderr.split("\n"):
                    if component_file in line:
                        errors.append(line.strip())
        except subprocess.TimeoutExpired:
            errors.append("TypeScript type-check timed out after 60 seconds")
        except FileNotFoundError:
            errors.append("npx/tsc not found — cannot type-check")

        if errors:
            logger.warning("Type-check failed for %s: %d errors", component_id, len(errors))
            component_path.unlink(missing_ok=True)
            _cleanup_registry(generated_dir)
            return {"valid": False, "errors": errors, "preview_path": ""}

        # Step 2: Test render a still frame
        activity.heartbeat(f"Test-rendering {component_id}")
        preview_path = str(generated_dir / f"{component_id}_preview.png")

        try:
            # Merge component data props with required position prop
            render_props = {
                **component_props,
                "position": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.3},
            }
            test_plan = {
                "components": [{
                    "template": component_id,
                    "startFrame": 0,
                    "durationInFrames": 90,
                    "props": render_props,
                    "bounds": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.3},
                    "zIndex": 1,
                }],
                "colorPalette": [],
                "includeBaseVideo": False,
            }

            render_still(
                composition_id="MotionOverlay",
                frame=30,
                props=test_plan,
                output_path=preview_path,
            )
        except Exception as e:
            err_msg = str(e)
            if len(err_msg) > 500:
                err_msg = err_msg[:500] + "..."
            errors.append(f"Render test failed: {err_msg}")
            component_path.unlink(missing_ok=True)
            _cleanup_registry(generated_dir)
            return {"valid": False, "errors": errors, "preview_path": ""}

    logger.info("Validation passed for %s", component_id)
    return {"valid": True, "errors": [], "preview_path": preview_path}


def _derive_export_name(component_id: str) -> str:
    """Derive PascalCase export name from component_id."""
    name = "".join(
        word.capitalize() for word in component_id.replace("-", "_").split("_")
    )
    if name and name[0].isdigit():
        name = "Ig" + name
    return name


def _rebuild_registry(
    generated_dir: Path,
    ensure_component: tuple[str, str] | None = None,
    ensure_components: dict[str, str] | None = None,
) -> None:
    """Rebuild _registry.ts from all .tsx files in generated/.

    Args:
        ensure_component: Optional (component_id, export_name) to guarantee
            inclusion even if glob hasn't caught up yet.
        ensure_components: Optional mapping of {component_id: export_name}
            to override derived names (used for library templates whose
            export names don't match their filenames).
    """
    registry_path = generated_dir / "_registry.ts"

    components: dict[str, str] = {}
    for tsx_file in generated_dir.glob("*.tsx"):
        if tsx_file.name.startswith("_"):
            continue
        cid = tsx_file.stem
        components[cid] = _derive_export_name(cid)

    if ensure_component:
        cid, ename = ensure_component
        components[cid] = ename

    if ensure_components:
        components.update(ensure_components)

    if not components:
        registry_path.write_text(
            'import React from "react";\n\n'
            "type ComponentMap = { [key: string]: React.FC<any> };\n\n"
            "export const GeneratedRegistry: ComponentMap = {};\n"
        )
        return

    imports = []
    entries = []
    for cid in sorted(components):
        ename = components[cid]
        imports.append(f'import {{ {ename} }} from "./{cid}";')
        entries.append(f'  "{cid}": {ename} as React.FC<any>,')

    registry_code = (
        'import React from "react";\n'
        + "\n".join(imports) + "\n"
        "\n"
        "type ComponentMap = { [key: string]: React.FC<any> };\n"
        "\n"
        "export const GeneratedRegistry: ComponentMap = {\n"
        + "\n".join(entries) + "\n"
        "};\n"
    )
    registry_path.write_text(registry_code)


def _write_temp_registry(generated_dir: Path, component_id: str, export_name: str) -> None:
    """Write cumulative registry ensuring component_id is included."""
    _rebuild_registry(generated_dir, ensure_component=(component_id, export_name))


def _cleanup_registry(generated_dir: Path) -> None:
    """Rebuild registry after removing a failed component."""
    _rebuild_registry(generated_dir)


# ---------------------------------------------------------------------------
# A4: Build generated registry
# ---------------------------------------------------------------------------


@activity.defn(name="vfx_build_generated_registry")
def build_generated_registry(input_data: dict) -> dict:
    """Write the final auto-generated registry and convert specs to frame-based ComponentSpec list.

    Input: {
        "generated_components": list[dict],  # [{component_id, export_name, spec, props}]
        "video_fps": int,
        "style_config": dict | None,
    }
    Output: {
        "components": list[dict],  # Remotion-ready ComponentSpec dicts
        "registry_path": str,
    }
    """
    components_data = input_data["generated_components"]
    fps = input_data.get("video_fps", 30)
    style_config = input_data.get("style_config")

    if not components_data:
        return {"components": [], "registry_path": ""}

    remotion_dir = _get_remotion_dir()
    generated_dir = remotion_dir / "src" / "components" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Build registry file
    imports = []
    entries = []
    for comp in components_data:
        cid = comp["component_id"]
        ename = comp["export_name"]
        imports.append(f'import {{ {ename} }} from "./{cid}";')
        entries.append(f'  "{cid}": {ename} as React.FC<any>,')

    registry_code = (
        'import React from "react";\n'
        + "\n".join(imports) + "\n"
        "\n"
        "type ComponentMap = { [key: string]: React.FC<any> };\n"
        "\n"
        "export const GeneratedRegistry: ComponentMap = {\n"
        + "\n".join(entries) + "\n"
        "};\n"
    )

    registry_path = generated_dir / "_registry.ts"
    registry_path.write_text(registry_code)
    logger.info("Wrote generated registry with %d components", len(components_data))

    # Convert time-based specs to frame-based Remotion ComponentSpec dicts
    remotion_components = []
    for comp in components_data:
        spec = comp["spec"]
        start_frame = round(spec["start_time"] * fps)
        end_frame = round(spec["end_time"] * fps)
        duration_frames = max(1, end_frame - start_frame)

        remotion_components.append({
            "template": comp["component_id"],
            "startFrame": start_frame,
            "durationInFrames": duration_frames,
            "props": comp["props"],
            "bounds": spec.get("bounds", {"x": 0.1, "y": 0.1, "w": 0.35, "h": 0.3}),
            "zIndex": 10,  # generated components render above templates
            "anchor": spec.get("anchor", "static"),
        })

    return {
        "components": remotion_components,
        "registry_path": str(registry_path),
    }


# ---------------------------------------------------------------------------
# A5: Materialize library templates
# ---------------------------------------------------------------------------

_LIBRARY_IMPORT_PREAMBLE = """\
import React, { useMemo, useCallback, useRef, useEffect, useState } from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring, AbsoluteFill, Sequence, Img } from "remotion";
"""


@activity.defn(name="vfx_materialize_library_templates")
def materialize_library_templates(input_data: dict) -> dict:
    """Write library template TSX files to generated/ and rebuild registry.

    Input: {"template_ids": list[str]}
    Output: {"materialized": list[str]}
    """
    template_ids = input_data.get("template_ids", [])
    if not template_ids:
        return {"materialized": []}

    remotion_dir = _get_remotion_dir()
    generated_dir = remotion_dir / "src" / "components" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    materialized = []

    with _REGISTRY_LOCK:
        export_map: dict[str, str] = {}

        for tid in template_ids:
            tpl = template_library.get_template(tid)
            if tpl is None:
                logger.warning("Library template '%s' not found, skipping", tid)
                continue

            component_path = generated_dir / f"{tid}.tsx"
            if not component_path.exists():
                tsx_code = tpl.tsx_code
                tsx_code_clean = re.sub(r"^import\s+.*?['\";]\s*$", "", tsx_code, flags=re.MULTILINE)
                full_code = _LIBRARY_IMPORT_PREAMBLE + "\n" + tsx_code_clean
                component_path.write_text(full_code)
                logger.info("Materialized library template: %s", tid)

            export_map[tid] = tpl.export_name
            materialized.append(tid)

        _rebuild_registry(generated_dir, ensure_components=export_map)

    return {"materialized": materialized}
