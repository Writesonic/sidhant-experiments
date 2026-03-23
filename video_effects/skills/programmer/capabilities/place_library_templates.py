import logging

from video_effects.core import BaseCapability
from video_effects.helpers.llm import call_structured, load_prompt
from video_effects.helpers.templates import render_template_section
from video_effects.schemas.programmer import TemplatePlacementResponse
from video_effects.schemas import template_library
from video_effects.helpers.prompts import build_style_guide, build_spatial_user_message
from video_effects.skills.programmer.schemas import PlaceLibraryTemplatesRequest, PlaceLibraryTemplatesResponse

logger = logging.getLogger(__name__)


def _format_existing_components(components: list[dict]) -> str:
    if not components:
        return "No existing components.\n"
    lines = []
    for c in components:
        tpl = c.get("template", c.get("id", "?"))
        if "start_time" in c:
            st, et = c["start_time"], c["end_time"]
        elif "startFrame" in c:
            fps = 30
            st = c["startFrame"] / fps
            et = (c["startFrame"] + c.get("durationInFrames", 1)) / fps
        else:
            continue
        bounds = c.get("bounds", {})
        lines.append(f"- [{st:.1f}s - {et:.1f}s] {tpl} at ({bounds.get('x', 0):.2f}, {bounds.get('y', 0):.2f}, {bounds.get('w', 0):.2f}x{bounds.get('h', 0):.2f})")
    return "\n".join(lines) + "\n" if lines else "No existing components.\n"


def _validate_placement_props(placements: list[dict]) -> list[dict]:
    validated = []
    for p in placements:
        tid = p.get("template_id", "")
        tpl = template_library.get_template(tid)
        if tpl is None:
            logger.warning("Placement references unknown template '%s', skipping", tid)
            continue
        raw_props = p.get("props", {})
        if not isinstance(raw_props, dict):
            raw_props = {}
        props = dict(raw_props)
        skip = False
        for spec in tpl.props:
            if spec.name not in props:
                if spec.required:
                    logger.warning("[%s] Missing required prop '%s', skipping placement", tid, spec.name)
                    skip = True
                    break
                if spec.default is not None:
                    props[spec.name] = spec.default
            else:
                val = props[spec.name]
                if spec.type in ("int", "float") and isinstance(val, (int, float)):
                    if spec.min_value is not None and val < spec.min_value:
                        val = spec.min_value
                    if spec.max_value is not None and val > spec.max_value:
                        val = spec.max_value
                    if spec.type == "int":
                        val = int(val)
                    props[spec.name] = val
                if spec.choices and val not in spec.choices:
                    props[spec.name] = spec.default if spec.default is not None else spec.choices[0]
        if skip:
            continue
        p["props"] = props
        validated.append(p)
    return validated


class PlaceLibraryTemplatesCapability(BaseCapability[PlaceLibraryTemplatesRequest, PlaceLibraryTemplatesResponse]):
    async def execute(self, request):
        template_sections = []
        for tpl_data in request.pinned_templates:
            tid = tpl_data["id"] if isinstance(tpl_data, dict) else tpl_data
            tpl = template_library.get_template(tid)
            if tpl is None:
                self.logger.warning("Pinned template '%s' not found", tid)
                continue
            spec = template_library.as_mg_template_spec(tpl)
            template_sections.append(render_template_section(spec))
        if not template_sections:
            return PlaceLibraryTemplatesResponse(placements=[])
        base_prompt = load_prompt("place_library_templates.md")
        style_guide = build_style_guide(request.style_config)
        existing_text = _format_existing_components(request.existing_components)
        system_prompt = (
            base_prompt
            .replace("{TEMPLATE_SECTIONS}", "\n".join(template_sections))
            .replace("{EXISTING_COMPONENTS}", existing_text)
            .replace("{STYLE_GUIDE}", style_guide)
        )
        user_message = build_spatial_user_message(request.model_dump())
        self.heartbeat_sync("Placing library templates via LLM")
        raw = call_structured(system_prompt=system_prompt, user_message=user_message, response_model=TemplatePlacementResponse)
        placements = raw.get("placements", [])
        placements = _validate_placement_props(placements)
        self.logger.info("Placed %d library template(s)", len(placements))
        return PlaceLibraryTemplatesResponse(placements=placements)
