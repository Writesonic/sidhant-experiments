import json

from video_effects.core import BaseCapability
from video_effects.config import settings
from video_effects.helpers.llm import call_text, load_prompt
from video_effects.helpers.prompts import derive_export_name, strip_markdown_fences
from video_effects.helpers.remotion import _get_remotion_dir
from video_effects.skills.infographic.schemas import GenerateInfographicCodeRequest, GenerateInfographicCodeResponse


def _build_codegen_prompt(api_reference: str) -> str:
    base = load_prompt("generate_infographic_code.md")
    components_dir = _get_remotion_dir() / "src" / "components"
    examples = []
    for name in ("DataAnimation.tsx", "AnimatedTitle.tsx"):
        path = components_dir / name
        if path.exists():
            examples.append(f"### Example: {name}\n\n```tsx\n{path.read_text()}\n```")
    examples_section = "## Real Component Examples\n\n" + "\n\n".join(examples) if examples else ""
    return base.replace("{API_REFERENCE}", f"## Allowed Imports (API Reference)\n\n{api_reference}").replace("{EXAMPLES}", examples_section)


class GenerateInfographicCodeCapability(BaseCapability[GenerateInfographicCodeRequest, GenerateInfographicCodeResponse]):
    async def execute(self, request):
        spec = request.spec
        api_reference = load_prompt("infographic_api_reference.md")
        system_prompt = _build_codegen_prompt(api_reference)
        component_id = spec["id"]
        export_name = derive_export_name(component_id)
        lines = [
            f"## Infographic Spec",
            f"- Component ID: `{component_id}`",
            f"- Export name: `{export_name}`",
            f"- Type: {spec['type']}",
            f"- Title: {spec['title']}",
            f"- Description: {spec['description']}",
            f"- Duration: {spec['end_time'] - spec['start_time']:.1f}s",
            f"- Video: {request.video_info.get('width', 1920)}x{request.video_info.get('height', 1080)} @ {request.video_info.get('fps', 30)}fps",
            "", "## Data to Visualize", "```json",
            json.dumps(spec.get("data", {}), indent=2), "```",
        ]
        if request.style_config:
            palette = request.style_config.get("palette", [])
            if palette:
                lines.append(f"\n## Style")
                lines.append(f"Use useStyle() for colors, but for reference the palette is: {palette}")
        if request.attempt > 1 and request.previous_errors:
            lines.append(f"\n## RETRY (attempt {request.attempt})")
            lines.append("The previous code had these errors:")
            for err in request.previous_errors:
                lines.append(f"- {err}")
            if request.previous_code:
                lines.append(f"\n## Previous Code (fix these issues)\n\n```tsx\n{request.previous_code}\n```")
        user_message = "\n".join(lines)
        self.heartbeat_sync(f"Generating code for {component_id} (attempt {request.attempt})")
        raw_code = call_text(system_prompt=system_prompt, user_message=user_message, model=settings.INFOGRAPHIC_LLM_MODEL)
        tsx_code = strip_markdown_fences(raw_code)
        props = dict(spec.get("data", {}))
        return GenerateInfographicCodeResponse(component_id=component_id, tsx_code=tsx_code, export_name=export_name, props=props)
