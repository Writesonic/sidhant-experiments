import json

from video_effects.core import BaseCapability
from video_effects.config import settings
from video_effects.helpers.llm import call_structured, load_prompt
from video_effects.schemas.programmer import ProgrammerPlanResponse
from video_effects.skills.programmer.schemas import ProgrammerCritiqueRequest, ProgrammerCritiqueResponse


class ProgrammerCritiqueCapability(BaseCapability[ProgrammerCritiqueRequest, ProgrammerCritiqueResponse]):
    async def execute(self, request):
        if not request.proposals:
            return ProgrammerCritiqueResponse(components=[])
        base_prompt = load_prompt("programmer_critique.md")
        system_prompt = base_prompt.replace("{MAX_SPECS}", str(request.max_specs))
        lines = [
            "## Proposals to Evaluate\n",
            f"```json\n{json.dumps(request.proposals, indent=2)}\n```\n",
            f"## Constraints",
            f"- Maximum components to keep: {request.max_specs}",
            f"- Subtitle zone (y >= 0.78) is reserved",
            f"- Minimum 2 second gap between components",
        ]
        context = request.spatial_context
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
        self.heartbeat_sync("Critiquing proposals via LLM")
        raw = call_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=ProgrammerPlanResponse,
            model=settings.PROGRAMMER_LLM_MODEL,
            max_tokens=16384,
        )
        components = raw.get("components", [])
        components.sort(key=lambda c: c.get("score", 50), reverse=True)
        components = components[:request.max_specs]
        self.logger.info("Critique kept %d of %d proposals", len(components), len(request.proposals))
        return ProgrammerCritiqueResponse(components=components)
