from video_effects.core import BaseCapability
from video_effects.config import settings
from video_effects.helpers.llm import call_structured, load_prompt
from video_effects.schemas.programmer import ProgrammerPlanResponse
from video_effects.helpers.prompts import build_style_guide, build_spatial_user_message
from video_effects.skills.programmer.schemas import ProgrammerBrainstormRequest, ProgrammerBrainstormResponse


class ProgrammerBrainstormCapability(BaseCapability[ProgrammerBrainstormRequest, ProgrammerBrainstormResponse]):
    async def execute(self, request):
        base_prompt = load_prompt("programmer_brainstorm.md")
        style_guide = build_style_guide(request.style_config)
        system_prompt = base_prompt.replace("{STYLE_GUIDE}", style_guide)
        user_message = build_spatial_user_message(request.model_dump())
        self.heartbeat_sync("Brainstorming visual components via LLM")
        raw = call_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=ProgrammerPlanResponse,
            model=settings.PROGRAMMER_LLM_MODEL,
            max_tokens=16384,
        )
        components = raw.get("components", [])
        self.logger.info("Brainstorm proposed %d component(s)", len(components))
        return ProgrammerBrainstormResponse(reasoning=raw.get("reasoning", ""), components=components)
