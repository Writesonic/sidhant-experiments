from video_effects.core import BaseCapability
from video_effects.config import settings
from video_effects.helpers.llm import call_structured, load_prompt
from video_effects.helpers.prompts import build_style_guide, build_spatial_user_message
from video_effects.schemas.infographic import InfographicPlanResponse
from video_effects.skills.infographic.schemas import PlanCategoryRequest, PlanCategoryResponse


class PlanCategoryCapability(BaseCapability[PlanCategoryRequest, PlanCategoryResponse]):
    async def execute(self, request):
        base_prompt = load_prompt(request.prompt_filename)
        style_guide = build_style_guide(request.style_config)
        system_prompt = base_prompt.replace("{STYLE_GUIDE}", style_guide)
        user_message = build_spatial_user_message(request.model_dump())
        self.heartbeat_sync(f"Planning {request.category} via LLM")
        raw = call_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=InfographicPlanResponse,
            model=settings.INFOGRAPHIC_LLM_MODEL,
        )
        infographics = raw.get("infographics", [])
        self.logger.info("LLM planned %d %s component(s)", len(infographics), request.category)
        return PlanCategoryResponse(infographics=infographics, reasoning=raw.get("reasoning", ""))
