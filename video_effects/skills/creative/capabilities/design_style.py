from pathlib import Path

from video_effects.config import settings
from video_effects.core import BaseCapability
from video_effects.helpers.llm import call_structured, call_text
from video_effects.schemas.styles import StyleDesignResponse, get_style
from video_effects.skills.creative.schemas import DesignStyleRequest, DesignStyleResponse

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


class DesignStyleCapability(BaseCapability[DesignStyleRequest, DesignStyleResponse]):
    async def execute(self, request: DesignStyleRequest) -> DesignStyleResponse:
        if request.style_override:
            self.logger.info("Style override: %s", request.style_override)
            return DesignStyleResponse(
                config=get_style(request.style_override).config.model_dump(),
                preset_name=request.style_override,
            )

        system_prompt = (_PROMPT_DIR / "design_style.md").read_text()

        _MAX_TRANSCRIPT_CHARS = 2000
        if len(request.transcript) <= _MAX_TRANSCRIPT_CHARS:
            excerpt = request.transcript
        else:
            self.heartbeat_sync("Summarizing long transcript for style design")
            summarizer_prompt = (_PROMPT_DIR / "summarize_transcript.md").read_text()
            excerpt = call_text(
                system_prompt=summarizer_prompt,
                user_message=request.transcript,
                model=settings.SMALL_LLM_MODEL,
                max_tokens=1024,
            ).strip()

        user_message = (
            f"## Video Info\n"
            f"- Duration: {request.video_duration:.1f}s\n"
            f"- FPS: {request.video_fps}\n\n"
            f"## Transcript\n\n{excerpt}\n"
        )

        self.heartbeat_sync("Calling LLM for style design")

        raw = call_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=StyleDesignResponse,
            model=settings.SMALL_LLM_MODEL,
        )

        preset_name = raw.get("preset", "default")
        adjustments = raw.get("adjustments", {})
        reasoning = raw.get("reasoning", "")

        self.logger.info("Creative designer picked preset=%s reason=%s", preset_name, reasoning)

        base_config = get_style(preset_name).config.model_dump()

        for key, value in adjustments.items():
            if key in base_config:
                if key == "font_weights" and isinstance(value, dict):
                    base_config["font_weights"].update(value)
                elif key == "palette" and isinstance(value, dict):
                    palette = list(base_config["palette"])
                    key_map = {"text": 0, "secondary": 1, "accent": 2}
                    for pkey, pval in value.items():
                        idx = key_map.get(pkey)
                        if idx is not None and isinstance(pval, str) and pval.startswith("#"):
                            palette[idx] = pval
                    base_config["palette"] = palette
                else:
                    base_config[key] = value

        return DesignStyleResponse(config=base_config, preset_name=preset_name)
