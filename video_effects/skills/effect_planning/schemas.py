from pydantic import BaseModel


class ParseEffectCuesRequest(BaseModel):
    transcript: str
    segments: list[dict]
    duration: float = 0
    feedback: str = ""
    style_config: dict | None = None
    style_preset_name: str = ""
    dev_mode: bool = False


class ParseEffectCuesResponse(BaseModel):
    effects: list[dict]
    reasoning: str = ""


class ValidateTimelineRequest(BaseModel):
    effects: list[dict]
    duration: float = 0


class ValidateTimelineResponse(BaseModel):
    effects: list[dict]
    conflicts_resolved: int = 0
    total_duration: float = 0
