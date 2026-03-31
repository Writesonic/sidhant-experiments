from pydantic import BaseModel


class DesignStyleRequest(BaseModel):
    transcript: str = ""
    video_duration: float = 30.0
    video_fps: float = 30.0
    style_override: str = ""


class DesignStyleResponse(BaseModel):
    config: dict
    preset_name: str
