from pydantic import BaseModel


class DetectFacesRequest(BaseModel):
    video_path: str
    video_info: dict
    cache_dir: str


class DetectFacesResponse(BaseModel):
    face_data_path: str
    frames_detected: int
    from_cache: bool


class BuildRemotionContextRequest(BaseModel):
    video_info: dict
    transcript: str
    segments: list[dict]
    effects: list[dict]
    cache_dir: str = ""
    face_data_path: str = ""
    zoom_state_path: str = ""
    style_config: dict | None = None
    style_preset_name: str = ""


class BuildRemotionContextResponse(BaseModel):
    context: dict
