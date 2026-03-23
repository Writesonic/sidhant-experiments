from pydantic import BaseModel


class ComposeFinalRequest(BaseModel):
    processed_video: str
    audio_path: str = ""
    output_path: str
    has_audio: bool = True


class ComposeFinalResponse(BaseModel):
    output_video: str


class CompositeMotionGraphicsRequest(BaseModel):
    base_video: str
    overlay_video: str
    output_path: str
    temp_dir: str = ""


class CompositeMotionGraphicsResponse(BaseModel):
    output_video: str
