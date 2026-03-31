from pydantic import BaseModel


class StartStudioRequest(BaseModel):
    mg_plan: dict
    base_video_path: str = ""
    face_data_path: str = ""
    zoom_state_path: str = ""
    video_info: dict = {}


class StartStudioResponse(BaseModel):
    studio_url: str = ""
    pid: int = 0
    port: int = 0


class StopStudioRequest(BaseModel):
    pid: int


class StopStudioResponse(BaseModel):
    stopped: bool = True


class UpdateStudioPreviewRequest(BaseModel):
    mg_plan: dict
    video_info: dict = {}


class UpdateStudioPreviewResponse(BaseModel):
    updated: bool = True
