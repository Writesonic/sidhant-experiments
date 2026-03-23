from pydantic import BaseModel


class ApplyEffectsRequest(BaseModel):
    video_path: str
    output_dir: str
    effects: list[dict]
    video_info: dict


class ApplyEffectsResponse(BaseModel):
    processed_video: str
    phases_executed: int


class PrepareRenderRequest(BaseModel):
    video_path: str
    effects: list[dict]
    video_info: dict


class PrepareRenderResponse(BaseModel):
    decoded_width: int = 0
    decoded_height: int = 0
    is_hdr: bool = False
    phase_summary: list[dict] = []
    active_intervals: list = []
    active_frame_count: int = 0
    total_phases: int = 0
    has_effects: bool = False


class SetupProcessorsRequest(BaseModel):
    video_path: str
    effects: list[dict]
    video_info: dict
    cache_dir: str


class SetupProcessorsResponse(BaseModel):
    setup_summary: list[dict] = []
    processors_ready: bool = True


class RenderVideoRequest(BaseModel):
    video_path: str
    output_dir: str
    effects: list[dict]
    video_info: dict
    render_plan: dict
    cache_dir: str


class RenderVideoResponse(BaseModel):
    processed_video: str
    phases_executed: int
