from pydantic import BaseModel

from video_effects.schemas.effects import VideoInfo


class GetVideoInfoRequest(BaseModel):
    video_path: str


class GetVideoInfoResponse(BaseModel):
    video_info: VideoInfo


class ExtractAudioRequest(BaseModel):
    video_path: str
    output_dir: str


class ExtractAudioResponse(BaseModel):
    audio_path: str
    original_audio_path: str
