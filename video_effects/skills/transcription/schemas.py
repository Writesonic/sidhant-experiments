from pydantic import BaseModel


class TranscribeAudioRequest(BaseModel):
    audio_path: str


class TranscribeAudioResponse(BaseModel):
    transcript: str
    segments: list[dict]
