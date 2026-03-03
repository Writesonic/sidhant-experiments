from typing import Optional

from pydantic import BaseModel, Field


class VideoEffectsInput(BaseModel):
    input_video: str = Field(description="Path to input video file")
    output_video: str = Field(description="Path for output video file")
    auto_approve: bool = Field(False, description="Skip CLI approval step")


class VideoEffectsOutput(BaseModel):
    output_video: str = Field(description="Path to the final output video")
    effects_applied: int = Field(0, description="Number of effects applied")
    transcript_length: int = Field(0, description="Length of transcript in chars")
    phases_executed: int = Field(0, description="Number of phases executed")
    error: Optional[str] = None
