from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class VideoEffectsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="VFX_",
    )

    TASK_QUEUE: str = "video_effects_queue"
    TEMPORAL_NAMESPACE: str = "default"
    TEMPORAL_ENDPOINT: str = "localhost:7233"
    TEMPORAL_API_KEY: Optional[str] = None

    # Anthropic API for LLM cue parsing
    ANTHROPIC_API_KEY: Optional[str] = None
    LLM_MODEL: str = "claude-sonnet-4-6"
    SMALL_LLM_MODEL: str = "claude-haiku-4-5"

    # ElevenLabs for transcription (falls back to local whisper)
    ELEVENLABS_API_KEY: Optional[str] = None

    # Paths
    TEMP_DIR: str = "/tmp/video_effects"
    FACE_LANDMARKER_PATH: str = "cv_experiments/face_landmarker.task"  # Path to face_landmarker.task model

    # Processing
    FACE_DETECTION_STRIDE: int = 3
    SMOOTHING_ALPHA: float = 0.1

    # Remotion motion graphics
    REMOTION_DIR: Optional[str] = None  # Override path to remotion/ project (default: auto-detected)
    REMOTION_CONCURRENCY: Optional[int] = None  # Remotion render concurrency (default: Remotion auto)
    # Infographic code generation
    INFOGRAPHIC_MAX_RETRIES: int = 3  # Max code-gen + validate attempts per infographic
    INFOGRAPHIC_LLM_MODEL: str = "claude-opus-4-6"  # Use Opus for codegen quality

    # Programmer workflow
    PROGRAMMER_MAX_RETRIES: int = 3  # Max code-gen + validate attempts per component
    PROGRAMMER_LLM_MODEL: str = "claude-opus-4-6"  # Opus for creative code gen

    # API server
    API_PORT: int = 8000
    API_CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    ALLOWED_FILE_DIRS: list[str] = ["/tmp/video_effects", "/Users/sidhant/sidhant-experiments"]


settings = VideoEffectsSettings()
