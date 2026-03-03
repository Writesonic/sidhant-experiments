from .video import get_video_info, extract_audio
from .transcribe import transcribe_audio
from .parse_cues import parse_effect_cues
from .validate import validate_timeline
from .apply_effects import apply_effects
from .compose import compose_final

ALL_VIDEO_EFFECTS_ACTIVITIES = [
    get_video_info,
    extract_audio,
    transcribe_audio,
    parse_effect_cues,
    validate_timeline,
    apply_effects,
    compose_final,
]
