"""Video Effects activities — shim that loads all skills via discovery.

This module exists for backward compatibility. The actual activity
implementations live in video_effects/skills/<skill_name>/.
"""

from video_effects.skills.discovery import load_all_skills
from video_effects.skills.registry import get_all_activities, get_activities_by_queue

# Trigger registration of all skill activities
load_all_skills()

# Build ALL_VIDEO_EFFECTS_ACTIVITIES from the registry
ALL_VIDEO_EFFECTS_ACTIVITIES = [
    cfg.func for cfg in get_all_activities().values() if cfg.func is not None
]
