from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from functools import lru_cache

from video_effects.infrastructure.provider import get_activity_wrapper


@lru_cache(maxsize=None)
def _load_skill_queue(skill_name: str) -> str | None:
    skill_yml = Path(__file__).parent / skill_name / "skill.yml"
    try:
        with open(skill_yml) as f:
            return yaml.safe_load(f).get("queue")
    except FileNotFoundError:
        return None


@dataclass
class ActivityConfig:
    activity_id: str
    name: str
    queue: str
    description: str = ""
    func: Callable | None = None


_ACTIVITIES: dict[str, ActivityConfig] = {}


def register_activity(
    name: str,
    description: str = "",
    queue: str | None = None,
):
    def decorator(func: Callable) -> Callable:
        effective_queue = queue

        if effective_queue is None:
            try:
                parts = func.__module__.split(".")
                skills_idx = parts.index("skills")
                skill_name = parts[skills_idx + 1]
                effective_queue = _load_skill_queue(skill_name)
            except (ValueError, IndexError):
                pass

        if not effective_queue:
            raise ValueError(
                f"No queue for activity '{name}'. "
                f"Provide queue= in @register_activity or define it in skill.yml"
            )

        activity_wrapper = get_activity_wrapper()
        wrapped_func = activity_wrapper(func, name)

        _ACTIVITIES[name] = ActivityConfig(
            activity_id=name,
            name=name,
            queue=effective_queue,
            description=description,
            func=wrapped_func,
        )

        return func

    return decorator


def get_activity(activity_id: str) -> ActivityConfig:
    if activity_id not in _ACTIVITIES:
        raise ValueError(f"Unknown activity: {activity_id}. Available: {list(_ACTIVITIES.keys())}")
    return _ACTIVITIES[activity_id]


def get_all_activities() -> dict[str, ActivityConfig]:
    return _ACTIVITIES.copy()


def get_activities_by_queue(queue_name: str) -> list[Callable]:
    return [
        cfg.func
        for cfg in _ACTIVITIES.values()
        if cfg.queue == queue_name and cfg.func is not None
    ]
