from __future__ import annotations

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_loaded = False


def discover_skills() -> dict[str, Path]:
    skills_dir = Path(__file__).parent
    skills = {}
    for item in sorted(skills_dir.iterdir()):
        if item.is_dir() and (item / "skill.yml").exists():
            skills[item.name] = item
    return skills


def load_skill_activities(skill_name: str) -> bool:
    try:
        importlib.import_module(f"video_effects.skills.{skill_name}.activities")
        return True
    except Exception:
        logger.warning(f"Failed to load skill '{skill_name}'", exc_info=True)
        return False


def load_all_skills() -> None:
    global _loaded
    if _loaded:
        return
    skills = discover_skills()
    for skill_name in skills:
        load_skill_activities(skill_name)
    _loaded = True
