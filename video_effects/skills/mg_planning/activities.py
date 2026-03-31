"""Pass-through registration for MG planning activities.

These activities have 800+ lines of shared helpers (_validate_props,
_resolve_all_conflicts, _validate_plan, _compute_safe_regions, etc.)
that make individual capability extraction non-trivial. For now, we
register the existing @activity.defn-wrapped functions from remotion.py
directly into the skill registry. A follow-up refactor will extract
capabilities one at a time.
"""

from video_effects.skills.registry import _ACTIVITIES, ActivityConfig

from video_effects.activities.remotion import (
    plan_motion_graphics,
    validate_merged_plan,
    load_composition_plan,
    render_motion_overlay,
    preview_motion_graphics,
    render_preview_clip,
    edit_mg_plan,
)

_MG_ACTIVITIES = {
    "vfx_plan_motion_graphics": plan_motion_graphics,
    "vfx_validate_merged_plan": validate_merged_plan,
    "vfx_load_composition_plan": load_composition_plan,
    "vfx_render_motion_overlay": render_motion_overlay,
    "vfx_preview_motion_graphics": preview_motion_graphics,
    "vfx_render_preview_clip": render_preview_clip,
    "vfx_edit_mg_plan": edit_mg_plan,
}

for _name, _func in _MG_ACTIVITIES.items():
    _ACTIVITIES[_name] = ActivityConfig(
        activity_id=_name,
        name=_name,
        queue="video_effects_queue",
        description=f"MG planning: {_name}",
        func=_func,
    )
