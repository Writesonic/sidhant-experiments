"""Shared formatting helpers for CLI and web interfaces."""


def format_timeline_markdown(timeline: dict) -> str:
    """Render the effect timeline as a markdown table."""
    effects = timeline.get("effects", [])
    conflicts = timeline.get("conflicts_resolved", 0)

    if not effects:
        return "No effects detected."

    lines = [
        "| # | Type | Start | End | Conf | Cue |",
        "|---|------|-------|-----|------|-----|",
    ]
    for i, e in enumerate(effects, 1):
        etype = e.get("effect_type", "?")
        start = e.get("start_time", 0)
        end = e.get("end_time", 0)
        conf = e.get("confidence", 0)
        cue = e.get("verbal_cue", "")
        lines.append(f"| {i} | {etype} | {start:.1f}s | {end:.1f}s | {conf:.0%} | {cue} |")

    lines.append(f"\n**{len(effects)} effects** | {conflicts} conflicts resolved")
    return "\n".join(lines)


def format_mg_plan_markdown(plan: dict) -> str:
    """Render the MG component plan as a markdown table."""
    components = plan.get("components", [])

    if not components:
        return "No motion graphics components."

    lines = [
        "| # | Template | Start | End | Props |",
        "|---|----------|-------|-----|-------|",
    ]
    fps = 30
    for i, c in enumerate(components, 1):
        template = c.get("template", "?")
        start_frame = c.get("startFrame", 0)
        dur_frames = c.get("durationInFrames", 0)
        start_s = start_frame / fps
        end_s = (start_frame + dur_frames) / fps

        props = c.get("props", {})
        display = props.get("text", "") or props.get("name", "") or ""
        if len(display) > 40:
            display = display[:37] + "..."

        lines.append(f"| {i} | {template} | {start_s:.1f}s | {end_s:.1f}s | {display} |")

    lines.append(f"\n**{len(components)} component(s)**")
    return "\n".join(lines)


def compute_preview_frames(mg_plan: dict, video_info: dict) -> list[int]:
    """Select representative frames for MG preview snapshots."""
    fps = int(video_info.get("fps", 30))
    total_frames = int(video_info.get("duration", 10) * fps)

    # Composition length is bounded by the last component end frame
    components = mg_plan.get("components", [])
    max_comp_frame = max(
        (c.get("startFrame", 0) + c.get("durationInFrames", 0) for c in components),
        default=total_frames,
    )
    max_frame = min(total_frames, max_comp_frame) - 1  # 0-indexed, so last renderable = length - 1

    comp_midpoints = [
        c["startFrame"] + c["durationInFrames"] // 2
        for c in components
        if c.get("template") != "subtitles"
    ]
    keyframes = [int(max_frame * p) for p in [0.1, 0.25, 0.5, 0.75]]
    frames = sorted(set(comp_midpoints + keyframes))
    return [f for f in frames if 0 <= f <= max_frame][:8]
