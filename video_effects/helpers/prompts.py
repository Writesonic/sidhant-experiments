"""Shared prompt-building utilities for LLM-driven activities."""

from __future__ import annotations

import re


def build_style_guide(style_config: dict | None) -> str:
    if not style_config:
        return ""
    palette = style_config.get("palette", [])
    lines = ["## Style Guide\n"]
    if isinstance(palette, list) and len(palette) >= 3:
        lines.append(f"**Color palette**: {palette[0]} (text), {palette[1]} (secondary), {palette[2]} (accent)")
    elif isinstance(palette, dict):
        parts = [f"{v} ({k})" for k, v in palette.items() if isinstance(v, str) and v.startswith("#")]
        if parts:
            lines.append(f"**Color palette**: {', '.join(parts)}")
    font = style_config.get("font_family", "")
    if font:
        lines.append(f"**Font**: {font}")
    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


def build_spatial_user_message(input_data: dict) -> str:
    """Build user message with video info, face windows, effects, and transcript.

    Works for both infographic and programmer pipelines. The effects section
    is included only when input_data contains a non-empty "effects" key.
    """
    context = input_data.get("spatial_context", {})
    transcript = input_data.get("transcript", "")
    segments = input_data.get("segments", [])
    lines = []
    video = context.get("video", {})
    lines.append("## Video Info")
    lines.append(f"- Duration: {video.get('duration', 0):.1f}s")
    lines.append(f"- Resolution: {video.get('width', '?')}x{video.get('height', '?')}")
    lines.append(f"- FPS: {video.get('fps', 30)}\n")
    face_windows = context.get("face_windows", [])
    if face_windows:
        lines.append("## Face Position (time windows)")
        for fw in face_windows[:15]:
            fr = fw["face_region"]
            safe_labels = [s["label"] for s in fw.get("safe_regions", [])]
            lines.append(
                f"- [{fw['start_time']:.1f}s - {fw['end_time']:.1f}s] "
                f"face at ({fr['x']:.2f}, {fr['y']:.2f}, {fr['w']:.2f}x{fr['h']:.2f}) safe: {', '.join(safe_labels) or 'none'}"
            )
        lines.append("")
    effects = input_data.get("effects", [])
    if effects:
        lines.append("## Existing Video Effects (do NOT overlap these)")
        for e in effects:
            lines.append(f"- [{e.get('start_time', 0):.1f}s - {e.get('end_time', 0):.1f}s] {e.get('effect_type', '?')}")
        lines.append("")
    if segments:
        lines.append("## Timestamped Transcript")
        for seg in segments:
            if seg.get("type") == "word":
                lines.append(f"[{seg.get('start', 0):.2f}s] {seg.get('text', '')}")
        lines.append("")
    if transcript:
        lines.append(f"## Full Transcript\n\n{transcript[:3000]}\n")
    return "\n".join(lines)


def derive_export_name(component_id: str) -> str:
    name = "".join(word.capitalize() for word in component_id.replace("-", "_").split("_"))
    if name and name[0].isdigit():
        name = "Ig" + name
    return name


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text
