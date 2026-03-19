"""Shared helpers for rendering template metadata into prompt sections."""

from video_effects.schemas.mg_templates import MGTemplateSpec, load_guidance


def render_template_section(spec: MGTemplateSpec) -> str:
    """Format a single template's metadata + guidance into a markdown section."""
    lines = [f"### {spec.name}", spec.description, ""]

    lines.append("| Prop | Type | Required | Default | Constraints |")
    lines.append("|------|------|----------|---------|-------------|")
    for p in spec.props:
        constraints = ""
        if p.choices:
            constraints = " | ".join(p.choices)
        elif p.min_value is not None or p.max_value is not None:
            lo = p.min_value if p.min_value is not None else ""
            hi = p.max_value if p.max_value is not None else ""
            constraints = f"{lo}-{hi}"
        lines.append(
            f"| `{p.name}` | {p.type} | {'yes' if p.required else 'no'} "
            f"| {p.default if p.default is not None else '-'} | {constraints} |"
        )
    lines.append("")

    lines.append(f"- **Duration**: {spec.duration_range[0]}-{spec.duration_range[1]} seconds")
    sy = spec.spatial
    lines.append(
        f"- **Typical placement**: y {sy.typical_y_range[0]:.0%}-{sy.typical_y_range[1]:.0%}, "
        f"x {sy.typical_x_range[0]:.0%}-{sy.typical_x_range[1]:.0%}"
        + (" (edge-aligned)" if sy.edge_aligned else "")
    )
    lines.append("")

    guidance = load_guidance(spec)
    if guidance:
        lines.append(guidance)
        lines.append("")

    return "\n".join(lines)
