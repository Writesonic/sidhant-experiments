# Zoom Follow Effect

## Overview

A video effect that smoothly zooms into a face, pans it to one side of the frame, fills the exposed background with an edge-fade, and optionally overlays crisp tracking text next to the face. Implemented in `zoom_text.py`.

## When to Use

- Speaker introduction or emphasis moments in talking-head videos
- Drawing attention to a face while displaying a name/title/caption beside it
- Creating dynamic "Ken Burns" style focus pulls on faces

## Function Signature

```python
create_zoom_follow_effect(
    input_path: str,       # Source video file path
    output_path: str,      # Output video file path
    zoom_max: float,       # Max zoom level (1.0 = no zoom, 1.5 = 50% zoom)
    t_start: float,        # Time (seconds) when zoom animation begins
    t_end: float,          # Time (seconds) when zoom animation completes
    face_side: str,        # "left" or "right" - where face lands on screen
    text_config: dict|None # Text overlay config (None = no text)
)
```

## Parameter Reference

| Parameter | Type | Range | Default | Notes |
|-----------|------|-------|---------|-------|
| `zoom_max` | float | 1.0-2.0 | 1.5 | 1.0-1.15 = subtle, 1.15-1.3 = moderate, 1.3+ = dramatic. Values >1.5 risk cropping |
| `t_start` | float | 0+ | 0 | Seconds into video. Animation eases in via smoothstep |
| `t_end` | float | >t_start | 5 | Must be > t_start. Duration = t_end - t_start |
| `face_side` | str | "left"/"right" | "right" | Where the face is positioned after zoom |

### text_config Dictionary

| Key | Type | Required | Default | Options |
|-----|------|----------|---------|---------|
| `content` | str | Yes | "Text" | The text to display |
| `position` | str | No | "left" | "left", "right", "top", "bottom" - relative to face |
| `color` | str | No | "white" | Any CSS color name or hex |
| `margin` | float | No | 1.3 | Multiplier for distance from face edge. 1.0 = touching face bounding box |
| `t_start` | float | No | same as effect t_start | When text starts appearing (fades in) |
| `t_end` | float | No | same as effect t_end | When text stops being visible |

## Critical Pairing Rule

**`face_side` and `text_config.position` must be on opposite sides.** The face occupies one side; text goes on the other where there's space.

| face_side | text position | Result |
|-----------|--------------|--------|
| "right" | "left" | Face right, text left |
| "left" | "right" | Face left, text right |
| "right" | "top"/"bottom" | Works but less common |
| "right" | "right" | Text overlaps face - BAD |

## Pipeline Integration

### As a Python Function Call

```python
from zoom_text import create_zoom_follow_effect

create_zoom_follow_effect(
    input_path="clip.mp4",
    output_path="clip_zoomed.mp4",
    zoom_max=1.1,
    t_start=1.0,
    t_end=6.0,
    face_side="right",
    text_config={
        "content": "Speaker Name",
        "position": "left",
        "color": "yellow"
    }
)
```

### As a CLI Command

```bash
cd /path/to/cv_experiments
python -c "
from zoom_text import create_zoom_follow_effect
create_zoom_follow_effect('input.mp4', 'output.mp4', zoom_max=1.1, t_start=1.0, t_end=6.0, face_side='right', text_config={'content': 'Hello', 'position': 'left', 'color': 'yellow'})
"
```

### LLM Decision Guide

When choosing parameters from a prompt like "zoom into the speaker and show their name":

1. **zoom_max**: Default to 1.1 for subtle, 1.2 for noticeable. Only go higher if explicitly asked for dramatic zoom.
2. **t_start/t_end**: Match the segment timing. If processing a 10s clip, `t_start=0.5, t_end=3.0` gives a quick zoom-in that holds.
3. **face_side**: If text is needed, put face on the opposite side from where text reads naturally. Default: face right, text left.
4. **text_config.content**: Extract from user intent (speaker name, caption, etc.)
5. **text_config.color**: "white" for dark backgrounds, "yellow" for general visibility.

## Dependencies

- `face_landmarker.task` model file must exist in the same directory as `zoom_text.py`
- Python packages: `opencv-python`, `numpy`, `mediapipe`, `moviepy`
- Environment: Use the venv at `/path/to/cv_experiments/.venv/`

## Constraints

- Processes one face only (first detected face)
- Output is always 24fps regardless of input fps
- Input must be a video file readable by OpenCV
- Text is rendered once at fontsize=80 and composited per-frame (not dynamically resized)
- No audio pass-through by default (moviepy handles this via `write_videofile`)
- Processing is CPU-bound and slow (~2-5x realtime depending on resolution)

## Architecture Notes

The pipeline processes each frame in this order:
1. Calculate zoom geometry from smoothed face tracking data
2. Warp the raw video frame (affine transform)
3. Compute screen-space face coordinates from the warp matrix
4. Apply edge-fade background effect
5. Composite text LAST onto the warped frame (keeps text crisp and unwarped)

Text is always drawn post-warp in screen space. Never pre-warp.
