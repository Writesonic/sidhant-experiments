# Motion Graphics

> **Deprecation notice:** The old template-based MG planner (`_plan_motion_graphics` / `vfx_plan_motion_graphics`) has been disabled. Both `--mg` and `--infographics` now route through the [code-gen infographic pipeline](infographics.md). The `vfx_plan_motion_graphics` activity, prompt (`plan_motion_graphics_base.md`), and guidance files (`mg_guidance/*.md`) remain in the codebase but are no longer called.

The Remotion motion graphics system adds animated overlays (titles, lower thirds, lists, data visualizations, subtitles) to the processed video. Overlays are now generated via the infographic code-gen pipeline with face-aware spatial validation.

## Preview

MG previews render in the browser via `@remotion/player` in the Next.js web UI (`app/src/components/MgApproval.tsx`). The Player imports `DynamicComposition` directly as a React component and passes the composition plan as props. Video and data files are served via HTTP from the FastAPI `/api/files` endpoint. No CLI preview rendering (`vfx_preview_motion_graphics`, `vfx_render_preview_clip`) is needed — these activities are skipped in the workflow.

## End-to-End Flow

```
G8a: Build Spatial Context
  │  (face windows, safe regions, zoom state)
  │
  ├──► Infographic Generator (child workflow)
  │      │  6 parallel planners → LLM code-gen → validate
  │      │  See infographics.md
  │      │
  │      ├──► Merge infographic components into MG plan
  │      │    Re-validate merged plan
  │      │
  │      ├──► Place pinned library templates (LLM)
  │      │    Context-aware timing, props, positioning
  │      │
  │      ├──► Inject subtitles (zIndex=100)
  │      │
  │      ▼
  │    G8e: Render Overlay
  │      │  (Remotion → ProRes 4444, transparent)
  │      │
  │      ▼
  │    G9: FFmpeg Composite
  │      │  (premultiply + overlay onto base)
  │      │
  │      ▼
  │    Final .mp4
```

**Key files:**
- Activities: `activities/remotion.py`, `activities/programmer.py` (template placement)
- Helpers: `helpers/remotion.py`, `helpers/templates.py` (shared template renderer)
- Schemas: `schemas/mg_templates.py`, `schemas/motion_graphics.py`, `schemas/programmer.py` (placement models)
- Prompts: `prompts/plan_motion_graphics_base.md`, `prompts/place_library_templates.md`, `prompts/motion_graphics_schema.py`

## LLM Planning

### System Prompt Assembly

`build_mg_system_prompt(style_config, style_preset_name)` dynamically assembles the prompt from:

1. **Base rules** (`prompts/plan_motion_graphics_base.md`): Spatial constraints (never occlude face, 10% edge margin, max 2 concurrent overlays), temporal rules (align to transcript timestamps, 0.1–0.3s delay after spoken cue), and density targets.
2. **Style guide** (from the active [style preset](styles.md)): Preferred/avoided animations, density range, color palette, template preferences.
3. **Template catalog** (from `schemas/mg_templates.py`): Available templates with props, duration ranges, placement constraints, and per-template [creative guidance](llm-prompts.md).

### User Message

The user message includes:
- Spatial context: face windows with safe regions per 3-second window
- Transcript with word-level timestamps
- Active OpenCV effects (so overlays avoid zoom transitions)
- Optional feedback from previous rejection

### Response Model

```python
class MotionGraphicsPlanResponse(BaseModel):
    components: list[MGComponentSpec]
    color_palette: list[str]    # 2-3 CSS hex colors
    reasoning: str

class MGComponentSpec(BaseModel):
    template: str               # e.g., "animated_title"
    start_time: float
    end_time: float
    props: dict                 # Template-specific
    bounds: MGComponentBounds   # Normalized 0-1 rect
    z_index: int
    anchor: Literal["static", "face-right", "face-left",
                    "face-below", "face-above", "face-beside"]
    reasoning: str
```

## Template Registry

Four implemented templates, defined in `schemas/mg_templates.py`:

| Template | Props | Duration | Spatial | Edge-Aligned |
|----------|-------|----------|---------|--------------|
| `animated_title` | text, style (fade/slide-in/typewriter/bounce), fontSize, color, fontWeight | 2–5s | y: 0.05–0.25 | No |
| `lower_third` | name, title, accentColor, style (slide/fade), fontSize, color | 3–6s | y: 0.75–0.88, x: 0.03–0.45 | Yes |
| `listicle` | items (max 5), style (pop/slide), listStyle, staggerDelay, fontSize, color, accentColor | 3–8s | y: 0.15–0.75 | No |
| `data_animation` | style (counter/stat-callout/bar), value, label, startValue, suffix, prefix, delta, items, fontSize, color, accentColor | 2–6s | y: 0.15–0.75 | No |

Each template has a guidance markdown file in `prompts/mg_guidance/` loaded dynamically into the LLM prompt.

`IMPLEMENTED_TEMPLATES` is the source of truth — only templates in this set are available.

## Spatial Validation (`_validate_plan()`)

A multi-phase validation pipeline in `activities/remotion.py` ensures overlays don't occlude the speaker or conflict with each other. Uses zero overlap tolerance — any intersection triggers relocation.

### One-Shot Corrections

| # | Check | Action |
|---|-------|--------|
| 1 | **Hard bounds clamping** | Keep within [0.02, 0.98], enforce min size 0.05 × 0.03 |
| 2 | **Time clamping** | Clamp to video duration, enforce min 0.5s |
| 3 | **Template duration limits** | Enforce max duration per template spec (e.g., lower_third 6s max) |
| 4 | **Concurrent count enforcement** | Drop lowest z_index if ≥2 non-edge-aligned components overlap in time |
| 5 | **Zoom viewport clamping** | During active zooms, clamp bounds to the visible inner area (e.g., 67% for 1.5× zoom) |
| 6 | **Zoom transition buffer** | Shift overlay timing away from zoom ease-in/ease-out windows |

### Single-Pass Conflict Resolution (free-rectangle tiling)

After one-shot corrections, step 7 resolves all face and inter-component overlaps in a single pass via `_resolve_all_conflicts()`:

1. Sort components by `z_index` descending — highest priority placed first.
2. For each component, build an obstacle list: padded face rects (time-overlapping) + already-placed component rects (time-overlapping). Edge-aligned components are registered as obstacles but never relocated.
3. If any obstacle overlaps the component's bounds, compute free rectangles within the safe frame (2% inset) using `_compute_free_rects()`, then find the best placement via `_find_best_free_placement()`.
4. Register final bounds as an obstacle for subsequent (lower z_index) components.

No convergence loop needed — each component is placed once and becomes a fixed obstacle.

### Key Helpers

- `_rect_overlap_fraction(a, b)` — Overlap as fraction of rect a's area
- `_compute_free_rects(frame, obstacles)` — Maximal free rectangles after subtracting obstacles (bin-packing split)
- `_find_best_free_placement(comp_w, comp_h, free_rects, original_pos)` — Picks closest fitting free rect, shrinks preserving aspect ratio if needed
- `_resolve_all_conflicts(components, face_windows, edge_aligned_templates, issues, static_obstacles=None, safe_frame=None, zoom_effects=None)` — Single-pass z_index-ordered placement with obstacle accumulation
- `_compute_zoom_stable_window(zoom_cue)` — Returns time range where zoom is stable:
  - Bounce: 25%–75% of duration
  - In: 60%–end
  - Out: start–40%

## ProRes 4444 Rendering

**File:** `helpers/remotion.py` → `render_media()`

```bash
npx remotion render MotionOverlay output.mov \
  --codec prores --prores-profile 4444 \
  --pixel-format yuva444p10le \
  --image-format png \
  --props '...'
```

| Setting | Value | Reason |
|---------|-------|--------|
| Codec | ProRes 4444 | Preserves alpha channel |
| Pixel format | yuva444p10le | 10-bit YUVA 4:4:4 with alpha |
| Image format | PNG | Lossless intermediate frames |
| Timeout | 600s | Long renders for complex overlays |

The `CompositionPlan` is serialized as JSON props, containing all components, color palette, face data path, zoom state path, and style config.

## FFmpeg Compositing

**File:** `helpers/remotion.py` → `composite_overlay()`

```bash
ffmpeg -y -i base.mp4 -i overlay.mov \
  -filter_complex "[1:v]premultiply=inplace=1[ovr];[0:v][ovr]overlay=0:0:shortest=1" \
  -c:v libx264 -crf 16 -c:a copy \
  output.mp4
```

1. **Premultiply**: Convert ProRes 4444 straight alpha to premultiplied
2. **Overlay**: Full-frame placement (0,0), trimmed to shorter stream
3. **Encode**: H.264 CRF 16, copy audio stream

## Subtitle Injection

After the MG plan is finalized, word-level transcript segments are injected as a `subtitles` component with `zIndex=100` (renders above all other overlays). The Remotion [Subtitles component](remotion-components.md#subtitles) handles word-by-word highlighting.

## Library Template Placement

When the user pins library templates to a workflow, they are placed via `vfx_place_library_templates` — an LLM activity that runs in the main workflow (not inside ProgrammerWorkflow or InfographicGeneratorWorkflow).

**Why main workflow?** Pinned templates need placement even when `enable_programmer` is false, and the activity needs ProgrammerWorkflow's output as input to avoid temporal conflicts.

**Flow:**
1. Resolve each pinned template ID → full `LibraryTemplate` → `MGTemplateSpec`
2. Render template metadata into the prompt via `render_template_section()` (props table, duration, spatial hints)
3. Build system prompt with template specs, existing component time windows, and style guide
4. LLM returns `TemplatePlacementResponse` — per-template timing, bounds, filled props, rationale
5. Validate props against each template's `PropSpec` (clamp ranges, apply defaults, validate choices)
6. Convert to frame-domain components and inject into MG plan

**Activity:** `vfx_place_library_templates` in `activities/programmer.py`
**Prompt:** `prompts/place_library_templates.md`
**Models:** `TemplatePlacement`, `TemplatePlacementResponse` in `schemas/programmer.py`

## Parallelization

MG planning (G8b), infographic generation (child workflow), and video rendering (G6c) all run concurrently after G8a completes. The merge happens after all three finish, followed by the final Remotion render and FFmpeg composite.
