# LLM Prompts

The system uses Claude for six distinct tasks: effect cue parsing, style detection, motion graphics planning, infographic planning, infographic code generation, and library template placement. All use structured output via tool-use except code generation which returns raw text.

**Key files:**
- LLM helper: `helpers/llm.py`
- Config: `config.py`
- Prompt directory: `prompts/`

## Prompt Files

| File | Task | Activity | Response Model |
|------|------|----------|----------------|
| `prompts/parse_effect_cues.md` | Infer effects from transcript energy | `vfx_parse_effect_cues` | `ParsedEffectCues` |
| `prompts/parse_effect_cues_dev.md` | Parse explicit verbal commands | `vfx_parse_effect_cues` (dev mode) | `ParsedEffectCues` |
| `prompts/design_style.md` | Auto-select style preset | `vfx_design_style` | `StyleDesignResponse` |
| `prompts/plan_motion_graphics_base.md` | ~~Plan overlay placements~~ (deprecated — old MG planner disabled, overlays now via code-gen) | ~~`vfx_plan_motion_graphics`~~ | `MotionGraphicsPlanResponse` |
| `prompts/plan_infographics.md` | Identify data visualization moments | `vfx_plan_infographics` | `InfographicPlanResponse` |
| `prompts/plan_diagrams.md` | Identify diagram/flowchart moments | `vfx_plan_diagrams` | `InfographicPlanResponse` |
| `prompts/plan_timelines.md` | Identify timeline/journey moments | `vfx_plan_timelines` | `InfographicPlanResponse` |
| `prompts/plan_quotes.md` | Identify key quotes/callouts | `vfx_plan_quotes` | `InfographicPlanResponse` |
| `prompts/plan_code_blocks.md` | Identify code snippet moments | `vfx_plan_code_blocks` | `InfographicPlanResponse` |
| `prompts/plan_comparisons.md` | Identify comparison/versus moments | `vfx_plan_comparisons` | `InfographicPlanResponse` |
| `prompts/generate_infographic_code.md` | Write TSX component code | `vfx_generate_infographic_code` | Raw text (TSX) |
| `prompts/programmer_brainstorm.md` | Creative brainstorm for visual components | `vfx_programmer_brainstorm` | `ProgrammerPlanResponse` |
| `prompts/programmer_critique.md` | Self-critique and filter proposals | `vfx_programmer_critique` | `ProgrammerPlanResponse` |
| `prompts/programmer_generate_code.md` | Generate TSX for programmer components | `vfx_programmer_generate_code` | Raw text (TSX) |
| `prompts/place_library_templates.md` | Context-aware placement of pinned library templates | `vfx_place_library_templates` | `TemplatePlacementResponse` |
| `prompts/infographic_api_reference.md` | Allowed imports for generated code | (embedded in codegen prompt) | — |
| `prompts/mg_guidance/animated_title.md` | Creative guidance for AnimatedTitle | (embedded in MG prompt) | — |
| `prompts/mg_guidance/lower_third.md` | Creative guidance for LowerThird | (embedded in MG prompt) | — |
| `prompts/mg_guidance/listicle.md` | Creative guidance for Listicle | (embedded in MG prompt) | — |
| `prompts/mg_guidance/data_animation.md` | Creative guidance for DataAnimation | (embedded in MG prompt) | — |

### Schema Files

| File | Models Defined |
|------|---------------|
| `prompts/schema.py` | `ParsedEffectCues` (effects list + reasoning) |
| `prompts/motion_graphics_schema.py` | `MGComponentBounds`, `MGComponentSpec`, `MotionGraphicsPlanResponse` |
| `schemas/programmer.py` | `ProgrammerComponentSpec`, `ProgrammerPlanResponse`, `TemplatePlacement`, `TemplatePlacementResponse` |

## Prompt Loading

**File:** `helpers/llm.py`

```python
def load_prompt(name: str) -> str:
    """Load a prompt file from video_effects/prompts/"""
```

Activities load prompts, perform string substitution for dynamic sections (`{STYLE_GUIDE}`, `{TEMPLATES}`, `{API_REFERENCE}`, `{EXAMPLES}`), then call the LLM.

## Structured Output via Tool-Use

**Function:** `call_structured(system_prompt, user_message, response_model, model)`

Implementation:
1. Extract JSON schema from Pydantic `response_model` via `model_json_schema()`
2. Create Anthropic tool named `"structured_output"` with schema as `input_schema`
3. Force tool use via `tool_choice`
4. Parse `block.input` from the tool_use response block
5. Return validated dict

Max tokens: 4096.

If the LLM doesn't return a tool_use block, raises `ValueError("LLM did not return structured output via tool use")`.

## Raw Text Output

**Function:** `call_text(system_prompt, user_message, model)`

Used only for infographic code generation (A2). Returns raw string response.

Max tokens: 8192. Markdown fencing is stripped from the response.

## Model Selection

**File:** `config.py`

| Setting | Default | Used For |
|---------|---------|----------|
| `VFX_LLM_MODEL` | `claude-sonnet-4-6` | Effect parsing, style detection, MG planning, infographic planning |
| `VFX_INFOGRAPHIC_LLM_MODEL` | `claude-opus-4-6` | Infographic code generation (higher quality for code) |

## Prompt Details

### parse_effect_cues.md

**Purpose:** Infer video effects from transcript content and speaker energy — not from explicit commands.

**Key guidance:**
- Read energy, not words — identify natural emphasis moments
- 60–70% of effects should be zooms
- Match effect density to content pace
- Includes detailed rules for zoom pairing (in/out matching)
- Duration guidelines and confidence scoring per effect type

**Dynamic sections:** `{STYLE_GUIDE}` with density targets, color grading, preferred/avoided effects from the active [style preset](styles.md).

### parse_effect_cues_dev.md

**Purpose:** Dev mode variant — parse effects from explicit verbal cues ("zoom in", "blur the face", etc.) instead of inferred energy.

### design_style.md

**Purpose:** Auto-detect best visual style preset from transcript analysis.

**Analysis factors:** Speaker energy, content type, tone, video duration. Returns preset name + optional adjustments.

### plan_motion_graphics_base.md

**Purpose:** Plan animated overlay placements given transcript, face tracking data, and existing effects.

**Key rules:**
- **Spatial:** Never occlude face, use safe_regions, 10% edge margin, zoom viewport shrink, max 2 concurrent (non-edge-aligned)
- **Temporal:** Align to word timestamps, 0.1–0.3s delay after spoken cue, max 2 concurrent, 1–2s spacing
- **Density:** 6–10 per 60s, 12–18 per 2min, favor complex multi-element templates
- **Animation:** Spring entrances, staggered reveals, numeric interpolation

**Dynamic sections:**
- `{STYLE_GUIDE}` with palette, preferred/avoided animations, density range, template preferences
- `{TEMPLATES}` with full prop specs, duration ranges, placement constraints, and per-template creative guidance from `mg_guidance/*.md`

### Category Planning Prompts (6 parallel planners)

All category planners share the same structure: rules, data format, positioning guide, and a `{STYLE_GUIDE}` slot. They use the shared `_plan_category()` helper and return `InfographicPlanResponse` with a `score` per spec.

| Prompt | Category | Key Rules |
|--------|----------|-----------|
| `plan_infographics.md` | Charts, stat dashboards | Only concrete data, max 4, 3–8s |
| `plan_diagrams.md` | Flowcharts, mind maps, architectures | Clear structure (nodes + edges), max 3, 4–8s |
| `plan_timelines.md` | Chronological events, milestones | 3+ sequential events, max 3, 4–8s |
| `plan_quotes.md` | Key takeaways, citations, callouts | Genuinely impactful statements, max 3, 3–6s |
| `plan_code_blocks.md` | Code snippets, commands | Specific code mentioned in transcript, max 3, 4–8s |
| `plan_comparisons.md` | Side-by-side, A vs B, pros/cons | Clear two-sided structure, max 3, 4–8s |

### place_library_templates.md

**Purpose:** Context-aware placement of pinned library templates — find the best transcript moment, fill props with real content, position to avoid faces and existing components.

**Dynamic sections:**
- `{TEMPLATE_SECTIONS}` — rendered via `render_template_section()` from `helpers/templates.py` (prop table, duration range, spatial hints, creative guidance per template)
- `{EXISTING_COMPONENTS}` — formatted time windows of already-generated components (from ProgrammerWorkflow + MG plan)
- `{STYLE_GUIDE}` — palette and font from active style config

**Key rules:**
- Align template timing to relevant transcript moment
- Fill ALL required props with real content extracted from transcript
- Respect template's `duration_range` min/max
- Avoid subtitle zone (y ≥ 0.78), maintain ≥2s gap from existing components
- Face avoidance via spatial context

**Activity:** `vfx_place_library_templates` in `activities/programmer.py`
**Response model:** `TemplatePlacementResponse` (list of `TemplatePlacement`)
**Post-LLM validation:** Props checked against each template's `PropSpec` — clamp ranges, apply defaults, validate choices

### generate_infographic_code.md

**Purpose:** Generate complete TSX source code for one infographic component.

**Hard constraints:** Single named export, accept position + anchor props, call `useFaceAwareLayout()`, fade-out in last 0.5s, use `useStyle()`, inline styles only, SVG for data viz, no external imports beyond API reference, TypeScript strict.

**Dynamic sections:**
- `{API_REFERENCE}` with full `infographic_api_reference.md`
- `{EXAMPLES}` with real component code (`DataAnimation.tsx`, `AnimatedTitle.tsx`)

### mg_guidance/ Files

Per-template creative guidance loaded into the MG planning prompt:

| File | Content |
|------|---------|
| `animated_title.md` | When to use (topic transitions, key quotes), style matching (fade=calm, bounce=energetic), max 1 per 15–20s |
| `lower_third.md` | When to use (speaker intros, segment markers), slide=professional, fade=casual, avoid during zooms |
| `listicle.md` | When to use (explicit lists, steps), only animate mentioned items, 1–5 items max, stagger with speech |
| `data_animation.md` | When to use (numbers, stats), never fabricate data, min 2s for animation, counter=single stat, bar=comparisons |

## Feedback / Rejection Loops

### Effect Cue Parsing (5 rounds max)

```
LLM parses effects → CLI displays table → User approves/rejects
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                                 Approve             Reject
                                    │               + feedback
                                    ▼                   │
                               Continue              ───┘
                                                 Re-invoke LLM
                                                 with header:
                                        "## IMPORTANT: Previous
                                         attempt was rejected..."
```

### MG Planning (5 rounds max)

Same pattern as effect parsing. Feedback embedded in user message.

### Infographic Code Generation (3 rounds max)

```
LLM generates TSX → Validate (tsc + render)
                         │
               ┌─────────┴──────────┐
               │                    │
             Valid               Invalid
               │                    │
               ▼                    │
           Collect              ───┘
           component        Re-invoke LLM with:
                            "## RETRY (attempt N)"
                            + error messages
                            + previous code
                                    │
                            After 3 failures:
                            Fall back to template
```

## HITL Approval Flow

One approval gate in the main workflow, controlled by signals:

| Signal | Triggered By | Max Retries |
|--------|-------------|-------------|
| `approve_timeline(bool, feedback)` | CLI after displaying effect table | 5 |

Skippable via `--auto-approve` CLI flag (sets `VideoEffectsInput.auto_approve = True`).

CLI display functions:
- `_print_timeline()` — Pretty-prints effect table (type, timing, confidence, cue text)

Supports `json` input to view raw data.
