# video_effects Documentation

Automated video post-production powered by LLM analysis, OpenCV frame processing, and Remotion motion graphics — orchestrated by Temporal workflows.

## Quick Start

```bash
# 1. Start the Temporal worker
python -m video_effects.worker

# 2. Start the API server
uvicorn video_effects.api:app --port 8000 --reload

# 3. Start the web UI
cd video_effects/app && npm run dev

# 4. Open http://localhost:3000, enter a video path, and start a workflow

# CLI alternative (auto-approve mode, no web UI needed):
python -m video_effects.cli run input.mp4 -o output.mp4 --mg --auto-approve --style bold-energy
```

## How It Works

1. **Transcribe** the video audio (ElevenLabs / Whisper)
2. **Analyze** the transcript with an LLM to infer effects cues
3. **Apply** OpenCV effects in a single-pass frame pipeline
4. **Plan** motion graphics overlays using LLM + face tracking context
5. **Render** transparent overlays via Remotion (ProRes 4444)
6. **Composite** everything with FFmpeg

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System overview, pipeline stages, end-to-end data flow |
| [Concurrency & Deployment](concurrency-and-deployment.md) | Threading model, heartbeat mechanics, Porter K8s + Modal GPU deployment |
| [Effects Pipeline](effects-pipeline.md) | OpenCV frame processing: phase ordering, each effect's internals, encoder settings |
| [Motion Graphics](motion-graphics.md) | Remotion MG system: LLM planning, spatial validation, ProRes rendering, FFmpeg compositing |
| [Infographics](infographics.md) | Code-gen pipeline: plan → generate TSX → validate → registry → fallback |
| [Remotion Components](remotion-components.md) | Component implementations, animation patterns, hooks API, props interfaces |
| [Face Tracking](face-tracking.md) | Face detection, spatial context, safe regions, zoom compensation, anchor modes |
| [Styles](styles.md) | Style presets, theming system, font loading, palette conventions, creative designer |
| [LLM Prompts](llm-prompts.md) | Prompt system, structured output, feedback loops, model selection |
| [CLI & Config](cli-and-config.md) | CLI commands, web UI, API server, interactive approval flow, environment variables, Temporal worker setup |

## Project Structure

```
video_effects/
├── cli.py                         # CLI entry point
├── api.py                         # FastAPI proxy for web UI
├── worker.py                      # Temporal worker (skill discovery, ThreadPoolExecutor)
├── config.py                      # Settings (VFX_ env vars, RUNTIME_MODE)
├── workflow.py                    # Main VideoEffectsWorkflow
├── creative_workflow.py           # Style auto-detection child workflow
├── infographic_workflow.py        # Code-gen child workflow
├── programmer_workflow.py         # Free-hand creative programmer child workflow
├── effect_registry.py             # Phase ordering & effect type → processor map
│
├── core/                          # Skill-capability abstractions (zero Temporal imports)
│   ├── base/
│   │   ├── capability.py          # BaseCapability[TRequest, TResponse], MockProgressReporter
│   │   └── context.py             # ExecutionContext (frozen dataclass), create_test_context()
│   └── interfaces/
│       └── progress.py            # ProgressReporter protocol, ProgressReport, EventType
│
├── infrastructure/                # Runtime provider layer
│   ├── provider.py                # run_capability / run_capability_sync, mode switching
│   ├── temporal/
│   │   └── implementations.py     # TemporalContextProvider, TemporalProgressReporter, wrap_as_temporal_activity
│   └── testing/
│       └── implementations.py     # TestContextProvider, RecordingProgressReporter
│
├── skills/                        # Each skill: activities.py + capabilities/ + schemas.py + skill.yml
│   ├── discovery.py               # Scans for skill.yml, dynamic import of activities
│   ├── registry.py                # @register_activity decorator, queue routing
│   ├── video_extraction/          # Video metadata extraction
│   ├── transcription/             # Audio transcription (ElevenLabs / Whisper)
│   ├── creative/                  # Style design activity
│   ├── effect_planning/           # LLM effect cue parsing + timeline validation
│   ├── face_detection/            # Face detection + Remotion spatial context
│   ├── mg_planning/               # MG planning, template placement
│   ├── rendering/                 # Single-pass OpenCV frame pipeline
│   ├── composition/               # FFmpeg compositing + audio mux
│   ├── studio/                    # Remotion Studio lifecycle
│   ├── infographic/               # Infographic code generation & validation
│   └── programmer/                # Programmer brainstorm, critique, code-gen
│
├── activities/
│   ├── __init__.py                # Discovery shim (load_all_skills → ALL_VIDEO_EFFECTS_ACTIVITIES)
│   └── remotion.py                # Legacy MG helpers (shared across mg_planning/face_detection)
│
├── effects/                       # OpenCV frame processors
│   ├── base.py                    # BaseEffect ABC
│   ├── zoom.py                    # Face-tracked zoom with easing
│   ├── blur.py                    # Gaussian, face pixelate, background, radial
│   ├── color.py                   # Color grading presets
│   ├── whip.py                    # Whip transition
│   ├── vignette.py                # Cinematic vignette
│   └── speed_ramp.py              # Visual speed effect
│
├── helpers/                       # Shared utilities (no Temporal imports)
│   ├── llm.py                     # Anthropic API wrapper (call_structured, call_text)
│   ├── face_tracking.py           # MediaPipe face detection pipeline
│   ├── remotion.py                # Remotion render + FFmpeg composite helpers
│   ├── templates.py               # Shared template metadata renderer
│   ├── studio.py                  # Remotion Studio process lifecycle management
│   ├── prompts.py                 # Shared prompt builders (style guide, spatial message, etc.)
│   └── effects.py                 # Effect validation utilities (validate_zoom_pairs)
│
├── schemas/                       # Pydantic models
│   ├── effects.py                 # EffectCue, EffectType, VideoInfo, effect params
│   ├── styles.py                  # StylePreset, StyleConfig, FontWeights
│   ├── mg_templates.py            # MG template registry & specs
│   ├── motion_graphics.py         # MotionGraphicsComponent, Plan
│   ├── programmer.py              # ProgrammerComponentSpec, TemplatePlacement models
│   ├── template_library.py        # User-created library template CRUD + conversion
│   ├── infographic.py             # InfographicSpec, InfographicType, fallback map
│   └── workflow.py                # VideoEffectsInput/Output
│
├── prompts/                       # LLM prompt templates
│   ├── parse_effect_cues.md
│   ├── parse_effect_cues_dev.md
│   ├── design_style.md
│   ├── plan_motion_graphics_base.md
│   ├── plan_infographics.md
│   ├── plan_diagrams.md
│   ├── plan_timelines.md
│   ├── plan_quotes.md
│   ├── plan_code_blocks.md
│   ├── plan_comparisons.md
│   ├── programmer_brainstorm.md
│   ├── programmer_critique.md
│   ├── programmer_generate_code.md
│   ├── place_library_templates.md
│   ├── summarize_transcript.md
│   ├── edit_mg_plan.md
│   ├── generate_template.md
│   ├── generate_infographic_code.md
│   ├── infographic_api_reference.md
│   ├── schema.py
│   ├── motion_graphics_schema.py
│   └── mg_guidance/
│
├── app/                           # Next.js web UI (preview + approval)
│   ├── package.json               # Next.js 16 + @remotion/player 4.0.242
│   ├── next.config.ts
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx
│       │   ├── workflow/[id]/page.tsx
│       │   └── templates/
│       ├── components/
│       ├── hooks/
│       └── lib/
│
├── remotion/                      # TypeScript/React Remotion project
│   ├── package.json               # Remotion 4.0.242 + React 18
│   └── src/
│       ├── Root.tsx
│       ├── DynamicComposition.tsx
│       ├── types.ts
│       ├── components/
│       │   └── generated/
│       └── lib/
│           ├── spatial.ts         # useFaceAwareLayout, zoom compensation
│           ├── styles.ts          # StyleProvider, useStyle
│           ├── context.ts         # FaceDataProvider, useFaceFrame
│           ├── zoom-context.ts    # ZoomDataProvider, useZoomFrame
│           ├── easing.ts          # Spring configs
│           ├── fonts.ts           # Google Fonts loader
│           ├── infographic-utils.ts
│           └── component-utils.ts
│
└── docs/
    ├── architecture.md
    ├── concurrency-and-deployment.md
    ├── effects-pipeline.md
    ├── face-tracking.md
    ├── motion-graphics.md
    ├── infographics.md
    ├── remotion-components.md
    ├── styles.md
    ├── llm-prompts.md
    └── cli-and-config.md
```
