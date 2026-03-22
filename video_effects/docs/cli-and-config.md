# CLI, Web UI & Config

## Web UI (Primary)

The web UI is a Next.js app that communicates with the workflow via a FastAPI proxy. MG previews render directly in the browser using `@remotion/player` — no CLI preview rendering needed.

```bash
# 1. Start the Temporal worker
python -m video_effects.worker

# 2. Start the API server (from repo root)
uvicorn video_effects.api:app --port 8000 --reload

# 3. Start the web UI
cd video_effects/app && npm run dev

# 4. Open http://localhost:3000
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/workflows` | POST | Start workflow (`{video_path, enable_programmer, enable_mg, style, dev_mode, enable_subtitles, pinned_templates}`) |
| `/api/workflows/{id}` | GET | Get stage + stage-specific data (timeline, mg_plan, video_paths, result) |
| `/api/workflows/{id}/signal` | POST | Send approval/rejection signal (`{signal, args}`) |
| `/api/files` | GET | Stream local file by path (supports HTTP Range requests for video seeking) |
| `/api/templates` | GET | List all library templates |
| `/api/templates` | POST | Create a new library template |
| `/api/templates/{id}` | GET | Get a single library template |
| `/api/templates/{id}` | PUT | Update a library template |
| `/api/templates/{id}` | DELETE | Delete a library template |
| `/api/templates/generate` | POST | Generate template code via LLM |

### Workflow Signals & Queries

| Signal | Args | Purpose |
|--------|------|---------|
| `approve_timeline` | `[bool, str]` | Approve/reject timeline with feedback |
| `approve_mg` | `[bool, str]` | Approve/reject MG plan with feedback (prefix `[component:N]` for per-component edits) |

| Query | Returns | Purpose |
|-------|---------|---------|
| `get_workflow_stage` | `str` | Current workflow stage |
| `get_timeline` | `dict` | Current effect timeline |
| `get_mg_plan` | `dict` | Current MG composition plan |
| `get_video_info` | `dict` | Video metadata (fps, width, height, duration) |
| `get_video_paths` | `dict` | Paths to base video, face data, zoom state |
| `get_mg_preview` | `dict \| None` | MG preview data for browser rendering |

## CLI Commands (Alternative)

**Entry point:** `python -m video_effects.cli run`

```
python -m video_effects.cli run <input_video> [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `input_video` | — | positional | required | Input video file path |
| `--output` | `-o` | str | `{input}_effects.mp4` | Output video file path |
| `--auto-approve` | — | flag | False | Skip interactive approval steps |
| `--motion-graphics` | `--mg` | flag | False | Enable code-gen motion graphics overlays (routes through infographic pipeline) |
| `--style` | `-s` | choice | auto-detect | Style preset name (see [Styles](styles.md)) |
| `--dev` | — | flag | False | Dev mode: effects from explicit verbal commands |
| `--infographics` | — | flag | False | Enable LLM-generated infographic overlays (same as `--mg`) |
| `--programmer` | — | flag | False | Enable free-hand creative programmer workflow |
| `--subtitles` | — | flag | False | Enable subtitle overlay from transcript |

### Examples

```bash
# Basic effects only
python -m video_effects.cli run interview.mp4

# Full pipeline with motion graphics
python -m video_effects.cli run talk.mp4 -o talk_final.mp4 --mg --style tech-sleek

# Everything, auto-approved
python -m video_effects.cli run vlog.mp4 -o vlog_final.mp4 --mg --infographics --auto-approve

# Dev mode (effects triggered by speaker saying "zoom in", etc.)
python -m video_effects.cli run demo.mp4 --dev
```

## Interactive Approval Flow

When `--auto-approve` is not set, the workflow pauses at two gates:

### 1. Timeline Approval

After the LLM parses effect cues and validates the timeline:

```
┌──────────────────────────────────────────────────────┐
│ #  Type          Start    End      Conf   Cue        │
│ 1  zoom          0:03.2   0:05.8   0.90   emphasis   │
│ 2  color_change  0:10.0   0:25.0   0.85   warm       │
│ 3  whip          0:28.5   0:29.5   0.80   transition │
│ ...                                                   │
│                                                       │
│ Total: 8 effects | Conflicts resolved: 1              │
└──────────────────────────────────────────────────────┘

Approve timeline? [y/n/json]:
```

- `y` / `yes` — Approve and continue
- `n` / `no` — Reject; you'll be prompted for feedback
- `json` — View raw timeline JSON

Up to 5 rejection rounds. Feedback is passed back to the LLM for re-planning.

### 2. MG Plan Approval

After motion graphics components are generated (if `--mg` or `--programmer` is enabled), the workflow pauses for review. In the web UI, `@remotion/player` renders the full composition (base video + overlays) directly in the browser. Per-component edit and remove actions are available.

Up to 5 rejection rounds. Feedback prefixed with `[component:N]` targets a specific component.

## Configuration

**File:** `config.py`

All settings are loaded via Pydantic `BaseSettings` with `VFX_` prefix. Set them as environment variables.

### Environment Variables

#### Temporal

| Variable | Default | Description |
|----------|---------|-------------|
| `VFX_TASK_QUEUE` | `video_effects_queue` | Temporal task queue name |
| `VFX_TEMPORAL_NAMESPACE` | `default` | Temporal namespace |
| `VFX_TEMPORAL_ENDPOINT` | `localhost:7233` | Temporal server endpoint |
| `VFX_TEMPORAL_API_KEY` | `None` | API key for Temporal Cloud |

#### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `VFX_ANTHROPIC_API_KEY` | `None` | Claude API key |
| `VFX_LLM_MODEL` | `claude-sonnet-4-6` | Model for most LLM tasks |
| `VFX_SMALL_LLM_MODEL` | `claude-haiku-4-5` | Lightweight model for simple tasks |
| `VFX_INFOGRAPHIC_LLM_MODEL` | `claude-opus-4-6` | Model for infographic code generation |

#### Face Tracking

| Variable | Default | Description |
|----------|---------|-------------|
| `VFX_FACE_LANDMARKER_PATH` | `cv_experiments/face_landmarker.task` | MediaPipe model path |
| `VFX_FACE_DETECTION_STRIDE` | `3` | Detect every Nth frame |
| `VFX_SMOOTHING_ALPHA` | `0.1` | EMA smoothing factor |

#### Remotion

| Variable | Default | Description |
|----------|---------|-------------|
| `VFX_REMOTION_DIR` | auto-detected | Path to `remotion/` project |
| `VFX_REMOTION_CONCURRENCY` | Remotion default | Render parallelism |

#### Infographics

| Variable | Default | Description |
|----------|---------|-------------|
| `VFX_INFOGRAPHIC_MAX_RETRIES` | `3` | Max code-gen + validate attempts |

#### Programmer

| Variable | Default | Description |
|----------|---------|-------------|
| `VFX_PROGRAMMER_LLM_MODEL` | `claude-opus-4-6` | Model for programmer code generation |
| `VFX_PROGRAMMER_MAX_RETRIES` | `3` | Max code-gen + validate attempts |

#### Template Library

| Variable | Default | Description |
|----------|---------|-------------|
| `VFX_TEMPLATE_LIBRARY_PATH` | `data/template_library.json` | Path to template library JSON |

#### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `VFX_API_PORT` | `8000` | API server port |
| `VFX_API_CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `VFX_ALLOWED_FILE_DIRS` | `["/tmp/video_effects", ...]` | Directories the `/api/files` endpoint can serve |

#### Misc

| Variable | Default | Description |
|----------|---------|-------------|
| `VFX_TEMP_DIR` | `/tmp/video_effects` | Temporary directory for intermediates |
| `VFX_ELEVENLABS_API_KEY` | `None` | ElevenLabs API key (transcription) |

## Temporal Worker Setup

**File:** `worker.py`

```bash
python -m video_effects.worker
```

### Configuration

- **Client:** Connects to `VFX_TEMPORAL_ENDPOINT` with `VFX_TEMPORAL_NAMESPACE`
- **Task queue:** `VFX_TASK_QUEUE`
- **Thread pool:** 4 max workers (`ThreadPoolExecutor`)
- **Data converter:** `pydantic_data_converter`

### Registered Workflows

| Workflow | Purpose |
|----------|---------|
| `VideoEffectsWorkflow` | Main pipeline (G1–G9) |
| `CreativeDesignerWorkflow` | Auto-style detection |
| `InfographicGeneratorWorkflow` | Code generation (A0–A4) |
| `ProgrammerWorkflow` | Free-hand creative component generation |

### Registered Activities

All activities from `ALL_VIDEO_EFFECTS_ACTIVITIES` plus `design_style`. See [Architecture](architecture.md) for the full list.

## Workflow Input / Output

### Input (`VideoEffectsInput`)

```python
input_video: str
output_video: str
auto_approve: bool
enable_motion_graphics: bool
style: str
dev_mode: bool
enable_infographics: bool
enable_programmer: bool
enable_subtitles: bool
pinned_templates: list[dict]
```

### Output (`VideoEffectsOutput`)

```python
output_video: str
effects_applied: int
transcript_length: int
phases_executed: int
motion_graphics_applied: int
error: Optional[str]
```

## Timeouts

| Activity Type | Start-to-Close | Heartbeat |
|---------------|----------------|-----------|
| Standard | 10 minutes | — |
| Long render | 30 minutes | 5 minutes |
| Video render | 30 minutes | 2 minutes |
| HITL approval | 10 minutes | — |

## Infographic Test CLI

Standalone test harness that runs the infographic pipeline directly (no Temporal):

```bash
python -m video_effects.test_infographic --text "transcript..."
python -m video_effects.test_infographic --file transcript.txt
python -m video_effects.test_infographic --spec spec.json --retries 5
python -m video_effects.test_infographic --skip-validate --model claude-sonnet-4-6
```

| Flag | Description |
|------|-------------|
| `--text` | Inline transcript |
| `--file` | Transcript file path |
| `--spec` | Pre-made InfographicSpec JSON (skip A1) |
| `--type` | Filter to single infographic type |
| `--retries` | Max code-gen retries (default 3) |
| `--skip-validate` | Skip tsc + render validation |
| `--model` | Override LLM model |
| `--style` | Style preset name |
| `--video-duration` | Fake duration (default 60s) |
| `--video-fps` | Fake FPS (default 30) |
| `--no-show-code` | Don't print generated TSX |
