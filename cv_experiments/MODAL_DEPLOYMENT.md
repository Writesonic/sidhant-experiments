# Modal Deployment — Zoom-Bounce Processing

## Architecture Overview

The deployment is a **GPU-accelerated video processing service** that applies a "zoom-bounce" effect to videos, exposed via HTTP endpoints on Modal's serverless infrastructure.

### Two-file structure

- **`modal_config.py`** — shared config (single source of truth)
- **`modal_prod.py`** — production app with GPU classes + HTTP endpoints

---

## Image Build Pipeline

```
nvidia/cuda:12.2.0-devel-ubuntu22.04 (base)
  → apt: libgl1, libglib2.0-0, wget, xz-utils
  → ffmpeg: BtbN static build with NVENC (GPU-accelerated encoding)
  → pip: opencv, numpy, mediapipe, moviepy, cupy, PyNvVideoCodec
  → add_local_dir: cv_experiments source code baked into /root/cv_experiments
  → add_local_file: modal_config.py → /root/modal_config.py
```

Modal caches each layer, so rebuilds only re-run from the first changed step.

---

## GPU Tier System

Two `@app.cls` classes with identical processing logic but different GPU hardware:

| Tier | Class | GPU | Timeout | Scaledown |
|------|-------|-----|---------|-----------|
| `standard` | `ZoomBounceL4` | L4 | 30 min | 2 min warm |
| `premium` | `ZoomBounceL40S` | L40S | 30 min | 5 min warm |

Each class has:
- **`@modal.enter()` setup** — runs once per cold start, pre-imports `create_zoom_bounce_effect` so subsequent calls skip import overhead
- **`@modal.method() process`** — receives raw video bytes, writes to `/tmp`, runs the effect, returns result bytes

Both have `retries=Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0)` for transient failures.

---

## Auto-Routing & Fallback

**Routing** (`_route_request`):
1. If the request specifies `gpu_tier` explicitly → use that
2. Otherwise, auto-route by file size: ≥50 MB → `premium`, else → `standard`

**Fallback** (`_call_with_fallback`):
- Tries the chosen tier first
- On failure, walks `FALLBACK_ORDER = ["standard", "premium"]` from the chosen tier onward
- If all tiers fail → raises `RuntimeError`

This means a `standard` failure retries on `premium`. A `premium` failure has no fallback (it's last in the list).

---

## Storage Mounts

Every function/class gets two storage backends:

1. **Modal Volume** (`/data`) — defined in config, available for future use
2. **S3 CloudBucketMount** (`/s3data`) — the `ai-video-actions-*` bucket, mounted as a filesystem path. Files in S3 appear as local files at `/s3data/...`. Read and write operations translate to S3 API calls transparently. The `aws-s3-creds` Modal Secret injects `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` as env vars.

---

## HTTP Endpoints

All four are FastAPI endpoints served by Modal's web infrastructure:

### `POST /process` (sync)
1. Resolves input: downloads from `input_url` or reads `input_path`
2. Routes to a GPU tier
3. Calls `processor.process.remote(...)` — blocks until done
4. Returns the MP4 as a streaming download with `X-GPU-Tier` header

### `POST /submit` (async)
1. Same input resolution + routing
2. Calls `processor.process.spawn(...)` — returns immediately
3. Returns `{ call_id, status: "submitted", gpu_tier }` for polling

### `GET /status?call_id=...` (poll)
- `FunctionCall.from_id(call_id).get(timeout=0)`
- Returns `completed`, `pending`, or `failed`

### `GET /job_result?call_id=...` (download)
- Blocks until the spawned call finishes
- Returns the MP4 as a streaming download

---

## Local Entrypoint

```bash
modal run modal_prod.py --input-file video.mp4 [--gpu-tier premium]
```

Reads a local file, sends its bytes to the deployed GPU class via `.remote()`, and saves the result locally to `cv_experiments/outputs/`.

---

## External Access

`get_remote_processor(tier)` lets any Python script with Modal auth call the deployed classes:

```python
from modal_prod import get_remote_processor
proc = get_remote_processor("premium")
result = proc.process.remote(input_bytes=b"...", output_filename="out.mp4")
```

Uses `modal.Cls.from_name()` to look up the already-deployed app by name.

---

## Request Flow (sync example)

```
Client POST /process
  → Modal spins up web container (image, no GPU)
  → _resolve_input_bytes: downloads video from URL
  → _route_request: picks "standard" (< 50MB)
  → _call_with_fallback:
      → ZoomBounceL4().process.remote(bytes, filename)
          → Modal provisions L4 GPU container (or reuses warm one)
          → @modal.enter: imports create_zoom_bounce_effect
          → writes bytes to /tmp, runs effect, reads result
          → returns result bytes
  → StreamingResponse back to client
```

The key insight: **web endpoints run on cheap CPU containers**, while the heavy GPU work is dispatched via `.remote()` / `.spawn()` to the GPU-class containers. This keeps costs down since the web layer doesn't need a GPU.

---

## Deployment

```bash
# Create the S3 secret (one-time)
modal secret create aws-s3-creds \
  AWS_ACCESS_KEY_ID=<key> \
  AWS_SECRET_ACCESS_KEY=<secret> \
  AWS_REGION=us-east-1

# Deploy
modal deploy modal_prod.py
```
