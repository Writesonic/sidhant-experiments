# Concurrency, Threading & Deployment

## How the Temporal Worker Executes Activities

### Three Execution Layers

```
Temporal Core (Rust bridge, own threads)
  Polls server for workflow + activity tasks
  Sends heartbeats over gRPC
  Manages task tokens & cancellation
          |
Python Event Loop (1 thread)
  Workflow replay & decision-making
  Heartbeat I/O delivery to Rust bridge
  Activity task dispatching
  Awaits ThreadPoolExecutor futures
          |
ThreadPoolExecutor (10 threads)
  All activity execution happens here
  Each sync activity gets its own thread
```

### All Activities Are Sync

Every activity in this project is a sync `def` function. Temporal dispatches them to the `ThreadPoolExecutor`. This matches the original pre-refactor behavior and keeps the event loop free for workflow replay and heartbeat I/O.

The capabilities themselves are `async def execute()` but are invoked via `run_capability_sync()` which creates a throwaway `asyncio.new_event_loop()` per call. This means each activity runs its own mini event loop inside its thread.

### How `activity.heartbeat()` Works From a Thread

1. Activity calls `activity.heartbeat("frame 50/884")` from its executor thread
2. SDK calls `asyncio.run_coroutine_threadsafe(heartbeat_coro, main_event_loop)` - posts to the main event loop
3. Blocks the calling thread for up to 10 seconds waiting for the event loop to process it
4. Event loop picks it up, sends to Rust bridge, which sends to Temporal server over gRPC
5. If the event loop is overloaded, heartbeat delivery is delayed

If heartbeat delivery is delayed past the server-side heartbeat timeout (configured per-activity in the workflow), the server cancels the activity and schedules a retry.

### Why Activities Must Be Sync (Not Async)

Async activities run directly on the event loop. If an async activity does blocking work (CPU-bound frame processing, synchronous HTTP calls via `run_capability_sync`), it starves the event loop. This blocks:
- Workflow replay
- Heartbeat I/O for ALL other activities
- Activity task dispatching

With sync activities in the thread pool, the event loop stays free. This is why `vfx_render_video` heartbeats work reliably.

## Peak Concurrency During a Workflow Run

```
Time ---------------------------------------------------------------->

G1-G5 (sequential):
  [get_info] [extract_audio] [transcribe] [parse_cues] [validate]
  1 thread at a time

G6+G8 (parallel via asyncio.gather in workflow.py line 432):
  Thread 1: ================================ vfx_render_video (5-15 min)
  Thread 2: ==== brainstorm ==== critique
  Thread 3:                                == gen_code[0] == validate[0]
  Thread 4:                                == gen_code[1] == validate[1]
  Thread 5:                                == gen_code[2] == validate[2]
                                           ^ Semaphore(3) caps at 3

Peak: 6 threads occupied simultaneously out of 10 available
```

### Activity Categories

| Activity | Type | Duration | Thread Usage |
|----------|------|----------|-------------|
| vfx_render_video | CPU-bound (ffmpeg/OpenCV) | 5-30 min | Holds 1 thread entire time |
| vfx_detect_faces | CPU-bound (MediaPipe) | 1-5 min | Holds 1 thread |
| vfx_programmer_brainstorm | I/O-bound (LLM API) | 2-5 min | Holds 1 thread (blocks on HTTP) |
| vfx_programmer_generate_code | I/O-bound (LLM API) | 1-3 min | Up to 3 concurrent (semaphore) |
| vfx_validate_infographic | Subprocess (tsc + Chromium) | 30-60s | 1 thread |
| vfx_render_motion_overlay | Subprocess (Remotion/Chromium) | 2-10 min | 1 thread |
| vfx_compose_final | Subprocess (ffmpeg) | 10-30s | 1 thread, quick |
| vfx_get_video_info | Subprocess (ffprobe) | <1s | 1 thread, instant |

### GIL Considerations

Python's GIL means only 1 thread runs Python code at a time. This is fine because:
- CPU-bound work is in C extensions (numpy, OpenCV) or subprocesses (ffmpeg) which release the GIL
- I/O-bound work (HTTP/LLM calls) releases GIL during network I/O
- `activity.heartbeat()` briefly acquires GIL but completes in microseconds

## Worker Configuration

```python
# worker.py
ThreadPoolExecutor(max_workers=10)  # Hard limit on concurrent activities

# Not explicitly set (defaults):
# max_concurrent_activities = 100   (Temporal can offer up to 100 tasks)
# max_concurrent_activity_task_polls = 5
# workflow_task_executor = ThreadPoolExecutor(max_workers=500)  (separate pool)
```

The thread pool (10) is the real bottleneck, not max_concurrent_activities (100). Temporal will offer more tasks than threads can handle; excess tasks queue in the executor.

## Scaling: Single Machine

| Scenario | Threads Used | Headroom |
|----------|-------------|----------|
| 1 workflow, no code gen | 1-2 | 80% |
| 1 workflow, programmer enabled | 6 peak | 40% |
| 2 concurrent workflows | 12 peak | Needs max_workers=15+ |
| 3+ concurrent workflows | Thread-starved | Needs multi-worker deployment |

## Deployment Architecture (Kubernetes + Modal)

### Queue-Per-Workload Split

The skill.yml `queue` field determines which worker handles each activity. Change queue assignments to split workloads across pods:

```yaml
# I/O-bound skills (orchestrator pod)
effect_planning/skill.yml:   queue: video_effects_queue
creative/skill.yml:          queue: video_effects_queue
transcription/skill.yml:     queue: video_effects_queue
composition/skill.yml:       queue: video_effects_queue
infographic/skill.yml:       queue: video_effects_queue
programmer/skill.yml:        queue: video_effects_queue
video_extraction/skill.yml:  queue: video_effects_queue
studio/skill.yml:            queue: video_effects_queue

# CPU-heavy Remotion rendering (dedicated pod)
mg_planning/skill.yml:       queue: remotion_queue

# GPU rendering (Modal serverless)
rendering/skill.yml:         queue: gpu_render_queue
face_detection/skill.yml:    queue: gpu_render_queue
```

### Three Worker Types

```
orchestrator-worker (always on, 1 replica)
  Image: python:3.13-slim (no Node.js, no OpenCV, no mediapipe)
  Queue: video_effects_queue
  Resources: 2 CPU, 4GB RAM
  Handles: LLM calls, ffprobe, light ffmpeg mux, file ops

remotion-worker (autoscale 0-3 replicas)
  Image: node:20 + python:3.13 + chromium + remotion project
  Queue: remotion_queue
  Resources: 8-16 CPU, 16GB RAM
  Handles: Remotion renders, TypeScript validation
  VFX_REMOTION_CONCURRENCY=12

gpu-render (Modal serverless, scale to zero)
  GPU: T4/A10, 8 CPU cores, 16GB RAM
  Handles: vfx_render_video, vfx_detect_faces
  Webhook callback when complete
  No idle cost
```

### Worker CLI

Each pod runs the worker filtered to its queue:

```bash
# Orchestrator
python -m video_effects.worker --queue video_effects_queue

# Remotion renderer
python -m video_effects.worker --queue remotion_queue
```

`get_activities_by_queue()` ensures each worker only loads relevant skills. The orchestrator never imports cv2/mediapipe/chromium.

### Shared File Storage

Activities across pods need shared access to video files:

| Option | Pros | Cons |
|--------|------|------|
| S3/GCS bucket | Works across K8s + Modal, durable | Upload/download latency |
| K8s PersistentVolumeClaim | Simple, low latency within cluster | K8s-only, doesn't reach Modal |
| Modal Volumes | Fast for GPU workers | Modal-only |

Recommended: S3 for cross-boundary (K8s <-> Modal), PVC for intra-cluster (orchestrator <-> remotion).

### Autoscaling

- **Orchestrator**: Fixed at 1 replica (I/O-bound, handles dozens of concurrent workflows)
- **Remotion workers**: KEDA or Porter autoscaler watching Temporal queue depth for `remotion_queue`. Scale 0->3 based on pending tasks.
- **Modal GPU**: Serverless, automatic scale-to-zero

### Cost Profile

| Component | Idle Cost | Per-Video Cost |
|-----------|-----------|---------------|
| Orchestrator (2 CPU) | ~$15/mo | Negligible |
| Remotion worker (0 replicas idle) | $0 | ~$0.05-0.10 per video (8 CPU * 5 min) |
| Modal GPU (T4) | $0 | ~$0.02-0.05 per video |
| Temporal Cloud | $25/mo base | $0.00025 per action |
| LLM API (Claude) | $0 | ~$0.10-0.50 per video |

## Future: Fully Async Orchestrator

Once GPU rendering moves to Modal and Remotion moves to dedicated pods, the orchestrator handles only I/O-bound work. At that point:

1. Convert activities back to `async def` with `await run_capability()`
2. Remove `run_capability_sync` and `ThreadPoolExecutor`
3. Single event loop handles hundreds of concurrent workflows
4. No threads needed — pure async I/O

This is the end state where the skill-capability architecture fully pays off: capabilities are async, activities are async, and the only blocking work happens on remote compute.
