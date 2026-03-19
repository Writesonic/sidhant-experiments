"""FastAPI proxy for the Video Effects Temporal workflow.

Run with: uvicorn video_effects.api:app --port 8000
"""

import os
import re
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from temporalio.client import Client
from temporalio.common import QueryRejectCondition
from temporalio.contrib.pydantic import pydantic_data_converter

from video_effects.config import settings
from video_effects.schemas.workflow import VideoEffectsInput

app = FastAPI(title="VFX Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_client: Client | None = None


async def _get_client() -> Client:
    global _client
    if _client is None:
        _client = await Client.connect(
            settings.TEMPORAL_ENDPOINT,
            namespace=settings.TEMPORAL_NAMESPACE,
            data_converter=pydantic_data_converter,
        )
    return _client


# ── POST /api/workflows ──


@app.post("/api/workflows")
async def start_workflow(body: dict[str, Any]) -> dict:
    video_path = body.get("video_path", "")
    if not video_path:
        raise HTTPException(400, "video_path is required")

    video_path = os.path.abspath(video_path)
    base = video_path.rsplit(".", 1)[0]
    output_path = f"{base}_effects.mp4"

    input_data = VideoEffectsInput(
        input_video=video_path,
        output_video=output_path,
        auto_approve=False,
        enable_motion_graphics=body.get("enable_mg", False) or body.get("enable_programmer", False),
        style=body.get("style", ""),
        dev_mode=body.get("dev_mode", False),
        enable_infographics=body.get("enable_mg", False),
        enable_programmer=body.get("enable_programmer", False),
    )

    workflow_id = f"vfx-web-{uuid.uuid4().hex[:8]}"
    client = await _get_client()
    await client.start_workflow(
        "VideoEffectsWorkflow",
        input_data,
        id=workflow_id,
        task_queue=settings.TASK_QUEUE,
    )

    return {"workflow_id": workflow_id}


# ── GET /api/workflows/{id} ──


async def _query(handle, name: str, timeout: timedelta = timedelta(seconds=10)):
    return await handle.query(name, rpc_timeout=timeout)


@app.get("/api/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str) -> dict:
    client = await _get_client()
    handle = client.get_workflow_handle(workflow_id)

    try:
        stage = await _query(handle, "get_workflow_stage")
    except Exception as e:
        err = str(e)
        if "not found" in err.lower():
            raise HTTPException(404, f"Workflow not found: {e}")
        # Query timeout likely means worker is busy — return a transient stage
        return {"stage": "loading"}

    result: dict[str, Any] = {"stage": stage}

    try:
        if stage == "timeline_approval":
            result["timeline"] = await _query(handle, "get_timeline")

        elif stage == "mg_approval":
            result["mg_plan"] = await _query(handle, "get_mg_plan")
            result["video_info"] = await _query(handle, "get_video_info")
            result["video_paths"] = await _query(handle, "get_video_paths")

        elif stage == "done":
            wf_result = await handle.result()
            if isinstance(wf_result, dict):
                result["result"] = wf_result
            else:
                result["result"] = wf_result.model_dump()

        elif stage == "error":
            wf_result = await handle.result()
            error = wf_result.get("error") if isinstance(wf_result, dict) else wf_result.error
            result["error"] = error
    except Exception as e:
        result.setdefault("error", str(e))

    return result


# ── POST /api/workflows/{id}/signal ──


@app.post("/api/workflows/{workflow_id}/signal")
async def signal_workflow(workflow_id: str, body: dict[str, Any]) -> dict:
    signal_name = body.get("signal")
    args = body.get("args", [])

    if signal_name not in ("approve_timeline", "approve_mg"):
        raise HTTPException(400, f"Unknown signal: {signal_name}")

    client = await _get_client()
    handle = client.get_workflow_handle(workflow_id)

    try:
        await handle.signal(signal_name, args)
    except Exception as e:
        raise HTTPException(500, f"Signal failed: {e}")

    return {"ok": True}


# ── GET /api/files ──


def _is_path_allowed(path: str) -> bool:
    resolved = os.path.realpath(path)
    for allowed in settings.ALLOWED_FILE_DIRS:
        if resolved.startswith(os.path.realpath(allowed)):
            return True
    return False


@app.get("/api/files")
async def serve_file(request: Request, path: str = Query(...)):
    resolved = os.path.realpath(path)

    if not _is_path_allowed(resolved):
        raise HTTPException(403, "Path not in allowed directories")

    if not os.path.isfile(resolved):
        raise HTTPException(404, "File not found")

    file_size = os.path.getsize(resolved)

    # Determine content type
    ext = Path(resolved).suffix.lower()
    content_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
    }
    content_type = content_types.get(ext, "application/octet-stream")

    # Handle Range requests for video seeking
    range_header = request.headers.get("range")
    if range_header and content_type.startswith("video/"):
        match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if not match:
            raise HTTPException(416, "Invalid range")

        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else file_size - 1
        end = min(end, file_size - 1)

        if start >= file_size:
            raise HTTPException(416, "Range not satisfiable")

        length = end - start + 1

        def iter_range():
            with open(resolved, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
                "Accept-Ranges": "bytes",
            },
        )

    def iter_file():
        with open(resolved, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type=content_type,
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
        },
    )
