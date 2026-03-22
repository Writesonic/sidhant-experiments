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
from video_effects.schemas.template_library import (
    LibraryTemplate,
    list_templates as _list_templates,
    get_template as _get_template,
    save_template as _save_template,
    delete_template as _delete_template,
)

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

    pinned_ids = body.get("pinned_templates", [])
    resolved_pinned: list[dict] = []
    for tid in pinned_ids:
        tpl = _get_template(tid)
        if tpl:
            resolved_pinned.append({
                "id": tpl.id,
                "display_name": tpl.display_name,
                "spatial": tpl.spatial.model_dump(),
                "duration_range": list(tpl.duration_range),
            })

    input_data = VideoEffectsInput(
        input_video=video_path,
        output_video=output_path,
        auto_approve=False,
        enable_motion_graphics=body.get("enable_mg", False) or body.get("enable_programmer", False),
        style=body.get("style", ""),
        dev_mode=body.get("dev_mode", False),
        enable_infographics=body.get("enable_mg", False),
        enable_programmer=body.get("enable_programmer", False),
        enable_subtitles=body.get("enable_subtitles", False),
        pinned_templates=resolved_pinned,
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
        # Always include video_paths so the UI can show the video at every stage
        result["video_paths"] = await _query(handle, "get_video_paths")

        try:
            result["steps"] = await _query(handle, "get_steps")
        except Exception:
            pass  # Backward compat with old workflows

        if stage == "timeline_approval":
            result["timeline"] = await _query(handle, "get_timeline")

        elif stage == "mg_approval":
            result["mg_plan"] = await _query(handle, "get_mg_plan")
            result["video_info"] = await _query(handle, "get_video_info")

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


# ── Template Library CRUD ──


@app.get("/api/templates")
async def list_templates_endpoint() -> list[dict]:
    templates = _list_templates()
    return [
        {
            "id": t.id,
            "display_name": t.display_name,
            "description": t.description,
            "tags": t.tags,
            "created_at": t.created_at,
            "export_name": t.export_name,
            "duration_range": t.duration_range,
            "tsx_code": t.tsx_code,
        }
        for t in templates
    ]


@app.get("/api/templates/{template_id}")
async def get_template_endpoint(template_id: str) -> dict:
    tpl = _get_template(template_id)
    if tpl is None:
        raise HTTPException(404, f"Template not found: {template_id}")
    return tpl.model_dump()


@app.post("/api/templates")
async def create_template_endpoint(body: dict[str, Any]) -> dict:
    tpl = LibraryTemplate(**body)
    saved = _save_template(tpl)
    return saved.model_dump()


@app.put("/api/templates/{template_id}")
async def update_template_endpoint(template_id: str, body: dict[str, Any]) -> dict:
    existing = _get_template(template_id)
    if existing is None:
        raise HTTPException(404, f"Template not found: {template_id}")
    updated_data = existing.model_dump()
    updated_data.update(body)
    updated_data["id"] = template_id
    tpl = LibraryTemplate(**updated_data)
    saved = _save_template(tpl)
    return saved.model_dump()


@app.delete("/api/templates/{template_id}")
async def delete_template_endpoint(template_id: str) -> dict:
    if not _delete_template(template_id):
        raise HTTPException(404, f"Template not found: {template_id}")
    return {"ok": True}


@app.post("/api/templates/generate")
async def generate_template_code(body: dict[str, Any]) -> dict:
    from video_effects.helpers.llm import call_text, load_prompt
    from video_effects.helpers.remotion import _get_remotion_dir

    prompt_text = body.get("prompt", "")
    previous_code = body.get("previous_code", "")
    conversation = body.get("conversation", [])
    errors = body.get("errors", [])

    if not prompt_text:
        raise HTTPException(400, "prompt is required")

    base_prompt = load_prompt("generate_template.md")
    api_reference = load_prompt("infographic_api_reference.md")

    # Load real component examples
    components_dir = _get_remotion_dir() / "src" / "components"
    examples = []
    for name in ("DataAnimation.tsx", "AnimatedTitle.tsx"):
        path = components_dir / name
        if path.exists():
            examples.append(f"### Example: {name}\n\n```tsx\n{path.read_text()}\n```")
    examples_section = "## Real Component Examples\n\n" + "\n\n".join(examples) if examples else ""

    system_prompt = (
        base_prompt
        .replace("{API_REFERENCE}", f"## API Reference\n\n{api_reference}")
        .replace("{EXAMPLES}", examples_section)
    )

    # Build user message
    lines = [f"Create this component: {prompt_text}"]
    if previous_code:
        lines.append(f"\n## Previous Code\n\n```tsx\n{previous_code}\n```")
    if errors:
        lines.append("\n## Compilation Errors (fix these)\n")
        for err in errors:
            lines.append(f"- {err}")
    if conversation:
        lines.append("\n## Conversation History\n")
        for msg in conversation[-6:]:
            lines.append(f"**{msg.get('role', 'user')}**: {msg.get('content', '')}")

    user_message = "\n".join(lines)

    raw_code = call_text(
        system_prompt=system_prompt,
        user_message=user_message,
        model=settings.INFOGRAPHIC_LLM_MODEL,
    )

    # Strip markdown fencing if present
    code = raw_code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```\w*\n?", "", code)
        code = re.sub(r"\n?```$", "", code)

    summary = f"Generated component for: {prompt_text[:100]}"
    return {"code": code, "summary": summary}


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
