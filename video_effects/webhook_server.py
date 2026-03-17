"""FastAPI webhook receiver for GPU service async completions.

Receives webhook callbacks from Modal GPU services (SAM segmentation, etc.)
and completes the corresponding Temporal async activities.

Usage:
    uvicorn video_effects.webhook_server:app --port 8001
"""

import base64
import logging

from fastapi import FastAPI, Request
from pydantic import BaseModel
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from video_effects.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="VFX Webhook Receiver")

_temporal_client: Client | None = None


async def _get_temporal_client() -> Client:
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await Client.connect(
            settings.TEMPORAL_ENDPOINT,
            namespace=settings.TEMPORAL_NAMESPACE,
            data_converter=pydantic_data_converter,
        )
    return _temporal_client


class WebhookPayload(BaseModel):
    task_token: str
    status: str  # "success" or "failure"
    result: dict | None = None
    error: str | None = None


@app.post("/webhook/gpu-complete")
async def gpu_complete(request: Request):
    """Receive GPU service completion webhook and complete Temporal async activity."""
    body = await request.body()
    payload = WebhookPayload.model_validate_json(body)
    task_token = base64.b64decode(payload.task_token)

    client = await _get_temporal_client()
    handle = client.get_async_activity_handle(task_token=task_token)

    if payload.status == "success":
        logger.info("Completing activity (success): result keys=%s", list((payload.result or {}).keys()))
        await handle.complete(payload.result or {})
    else:
        error_msg = payload.error or "GPU service reported failure"
        logger.warning("Failing activity: %s", error_msg)
        await handle.fail(Exception(error_msg))

    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
