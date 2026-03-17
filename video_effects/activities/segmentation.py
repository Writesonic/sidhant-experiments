"""Activities for SAM person segmentation via async webhook pattern."""

import base64
import logging

import httpx
from temporalio import activity

from video_effects.config import settings
from video_effects.helpers.s3 import download_file, upload_file

logger = logging.getLogger(__name__)


@activity.defn(name="vfx_segment_person")
async def segment_person(input_data: dict) -> dict:
    """Upload video to S3, POST to SAM service, then raise CompleteAsyncError.

    The SAM service processes asynchronously and calls back via webhook,
    which completes this activity externally with the result.

    Input: {
        "video_path": str,      # local path to effects-applied video
        "cache_dir": str,       # temp dir for downloads
    }
    Output (set by webhook): {
        "output_path": str,     # S3 URI of the mask
        "processing_seconds": float,
        "gpu_tier": str,
    }
    """
    info = activity.info()
    workflow_id = info.workflow_id
    task_token = info.task_token

    video_path = input_data["video_path"]

    # 1. Upload effects-applied video to S3
    input_s3_key = f"videos/{workflow_id}/rendered.mp4"
    upload_file(video_path, input_s3_key)

    # 2. POST to SAM /segment_to_s3 with webhook fields
    output_s3_key = f"masks/{workflow_id}/mask.mp4"
    task_token_b64 = base64.b64encode(task_token).decode("utf-8")

    request_body = {
        "input_s3_key": input_s3_key,
        "output_s3_key": output_s3_key,
        "auto_detect": True,
        "output_mode": "mask_video",
        "_webhook_url": settings.WEBHOOK_URL,
        "_task_token": task_token_b64,
        "_workflow_id": workflow_id,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(settings.SAM_ENDPOINT_URL, json=request_body)
        resp.raise_for_status()

    logger.info(
        "SAM segmentation requested: input=%s output=%s",
        input_s3_key, output_s3_key,
    )

    # 3. Tell Temporal this activity completes externally via webhook
    raise activity.CompleteAsyncError()


@activity.defn(name="vfx_download_s3")
def download_s3(input_data: dict) -> dict:
    """Download a file from S3 to a local path.

    Input: {"s3_key": str, "local_path": str}
    Output: {"local_path": str}
    """
    s3_key = input_data["s3_key"]
    local_path = input_data["local_path"]
    download_file(s3_key, local_path)
    return {"local_path": local_path}
