"""S3 upload/download helpers using boto3."""

import logging
import os

import boto3

from video_effects.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return _client


def upload_file(local_path: str, s3_key: str) -> str:
    """Upload a local file to S3. Returns the s3_key."""
    client = _get_client()
    logger.info("Uploading %s -> s3://%s/%s", local_path, settings.S3_BUCKET, s3_key)
    client.upload_file(local_path, settings.S3_BUCKET, s3_key)
    logger.info("Upload complete: %s (%d bytes)", s3_key, os.path.getsize(local_path))
    return s3_key


def download_file(s3_key: str, local_path: str) -> str:
    """Download a file from S3 to a local path. Returns the local_path."""
    client = _get_client()
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    logger.info("Downloading s3://%s/%s -> %s", settings.S3_BUCKET, s3_key, local_path)
    client.download_file(settings.S3_BUCKET, s3_key, local_path)
    logger.info("Download complete: %s (%d bytes)", local_path, os.path.getsize(local_path))
    return local_path
