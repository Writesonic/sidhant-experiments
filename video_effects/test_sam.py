"""End-to-end test: SAM segmentation + Remotion background render.

Uploads a video to S3, calls SAM synchronously to get a mask,
then renders via Remotion with an animated background (aurora, particles, etc.).

Usage:
    python -m video_effects.test_sam info_out7.mp4 --background aurora
    python -m video_effects.test_sam info_out7.mp4 --background particles -o output.mp4
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

from video_effects.config import settings
from video_effects.helpers.remotion import composite_with_mask, render_media
from video_effects.helpers.s3 import download_file, upload_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> dict:
    """Get width, height, fps, and total_frames via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    for stream in data["streams"]:
        if stream["codec_type"] == "video":
            fps_parts = stream["r_frame_rate"].split("/")
            fps = int(fps_parts[0]) // int(fps_parts[1]) if len(fps_parts) == 2 else 30
            return {
                "width": int(stream["width"]),
                "height": int(stream["height"]),
                "fps": fps,
                "total_frames": int(stream.get("nb_frames", 0)),
            }
    raise RuntimeError("No video stream found")


def main():
    parser = argparse.ArgumentParser(description="Test SAM + Remotion background pipeline")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--background", "--bg", default="aurora", help="Background type (aurora, particles, meshGradient, gridPattern)")
    parser.add_argument("--bg-config", type=json.loads, default={}, help="Background config as JSON (e.g. '{\"speed\": 0.5}')")
    parser.add_argument("--output", "-o", default="bg_output.mp4", help="Output video path (default: bg_output.mp4)")
    parser.add_argument("--skip-sam", action="store_true", help="Skip SAM, use --mask-path instead")
    parser.add_argument("--mask-path", help="Pre-existing mask video (use with --skip-sam)")
    args = parser.parse_args()

    if not args.skip_sam and not settings.SAM_ENDPOINT_URL:
        print("Error: VFX_SAM_ENDPOINT_URL not set in .env")
        sys.exit(1)

    video_path = os.path.abspath(args.video)
    info = get_video_info(video_path)
    logger.info("Video: %dx%d @ %dfps, %d frames", info["width"], info["height"], info["fps"], info["total_frames"])

    with tempfile.TemporaryDirectory(prefix="vfx_test_") as tmp_dir:
        mask_local = os.path.join(tmp_dir, "mask.mp4")

        # ── Step 1: SAM Segmentation ──
        if args.skip_sam:
            if not args.mask_path:
                print("Error: --skip-sam requires --mask-path")
                sys.exit(1)
            shutil.copy2(args.mask_path, mask_local)
            logger.info("Using existing mask: %s", args.mask_path)
        else:
            s3_input_key = "test/sam_input.mp4"
            s3_output_key = "test/sam_mask.mp4"

            logger.info("Uploading %s to S3...", video_path)
            upload_file(video_path, s3_input_key)

            request_body = {
                "input_s3_key": s3_input_key,
                "output_s3_key": s3_output_key,
                "auto_detect": True,
                "output_mode": "mask_video",
            }

            logger.info("Calling SAM endpoint (synchronous): %s", settings.SAM_ENDPOINT_URL)
            t0 = time.time()
            resp = requests.post(settings.SAM_ENDPOINT_URL, json=request_body, timeout=600)
            resp.raise_for_status()
            elapsed = time.time() - t0
            result = resp.json()
            logger.info("SAM completed in %.1fs: %s", elapsed, result)

            # Download the mask from S3
            output_key = result.get("output_path", s3_output_key)
            if output_key.startswith("s3://"):
                output_key = output_key.split("/", 3)[-1]
            download_file(output_key, mask_local)
            logger.info("Mask downloaded: %s", mask_local)

        # ── Step 2: Render background-only via Remotion ──
        bg_only_path = os.path.join(tmp_dir, "bg_only.mp4")

        plan = {
            "components": [],
            "colorPalette": [],
            "includeBaseVideo": False,
            "backgroundMode": {
                "originalSrc": "",
                "maskSrc": "",
                "backgroundType": args.background,
                "backgroundConfig": args.bg_config,
            },
            "durationInFrames": info["total_frames"],
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"],
        }

        logger.info("Rendering %s background via Remotion (background only)...", args.background)
        t0 = time.time()

        render_media(
            composition_id="MotionOverlay",
            props=plan,
            output_path=bg_only_path,
            width=info["width"],
            height=info["height"],
            fps=info["fps"],
            duration_in_frames=info["total_frames"],
            codec="h264",
        )

        bg_elapsed = time.time() - t0
        logger.info("Background render completed in %.1fs", bg_elapsed)

        # ── Step 3: FFmpeg mask composite (person onto background) ──
        output_path = os.path.abspath(args.output)
        logger.info("Compositing person onto background via FFmpeg...")
        t0 = time.time()

        composite_with_mask(
            background_video=bg_only_path,
            original_video=video_path,
            mask_video=mask_local,
            output_path=output_path,
        )

        comp_elapsed = time.time() - t0
        logger.info("FFmpeg composite completed in %.1fs", comp_elapsed)
        logger.info("Total render time: %.1fs (bg: %.1fs + composite: %.1fs)", bg_elapsed + comp_elapsed, bg_elapsed, comp_elapsed)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Done! Output: %s (%.1f MB)", output_path, file_size)


if __name__ == "__main__":
    main()
