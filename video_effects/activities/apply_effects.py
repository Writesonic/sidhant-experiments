"""Activity: execute effects in phase order using the effect processor pipeline."""

import logging
import os

import cv2
import numpy as np
from temporalio import activity

from video_effects.effect_registry import group_by_phase
from video_effects.effects import ZoomEffect, BlurEffect, ColorEffect, SubtitleEffect
from video_effects.effects.base import EffectContext
from video_effects.schemas.effects import EffectCue, EffectType, VideoInfo

logger = logging.getLogger(__name__)

# Map effect types to processor classes
EFFECT_PROCESSORS = {
    EffectType.ZOOM: ZoomEffect,
    EffectType.BLUR: BlurEffect,
    EffectType.COLOR_CHANGE: ColorEffect,
    EffectType.SUBTITLE: SubtitleEffect,
}


@activity.defn(name="vfx_apply_effects")
def apply_effects(input_data: dict) -> dict:
    """Apply effects to video in phase order.

    Each phase reads the current video and writes the next version.
    Phases are sequential; within a phase, effects are applied per-frame.

    Input: {
        "video_path": str,
        "output_dir": str,
        "effects": list[dict],
        "video_info": dict,
    }
    Output: {"processed_video": str, "phases_executed": int}
    """
    video_path = input_data["video_path"]
    output_dir = input_data["output_dir"]
    effects = [EffectCue(**e) for e in input_data["effects"]]
    video_info = VideoInfo(**input_data["video_info"])

    os.makedirs(output_dir, exist_ok=True)

    if not effects:
        logger.info("No effects to apply, copying input video")
        return {"processed_video": video_path, "phases_executed": 0}

    # Group effects by phase
    phase_groups = group_by_phase(effects)
    logger.info(f"Executing {len(phase_groups)} phases with {len(effects)} total effects")

    current_video = video_path
    phases_executed = 0

    for phase_num, phase_effects in phase_groups.items():
        logger.info(
            f"Phase {phase_num}: {len(phase_effects)} effects "
            f"({phase_effects[0].effect_type.value})"
        )

        # Get the processor for this phase's effect type
        effect_type = phase_effects[0].effect_type
        processor_cls = EFFECT_PROCESSORS.get(effect_type)
        if processor_cls is None:
            logger.warning(f"No processor for effect type {effect_type}, skipping")
            continue

        processor = processor_cls()
        processor.setup(video_info, phase_effects)

        # Process video frame by frame
        current_video = _process_video_with_effect(
            current_video, output_dir, f"phase_{phase_num}",
            processor, video_info,
        )
        phases_executed += 1

    return {
        "processed_video": current_video,
        "phases_executed": phases_executed,
    }


def _process_video_with_effect(
    input_path: str,
    output_dir: str,
    phase_label: str,
    processor,
    video_info: VideoInfo,
) -> str:
    """Process all frames of a video through an effect processor.

    Uses FFV1 lossless codec for intermediates to prevent color drift.
    Returns output video path.
    """
    lossless_path = os.path.join(output_dir, f"{phase_label}.avi")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter(
        lossless_path,
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )

    frame_index = 0
    total_frames = video_info.total_frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_index / video_info.fps
            context = EffectContext(
                video_info=video_info,
                frame_index=frame_index,
                timestamp=timestamp,
                total_frames=total_frames,
            )

            processed = processor.apply_frame(frame, timestamp, context)
            out.write(processed)
            frame_index += 1

            if frame_index % 100 == 0:
                activity.heartbeat(f"frame {frame_index}/{total_frames}")
    finally:
        cap.release()
        out.release()

    logger.info(f"Processed {frame_index} frames -> {lossless_path}")
    return lossless_path
