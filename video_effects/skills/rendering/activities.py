import os
import uuid

import boto3
import modal
from temporalio import activity

from video_effects.skills.registry import register_activity
from video_effects.skills.rendering.capabilities._pipeline import (
    EFFECT_PROCESSORS, _build_merged_intervals, _process_single_pass, _probe_decoded_size, _is_hdr, group_by_phase,
)
from video_effects.schemas.effects import EffectCue, VideoInfo
from video_effects.skills.rendering.schemas import (
    ApplyEffectsRequest, PrepareRenderRequest, SetupProcessorsRequest, RenderVideoRequest,
)


@register_activity(name="vfx_apply_effects", description="Apply effects in phase order (legacy)")
def vfx_apply_effects(input_data: dict) -> dict:
    request = ApplyEffectsRequest(**input_data)
    effects = [EffectCue(**e) for e in request.effects]
    video_info = VideoInfo(**request.video_info)
    os.makedirs(request.output_dir, exist_ok=True)
    if not effects:
        return {"processed_video": request.video_path, "phases_executed": 0}
    phase_groups = group_by_phase(effects)
    processors = []
    for phase_num, phase_effects in sorted(phase_groups.items()):
        effect_type = phase_effects[0].effect_type
        processor_cls = EFFECT_PROCESSORS.get(effect_type)
        if processor_cls is None:
            continue
        processor = processor_cls()
        processor.setup(video_info, phase_effects)
        processors.append(processor)
    if not processors:
        return {"processed_video": request.video_path, "phases_executed": 0}
    all_intervals = _build_merged_intervals(processors, video_info)
    output_path = os.path.join(request.output_dir, "processed.mp4")
    _process_single_pass(
        request.video_path, output_path, processors, video_info, all_intervals,
        heartbeat_fn=lambda msg: activity.heartbeat(msg),
    )
    return {"processed_video": output_path, "phases_executed": len(processors)}


@register_activity(name="vfx_prepare_render", description="Probe dimensions and build render plan")
def vfx_prepare_render(input_data: dict) -> dict:
    request = PrepareRenderRequest(**input_data)
    effects = [EffectCue(**e) for e in request.effects]
    video_info = VideoInfo(**request.video_info)
    if not effects:
        return {"decoded_width": 0, "decoded_height": 0, "is_hdr": False,
                "phase_summary": [], "active_intervals": [],
                "active_frame_count": 0, "total_phases": 0, "has_effects": False}
    decoded_width, decoded_height = _probe_decoded_size(request.video_path)
    hdr = _is_hdr(video_info)
    phase_groups = group_by_phase(effects)
    phase_summary = []
    processors = []
    for phase_num, phase_effects in sorted(phase_groups.items()):
        effect_type = phase_effects[0].effect_type
        processor_cls = EFFECT_PROCESSORS.get(effect_type)
        if processor_cls is None:
            continue
        processor = processor_cls()
        processor.set_cues(phase_effects)
        processors.append(processor)
        phase_summary.append({"phase": phase_num, "effect_type": effect_type.value, "count": len(phase_effects)})
    active_intervals = _build_merged_intervals(processors, video_info)
    active_frame_count = sum(e - s for s, e in active_intervals)
    return {
        "decoded_width": decoded_width, "decoded_height": decoded_height, "is_hdr": hdr,
        "phase_summary": phase_summary, "active_intervals": active_intervals,
        "active_frame_count": active_frame_count, "total_phases": len(processors),
        "has_effects": len(processors) > 0,
    }


@register_activity(name="vfx_setup_processors", description="Initialize effect processors")
def vfx_setup_processors(input_data: dict) -> dict:
    request = SetupProcessorsRequest(**input_data)
    effects = [EffectCue(**e) for e in request.effects]
    video_info = VideoInfo(**request.video_info)
    os.makedirs(request.cache_dir, exist_ok=True)
    phase_groups = group_by_phase(effects)
    setup_summary = []
    for phase_num, phase_effects in sorted(phase_groups.items()):
        effect_type = phase_effects[0].effect_type
        processor_cls = EFFECT_PROCESSORS.get(effect_type)
        if processor_cls is None:
            continue
        processor = processor_cls()
        processor.setup(video_info, phase_effects, cache_dir=request.cache_dir, video_path=request.video_path)
        setup_summary.append({"phase": phase_num, "effect_type": effect_type.value})
        activity.heartbeat(f"setup phase {phase_num}")
    return {"setup_summary": setup_summary, "processors_ready": True}


_S3_BUCKET = "ai-video-actions-755620792222-us-east-1"


def _get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("VFX_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("VFX_AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("VFX_AWS_REGION", "us-east-1"),
    )


@register_activity(name="vfx_render_video", description="Run single-pass frame pipeline via Modal GPU")
def vfx_render_video(input_data: dict) -> dict:
    request = RenderVideoRequest(**input_data)
    if not request.effects:
        return {"processed_video": request.video_path, "phases_executed": 0}

    os.makedirs(request.output_dir, exist_ok=True)
    output_path = os.path.join(request.output_dir, "processed.mp4")
    job_id = uuid.uuid4().hex[:12]
    input_s3_key = f"vfx-render/inputs/{job_id}.mp4"
    output_s3_key = f"vfx-render/outputs/{job_id}.mp4"

    s3 = _get_s3_client()

    activity.heartbeat("uploading to S3")
    s3.upload_file(request.video_path, _S3_BUCKET, input_s3_key)

    activity.heartbeat("rendering on Modal GPU")
    processor = modal.Cls.from_name("vfx-render-prod", "VfxRenderWorker")()
    processor.process.remote(
        output_filename=f"{job_id}.mp4",
        effects=request.effects,
        video_info=request.video_info,
        render_plan=request.render_plan,
        input_s3_key=input_s3_key,
        output_s3_key=output_s3_key,
        mux_audio=False,
    )

    activity.heartbeat("downloading result")
    s3.download_file(_S3_BUCKET, output_s3_key, output_path)

    # Cleanup S3 temp files
    s3.delete_object(Bucket=_S3_BUCKET, Key=input_s3_key)
    s3.delete_object(Bucket=_S3_BUCKET, Key=output_s3_key)

    return {"processed_video": output_path, "phases_executed": len(request.effects)}
