"""Temporal workflow: VideoEffectsWorkflow.

Pipeline (7 groups):
  G1: Extract video info + audio
  G2: Transcribe audio
  G3: LLM: parse effect cues from transcript
  G4: Validate timeline + resolve conflicts
  G5: CLI approval (handled externally via signal)
  G6: Execute effects in dependency order
  G7: Final composition + audio mux
"""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from video_effects.schemas.workflow import VideoEffectsInput, VideoEffectsOutput


@workflow.defn(name="VideoEffectsWorkflow")
class VideoEffectsWorkflow:

    def __init__(self):
        self._approved = False
        self._timeline_data: dict | None = None

    @workflow.signal
    async def approve_timeline(self, approved: bool) -> None:
        """Signal from CLI to approve or reject the effect timeline."""
        self._approved = approved

    @workflow.query
    def get_timeline(self) -> dict | None:
        """Query to get the current timeline for CLI display."""
        return self._timeline_data

    @workflow.run
    async def run(self, input: VideoEffectsInput) -> VideoEffectsOutput:
        video_path = input.input_video
        output_path = input.output_video
        temp_dir = f"/tmp/video_effects/{workflow.info().workflow_id}"

        activity_timeout = timedelta(minutes=10)
        long_timeout = timedelta(minutes=30)

        # ── G1: Extract video info + audio (parallel) ──
        video_info_task = workflow.execute_activity(
            "vfx_get_video_info",
            video_path,
            start_to_close_timeout=activity_timeout,
        )
        extract_audio_task = workflow.execute_activity(
            "vfx_extract_audio",
            {"video_path": video_path, "output_dir": temp_dir},
            start_to_close_timeout=activity_timeout,
        )

        video_info, audio_result = await asyncio.gather(
            video_info_task, extract_audio_task
        )
        audio_path = audio_result["audio_path"]

        # ── G2: Transcribe audio ──
        transcript_result = await workflow.execute_activity(
            "vfx_transcribe_audio",
            {"audio_path": audio_path},
            start_to_close_timeout=activity_timeout,
        )

        # ── G3: LLM parse effect cues ──
        parse_result = await workflow.execute_activity(
            "vfx_parse_effect_cues",
            {
                "transcript": transcript_result["transcript"],
                "segments": transcript_result["segments"],
                "duration": video_info["duration"],
            },
            start_to_close_timeout=activity_timeout,
        )

        # ── G4: Validate timeline ──
        timeline = await workflow.execute_activity(
            "vfx_validate_timeline",
            {
                "effects": parse_result["effects"],
                "duration": video_info["duration"],
            },
            start_to_close_timeout=activity_timeout,
        )
        self._timeline_data = timeline

        # ── G5: CLI approval ──
        if not input.auto_approve:
            # Wait for approval signal (timeout after 10 minutes)
            try:
                await workflow.wait_condition(
                    lambda: self._approved, timeout=timedelta(minutes=10)
                )
            except asyncio.TimeoutError:
                return VideoEffectsOutput(
                    output_video="",
                    error="Timeline approval timed out after 10 minutes",
                )

            if not self._approved:
                return VideoEffectsOutput(
                    output_video="",
                    error="Timeline was rejected by user",
                )

        effects = timeline.get("effects", [])
        if not effects:
            return VideoEffectsOutput(
                output_video=video_path,
                effects_applied=0,
                transcript_length=len(transcript_result["transcript"]),
            )

        # ── G6: Execute effects in phase order ──
        apply_result = await workflow.execute_activity(
            "vfx_apply_effects",
            {
                "video_path": video_path,
                "output_dir": temp_dir,
                "effects": effects,
                "video_info": video_info,
            },
            start_to_close_timeout=long_timeout,
            heartbeat_timeout=timedelta(minutes=2),
        )

        # ── G7: Final composition + audio mux ──
        compose_result = await workflow.execute_activity(
            "vfx_compose_final",
            {
                "processed_video": apply_result["processed_video"],
                "audio_path": audio_path,
                "output_path": output_path,
                "has_audio": video_info.get("has_audio", True),
            },
            start_to_close_timeout=activity_timeout,
        )

        return VideoEffectsOutput(
            output_video=compose_result["output_video"],
            effects_applied=len(effects),
            transcript_length=len(transcript_result["transcript"]),
            phases_executed=apply_result["phases_executed"],
        )
