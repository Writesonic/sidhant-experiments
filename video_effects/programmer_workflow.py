"""Temporal child workflow: ProgrammerWorkflow.

Free-hand creative agent that analyzes the transcript and invents whatever
motion graphics would most elevate the video. No fixed types, no category
constraints. Uses a 2-step brainstorm→critique pipeline, then generates +
validates TSX code for each approved spec.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from video_effects.config import settings


@workflow.defn(name="ProgrammerWorkflow")
class ProgrammerWorkflow:

    @workflow.run
    async def run(self, input: dict) -> dict:
        """Generate creative visual components from transcript analysis.

        Uses a 2-step brainstorm→critique pipeline instead of fixed category
        planners, then reuses the same generate+validate+registry pattern.

        Input: {
            "spatial_context": dict,
            "transcript": str,
            "segments": list[dict],
            "style_config": dict | None,
            "video_fps": int,
            "video_info": dict,
            "workflow_prefix": str,
        }
        Output: {
            "generated_components": list[dict],
            "fallback_components": list[dict],
            "registry_path": str,
        }
        """
        spatial_context = input["spatial_context"]
        transcript = input.get("transcript", "")
        segments = input.get("segments", [])
        style_config = input.get("style_config")
        fps = input.get("video_fps", 30)
        video_info = input.get("video_info", {})
        prefix = input.get("workflow_prefix", "pg")

        activity_timeout = timedelta(minutes=5)
        max_retries = getattr(settings, "PROGRAMMER_MAX_RETRIES", 3)

        # Dynamic max specs based on video duration
        duration = video_info.get("duration", 60)
        max_specs = min(max(3, int(duration) // 15), 15)

        # ── A0: Cleanup generated/ directory ──
        await workflow.execute_activity(
            "vfx_cleanup_generated",
            {},
            start_to_close_timeout=timedelta(seconds=30),
        )

        # ── A1a: Creative brainstorm (free-form proposals) ──
        brainstorm_input = {
            "spatial_context": spatial_context,
            "transcript": transcript,
            "segments": segments,
            "style_config": style_config,
            "video_info": video_info,
            "effects": [],  # could pass existing effects if available
        }

        brainstorm_result = await workflow.execute_activity(
            "vfx_programmer_brainstorm",
            brainstorm_input,
            start_to_close_timeout=activity_timeout,
        )

        proposals = brainstorm_result.get("components", [])
        if not proposals:
            workflow.logger.info("Brainstorm produced no proposals, skipping")
            return {"generated_components": [], "fallback_components": [], "registry_path": ""}

        workflow.logger.info(
            "Brainstorm produced %d proposals, critiquing (max %d)",
            len(proposals), max_specs,
        )

        # ── A1b: Self-critique + filter ──
        critique_result = await workflow.execute_activity(
            "vfx_programmer_critique",
            {
                "proposals": proposals,
                "spatial_context": spatial_context,
                "transcript": transcript,
                "video_info": video_info,
                "max_specs": max_specs,
            },
            start_to_close_timeout=activity_timeout,
        )

        specs = critique_result.get("components", [])
        if not specs:
            workflow.logger.info("Critique filtered all proposals, skipping")
            return {"generated_components": [], "fallback_components": [], "registry_path": ""}

        workflow.logger.info(
            "Critique kept %d spec(s), generating code", len(specs),
        )

        # ── A2+A3: Generate + validate each component (parallel, semaphore=3) ──
        successful = []
        fallbacks = []

        sem = asyncio.Semaphore(3)

        async def _process_spec(i: int, spec: dict) -> tuple[dict, dict | None]:
            spec["id"] = f"{prefix}_{spec.get('id', f'component_{i}')}"
            try:
                async with sem:
                    generated = await self._generate_with_retries(
                        spec=spec,
                        style_config=style_config,
                        video_info=video_info,
                        max_retries=max_retries,
                        activity_timeout=activity_timeout,
                    )
                return (spec, generated)
            except Exception as exc:
                workflow.logger.warning(
                    "Spec %s generation raised: %s", spec["id"], exc,
                )
                return (spec, None)

        results = await asyncio.gather(
            *[_process_spec(i, spec) for i, spec in enumerate(specs)]
        )

        for spec, generated in results:
            if generated is not None:
                successful.append(generated)
            else:
                fallback = self._build_fallback(spec, fps)
                fallbacks.append(fallback)
                workflow.logger.info(
                    "Fell back to animated_title for %s", spec["id"],
                )

        # ── A4: Build registry (reused) ──
        registry_result = {"components": [], "registry_path": ""}
        if successful:
            registry_result = await workflow.execute_activity(
                "vfx_build_generated_registry",
                {
                    "generated_components": successful,
                    "video_fps": fps,
                    "style_config": style_config,
                },
                start_to_close_timeout=timedelta(seconds=30),
            )

        workflow.logger.info(
            "Programmer generation complete: %d generated, %d fallback",
            len(successful), len(fallbacks),
        )

        return {
            "generated_components": registry_result.get("components", []),
            "fallback_components": fallbacks,
            "registry_path": registry_result.get("registry_path", ""),
        }

    async def _generate_with_retries(
        self,
        spec: dict,
        style_config: dict | None,
        video_info: dict,
        max_retries: int,
        activity_timeout: timedelta,
    ) -> dict | None:
        """Try to generate + validate a component, retrying on failure."""
        previous_errors: list[str] = []
        previous_code = ""

        for attempt in range(1, max_retries + 1):
            workflow.logger.info(
                "Generating %s (attempt %d/%d)", spec["id"], attempt, max_retries
            )

            # A2: Generate code (programmer-specific activity)
            gen_result = await workflow.execute_activity(
                "vfx_programmer_generate_code",
                {
                    "spec": spec,
                    "style_config": style_config,
                    "video_info": video_info,
                    "attempt": attempt,
                    "previous_errors": previous_errors,
                    "previous_code": previous_code,
                },
                start_to_close_timeout=activity_timeout,
            )

            tsx_code = gen_result["tsx_code"]
            export_name = gen_result["export_name"]
            component_id = gen_result["component_id"]

            # A3: Validate (reused from infographic workflow)
            val_result = await workflow.execute_activity(
                "vfx_validate_infographic",
                {
                    "component_id": component_id,
                    "tsx_code": tsx_code,
                    "export_name": export_name,
                    "props": gen_result.get("props", {}),
                },
                start_to_close_timeout=activity_timeout,
            )

            if val_result["valid"]:
                workflow.logger.info("Validated %s on attempt %d", component_id, attempt)
                return {
                    "component_id": component_id,
                    "export_name": export_name,
                    "spec": spec,
                    "props": gen_result.get("props", {}),
                }

            previous_errors = val_result.get("errors", [])
            previous_code = tsx_code
            workflow.logger.warning(
                "Validation failed for %s (attempt %d): %s",
                component_id, attempt, previous_errors[:3],
            )

        workflow.logger.warning(
            "All %d attempts failed for %s, falling back",
            max_retries, spec["id"],
        )
        return None

    @staticmethod
    def _build_fallback(spec: dict, fps: int) -> dict:
        """Map a failed spec to an animated_title fallback."""
        start_frame = round(spec.get("start_time", 0) * fps)
        end_time = spec.get("end_time", spec.get("start_time", 0) + 3)
        end_frame = round(end_time * fps)
        duration_frames = max(1, end_frame - start_frame)

        return {
            "template": "animated_title",
            "startFrame": start_frame,
            "durationInFrames": duration_frames,
            "props": {
                "text": spec.get("title", "Component"),
                "style": "fade",
            },
            "bounds": spec.get("bounds", {"x": 0.1, "y": 0.1, "w": 0.35, "h": 0.3}),
            "zIndex": 5,
            "anchor": spec.get("anchor", "static"),
        }
