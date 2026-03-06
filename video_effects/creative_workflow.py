"""Temporal child workflow: CreativeDesignerWorkflow.

Analyzes transcript + video metadata to pick and customize a visual style.
"""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from video_effects.schemas.styles import StyleConfig


@workflow.defn(name="CreativeDesignerWorkflow")
class CreativeDesignerWorkflow:
    @workflow.run
    async def run(self, input: dict) -> dict:
        result = await workflow.execute_activity(
            "vfx_design_style",
            input,
            start_to_close_timeout=timedelta(minutes=2),
        )
        return result
