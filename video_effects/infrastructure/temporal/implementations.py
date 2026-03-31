from __future__ import annotations

from temporalio import activity

from video_effects.core.base.context import ExecutionContext
from video_effects.core.interfaces.progress import ProgressReport


class TemporalContextProvider:
    def get_context(self) -> ExecutionContext:
        info = activity.info()
        return ExecutionContext(
            activity_id=info.activity_id,
            workflow_id=info.workflow_id,
            workflow_run_id=info.workflow_run_id,
            attempt=info.attempt,
        )


class TemporalProgressReporter:
    async def report(self, report: ProgressReport) -> None:
        activity.heartbeat(report.message)

    def report_sync(self, report: ProgressReport) -> None:
        activity.heartbeat(report.message)


def wrap_as_temporal_activity(func, name: str):
    return activity.defn(name=name)(func)
