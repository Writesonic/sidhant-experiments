from __future__ import annotations

from video_effects.core.base.context import ExecutionContext
from video_effects.core.interfaces.progress import ProgressReport


class TestContextProvider:
    def __init__(self, context: ExecutionContext | None = None) -> None:
        self._context = context or ExecutionContext.create_test_context()

    def get_context(self) -> ExecutionContext:
        return self._context


class RecordingProgressReporter:
    def __init__(self) -> None:
        self.records: list[ProgressReport] = []

    async def report(self, report: ProgressReport) -> None:
        self.records.append(report)

    def report_sync(self, report: ProgressReport) -> None:
        self.records.append(report)


def wrap_as_test_activity(func, name: str):
    return func
