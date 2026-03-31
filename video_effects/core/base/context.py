from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryInfo:
    attempt: int
    is_retry: bool

    @classmethod
    def from_attempt(cls, attempt: int) -> RetryInfo:
        return cls(attempt=attempt, is_retry=attempt > 1)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    activity_id: str
    workflow_id: str
    workflow_run_id: str
    attempt: int = 1

    @property
    def is_retry(self) -> bool:
        return self.attempt > 1

    @property
    def retry_info(self) -> RetryInfo:
        return RetryInfo.from_attempt(self.attempt)

    @classmethod
    def create_test_context(
        cls,
        activity_id: str | None = None,
        workflow_id: str | None = None,
        workflow_run_id: str | None = None,
        attempt: int = 1,
    ) -> ExecutionContext:
        return cls(
            activity_id=activity_id or f"test-activity-{uuid.uuid4().hex[:8]}",
            workflow_id=workflow_id or f"test-workflow-{uuid.uuid4().hex[:8]}",
            workflow_run_id=workflow_run_id or f"test-run-{uuid.uuid4().hex[:8]}",
            attempt=attempt,
        )
