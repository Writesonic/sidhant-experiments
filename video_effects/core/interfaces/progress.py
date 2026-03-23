from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class EventType(str, Enum):
    HEARTBEAT = "HEARTBEAT"
    STEP_STARTED = "STEP_STARTED"
    STEP_COMPLETED = "STEP_COMPLETED"
    PROGRESS_UPDATE = "PROGRESS_UPDATE"


@dataclass
class ProgressReport:
    message: str
    event_type: EventType = EventType.HEARTBEAT


@runtime_checkable
class ProgressReporter(Protocol):
    async def report(self, report: ProgressReport) -> None: ...
    def report_sync(self, report: ProgressReport) -> None: ...
