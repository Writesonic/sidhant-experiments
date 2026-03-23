from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from video_effects.core.base.context import ExecutionContext
from video_effects.core.interfaces.progress import EventType, ProgressReport, ProgressReporter

TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")


class MockProgressReporter:
    async def report(self, report: ProgressReport) -> None:
        pass

    def report_sync(self, report: ProgressReport) -> None:
        pass


class BaseCapability(ABC, Generic[TRequest, TResponse]):
    __slots__ = ("_reporter", "_context", "_logger")

    def __init__(
        self,
        progress_reporter: ProgressReporter,
        context: ExecutionContext,
        logger: logging.Logger | None = None,
    ) -> None:
        self._reporter = progress_reporter
        self._context = context
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    @property
    def context(self) -> ExecutionContext:
        return self._context

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    async def heartbeat(self, message: str) -> None:
        await self._reporter.report(
            ProgressReport(message=message, event_type=EventType.HEARTBEAT)
        )

    def heartbeat_sync(self, message: str) -> None:
        self._reporter.report_sync(
            ProgressReport(message=message, event_type=EventType.HEARTBEAT)
        )

    @abstractmethod
    async def execute(self, request: TRequest) -> TResponse: ...
