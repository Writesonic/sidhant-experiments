from __future__ import annotations

import logging

from video_effects.config import settings


def get_context_provider():
    if settings.RUNTIME_MODE == "TEMPORAL":
        from video_effects.infrastructure.temporal.implementations import TemporalContextProvider
        return TemporalContextProvider()
    from video_effects.infrastructure.testing.implementations import TestContextProvider
    return TestContextProvider()


def get_progress_reporter(context=None):
    if settings.RUNTIME_MODE == "TEMPORAL":
        from video_effects.infrastructure.temporal.implementations import TemporalProgressReporter
        return TemporalProgressReporter()
    from video_effects.core.base.capability import MockProgressReporter
    return MockProgressReporter()


async def run_capability(capability_class, request, logger=None):
    context_provider = get_context_provider()
    context = context_provider.get_context()
    reporter = get_progress_reporter(context)
    logger = logger or logging.getLogger(capability_class.__name__)
    capability = capability_class(reporter, context, logger)
    return await capability.execute(request)


def run_capability_sync(capability_class, request, logger=None):
    """Run a capability synchronously. For use in sync Temporal activities."""
    import asyncio
    context_provider = get_context_provider()
    context = context_provider.get_context()
    reporter = get_progress_reporter(context)
    logger = logger or logging.getLogger(capability_class.__name__)
    capability = capability_class(reporter, context, logger)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(capability.execute(request))
    finally:
        loop.close()


def get_activity_wrapper():
    if settings.RUNTIME_MODE == "TEMPORAL":
        from video_effects.infrastructure.temporal.implementations import wrap_as_temporal_activity
        return wrap_as_temporal_activity
    from video_effects.infrastructure.testing.implementations import wrap_as_test_activity
    return wrap_as_test_activity
