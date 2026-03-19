"""Anthropic API wrapper for structured LLM calls."""

import json
import logging
import random
import time
from pathlib import Path

import anthropic

from video_effects.config import settings

logger = logging.getLogger(__name__)

# Retryable: overloaded (529), rate limit (429), server errors (503, 504), timeouts
RETRYABLE_STATUS_CODES = {429, 503, 504, 529}
MAX_LLM_RETRIES = 5
INITIAL_RETRY_DELAY = 2.0
MAX_RETRY_DELAY = 120.0


def _is_retryable(error: BaseException) -> bool:
    """True if the error is transient and worth retrying."""
    if isinstance(error, anthropic.APITimeoutError):
        return True
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code in RETRYABLE_STATUS_CODES
    return False


def _sleep_with_jitter(attempt: int) -> None:
    """Exponential backoff with jitter to avoid thundering herd."""
    delay = min(INITIAL_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
    jitter = delay * 0.2 * (2 * random.random() - 1)
    sleep_time = max(0.1, delay + jitter)
    logger.warning("LLM transient error, retrying in %.1fs (attempt %d)", sleep_time, attempt + 1)
    time.sleep(sleep_time)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def get_client() -> anthropic.Anthropic:
    """Get an Anthropic client instance."""
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = _PROMPT_DIR / name
    return path.read_text()


def call_structured(
    system_prompt: str,
    user_message: str,
    response_model: type,
    model: str | None = None,
    max_tokens: int = 4096,
) -> dict:
    """Call Claude with structured output via tool use.

    Retries on transient errors (529 overloaded, 429 rate limit, 503/504, timeouts).

    Args:
        system_prompt: System prompt text.
        user_message: User message content.
        response_model: Pydantic model class for the response schema.
        model: Model ID override. Defaults to settings.LLM_MODEL.
        max_tokens: Max tokens in response.

    Returns:
        Parsed dict matching the response_model schema.
    """
    client = get_client()
    model = model or settings.LLM_MODEL

    # Build tool definition from Pydantic schema
    schema = response_model.model_json_schema()
    tool_name = "structured_output"
    tool = {
        "name": tool_name,
        "description": f"Return structured {response_model.__name__} response",
        "input_schema": schema,
    }

    last_error: BaseException | None = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
            )
            for block in response.content:
                if block.type == "tool_use":
                    return block.input
            raise ValueError("LLM did not return structured output via tool use")
        except ValueError:
            raise
        except Exception as e:
            last_error = e
            if _is_retryable(e) and attempt < MAX_LLM_RETRIES - 1:
                status = getattr(e, "status_code", None)
                logger.warning(
                    "LLM call failed (status=%s): %s",
                    status,
                    getattr(e, "message", str(e)),
                )
                _sleep_with_jitter(attempt)
            else:
                raise

    raise last_error or ValueError("LLM did not return structured output via tool use")


def call_text(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    max_tokens: int = 8192,
) -> str:
    """Call Claude and return raw text response (for code generation).

    Retries on transient errors (529 overloaded, 429 rate limit, 503/504, timeouts).
    """
    client = get_client()
    model = model or settings.LLM_MODEL
    last_error: BaseException | None = None

    for attempt in range(MAX_LLM_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            for block in response.content:
                if block.type == "text":
                    return block.text
            raise ValueError("LLM did not return text output")
        except Exception as e:
            last_error = e
            if _is_retryable(e) and attempt < MAX_LLM_RETRIES - 1:
                status = getattr(e, "status_code", None)
                logger.warning(
                    "LLM call failed (status=%s): %s",
                    status,
                    getattr(e, "message", str(e)),
                )
                _sleep_with_jitter(attempt)
            else:
                raise

    raise last_error or ValueError("LLM did not return text output")
