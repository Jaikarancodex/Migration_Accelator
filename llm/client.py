"""Single interface for all LLM calls, so the model is swappable and mockable.

Nothing outside this module should import `anthropic` directly — every call
site depends on the `LLMClient` protocol instead, which is what lets tests
run against `MockLLMClient` with no API key and no network access.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"


class LLMClient(ABC):
    """Minimal interface every LLM backend implements."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's text response for a single-turn system/user call."""
        raise NotImplementedError


class AnthropicLLMClient(LLMClient):
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> None:
        import anthropic  # imported lazily so MockLLMClient users never need the package

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key provided. Pass api_key= or set ANTHROPIC_API_KEY."
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        logger.info("llm_request", model=self.model, system_len=len(system), user_len=len(user))
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        logger.info("llm_response", model=self.model, response_len=len(text))
        return text


class MockLLMClient(LLMClient):
    """Deterministic stand-in for tests and offline development.

    `response` may be a fixed string or a callable(system, user) -> str for
    tests that need to react to the prompt content.
    """

    def __init__(self, response: str | Callable[[str, str], str]) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self._response):
            return self._response(system, user)
        return self._response
