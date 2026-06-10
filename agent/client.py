"""Anthropic client wrapper.

Centralizes API-key handling and the model id so:

* The deterministic ``solve`` CLI works with no API key (we never instantiate
  the client at module import).
* Test code can substitute a fake client in one place.

The wrapper exposes a single ``LLMClient`` protocol: ``call_tool(...)`` returns
the raw arguments dict produced by the model's tool call, or raises
``ParseFailedError``. The protocol is intentionally narrow — every higher-level
module (parse, explain) calls through this surface and is therefore easy to
mock in tests.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from core.exceptions import ParseFailedError

CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048


class LLMClient(Protocol):
    """Narrow protocol used by every agent module that calls the model."""

    def call_tool(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tool_name: str,
        tool_input_schema: dict[str, Any],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> dict[str, Any]: ...

    def call_text(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str: ...


class AnthropicClient:
    """Production client backed by the official Anthropic SDK.

    Constructed lazily — instantiating this requires ``ANTHROPIC_API_KEY``
    in the environment. The deterministic CLI never instantiates it.
    """

    def __init__(self, api_key: str | None = None, model: str = CLAUDE_MODEL) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. The Truffle chat mode requires an "
                "Anthropic API key; the deterministic `solve` command does not."
            )
        # Local import keeps the deterministic CLI free of the dependency at import time.
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def call_tool(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tool_name: str,
        tool_input_schema: dict[str, Any],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Force-tool call. Returns the model's tool input as a dict.

        We use ``tool_choice = {"type": "tool", "name": ...}`` to *require* the
        model to invoke our tool — this is how Anthropic guarantees the response
        is structured JSON conforming to ``tool_input_schema``.
        """
        resp = self._client.messages.create(
            model=self._model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=[
                {
                    "name": tool_name,
                    "description": "Return the structured parse result.",
                    "input_schema": tool_input_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return dict(block.input)
        raise ParseFailedError(
            f"Model did not invoke the {tool_name!r} tool as required. "
            f"Stop reason: {resp.stop_reason!r}."
        )

    def call_text(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        """Unstructured text call. Used by the explanation layer."""
        resp = self._client.messages.create(
            model=self._model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        chunks: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
        return "".join(chunks).strip()
