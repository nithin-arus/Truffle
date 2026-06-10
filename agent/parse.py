"""Natural language → ParseResult via tool-use structured output.

This module owns the contract between the LLM and the math layer. The
process per user turn is:

1. Build the structured-output schema from ``ParseEnvelope``.
2. Call the model with ``tool_choice`` forcing it to emit one tool call
   whose ``input`` matches that schema.
3. Validate the input as ``ParseEnvelope`` with Pydantic.
4. On Pydantic failure, send the validation error back to the model and
   ask it to repair (up to ``MAX_REPAIR_ATTEMPTS`` times). Two failures
   in a row escalate to :class:`ParseFailedError`; the chat loop catches
   that and asks the user to rephrase.

Nothing in this module touches CVXPY or numerics. The boundary is strict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent.client import LLMClient
from agent.schema import ParseEnvelope, ParseResult
from core.exceptions import ParseFailedError
from core.ir import PortfolioSpec

_PROMPT_PATH = Path(__file__).parent / "prompts" / "parse_system.md"
TOOL_NAME = "truffle_parse"
MAX_REPAIR_ATTEMPTS = 2


def _load_system_prompt() -> str:
    """Read the versioned system prompt from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _envelope_schema() -> dict[str, Any]:
    """Schema accepted by the Anthropic tool-use binder.

    We delegate to Pydantic's JSON schema generator and then strip the
    ``$defs``/``$ref`` indirection that Anthropic accepts but that occasionally
    confuses smaller models. (We keep the refs here; the SDK passes them through.)
    """
    return ParseEnvelope.model_json_schema()


def _format_user_message(
    user_text: str,
    current_spec: PortfolioSpec | None,
    universe_metadata: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    parts.append("USER_MESSAGE:")
    parts.append(user_text.strip())
    parts.append("")
    parts.append("CURRENT_SPEC (JSON, may be null):")
    if current_spec is None:
        parts.append("null")
    else:
        parts.append(json.dumps(current_spec.model_dump(), indent=2))
    parts.append("")
    parts.append("UNIVERSE_METADATA (JSON):")
    parts.append(json.dumps(universe_metadata or {}, indent=2))
    return "\n".join(parts)


def parse_user_message(
    user_text: str,
    *,
    client: LLMClient,
    current_spec: PortfolioSpec | None = None,
    universe_metadata: dict[str, Any] | None = None,
) -> ParseResult:
    """Run the parse with up to ``MAX_REPAIR_ATTEMPTS`` validation repair passes.

    Args:
        user_text: The raw user message for this turn.
        client: An ``LLMClient`` (production or fake).
        current_spec: The active spec from prior turns, if any. Drives the
            patch-vs-fresh decision the model is instructed to make.
        universe_metadata: Optional dict including at least ``tickers`` — the
            ticker pool the agent may reference.

    Returns:
        A validated ``ParseResult`` (one of FreshSpec / SpecPatch / Clarification).

    Raises:
        ParseFailedError: when the model cannot produce a valid result even
            after the repair attempts.
    """
    system = _load_system_prompt()
    schema = _envelope_schema()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _format_user_message(user_text, current_spec, universe_metadata)}
    ]

    last_error: str | None = None
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        raw = client.call_tool(
            system=system,
            messages=messages,
            tool_name=TOOL_NAME,
            tool_input_schema=schema,
        )
        try:
            envelope = ParseEnvelope.model_validate(raw)
            return envelope.result
        except ValidationError as e:
            last_error = str(e)
            if attempt >= MAX_REPAIR_ATTEMPTS:
                break
            # Send the model its own output + the validation error, ask for a fix.
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"toolu_repair_{attempt}",
                            "name": TOOL_NAME,
                            "input": raw,
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"toolu_repair_{attempt}",
                            "is_error": True,
                            "content": (
                                "Your previous tool call failed schema validation. "
                                "Re-emit the tool call with a valid envelope. "
                                f"Validation error:\n{e}"
                            ),
                        }
                    ],
                }
            )
    raise ParseFailedError(
        "Failed to obtain a valid ParseResult after "
        f"{MAX_REPAIR_ATTEMPTS + 1} attempts. Last validation error:\n{last_error}"
    )
