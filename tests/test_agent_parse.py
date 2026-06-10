"""Tests for agent/parse.py using a fake LLMClient.

No live API calls happen in the default pytest run — every test injects a
``FakeClient`` whose ``call_tool`` returns canned dicts. We exercise:
clarification dispatch, fresh-spec dispatch, patch dispatch, and the
two-attempt repair loop (success on attempt 2; failure after exhausting
attempts).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.parse import parse_user_message
from agent.schema import Clarification, FreshSpec, SpecPatch
from core.exceptions import ParseFailedError
from core.ir import MinVariance, PortfolioSpec


class FakeClient:
    """Returns canned responses in order; records the messages it saw."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.recorded_messages: list[list[dict[str, Any]]] = []

    def call_tool(self, *, system, messages, tool_name, tool_input_schema, **_kwargs):
        # Snapshot the messages so repair-loop tests can assert the assistant
        # got the validation error fed back in.
        self.recorded_messages.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError("FakeClient ran out of canned responses.")
        return self._responses.pop(0)

    def call_text(self, *, system, messages, **_kwargs):
        raise NotImplementedError


def test_parse_returns_clarification_on_canned_response() -> None:
    client = FakeClient(
        [
            {
                "result": {
                    "kind": "clarification",
                    "question": "What's the max percentage per position?",
                    "reason": "vague_quantity",
                }
            }
        ]
    )
    result = parse_user_message(
        "Minimize risk but not too concentrated.",
        client=client,
        current_spec=None,
        universe_metadata={"tickers": ["AAA", "BBB"]},
    )
    assert isinstance(result, Clarification)
    assert "max percentage" in result.question


def test_parse_returns_fresh_spec() -> None:
    client = FakeClient(
        [
            {
                "result": {
                    "kind": "fresh_spec",
                    "spec": {
                        "universe": ["AAA", "BBB", "CCC"],
                        "objective": {"kind": "min_variance"},
                        "constraints": [{"kind": "budget"}, {"kind": "long_only"}],
                    },
                }
            }
        ]
    )
    result = parse_user_message(
        "Minimize variance long only fully invested.",
        client=client,
        current_spec=None,
        universe_metadata={"tickers": ["AAA", "BBB", "CCC"]},
    )
    assert isinstance(result, FreshSpec)
    assert result.spec.universe == ["AAA", "BBB", "CCC"]


def test_parse_returns_spec_patch() -> None:
    client = FakeClient(
        [
            {
                "result": {
                    "kind": "spec_patch",
                    "add_constraints": [
                        {
                            "kind": "box",
                            "id": "cap_all",
                            "lower": 0.0,
                            "upper": 0.4,
                        }
                    ],
                }
            }
        ]
    )
    current = PortfolioSpec(
        universe=["AAA", "BBB"],
        objective=MinVariance(),
    )
    result = parse_user_message(
        "Cap each position at 40%.",
        client=client,
        current_spec=current,
        universe_metadata={"tickers": ["AAA", "BBB"]},
    )
    assert isinstance(result, SpecPatch)
    assert result.add_constraints[0].id == "cap_all"


def test_parse_repairs_after_invalid_first_response() -> None:
    """First canned response violates the schema (clarification with empty
    question); second is valid. Parse should succeed and the second call
    should have the validation error fed back."""
    client = FakeClient(
        [
            {"result": {"kind": "clarification", "question": "", "reason": "x"}},
            {
                "result": {
                    "kind": "clarification",
                    "question": "Which sector cap did you mean?",
                    "reason": "vague_quantity",
                }
            },
        ]
    )
    result = parse_user_message(
        "Cap sector.",
        client=client,
        current_spec=None,
        universe_metadata={"tickers": ["AAA"]},
    )
    assert isinstance(result, Clarification)
    # Verify the repair turn saw the prior bad output and the error.
    assert len(client.recorded_messages) == 2
    repair_msgs = client.recorded_messages[1]
    assert any(
        isinstance(m.get("content"), list)
        and any(
            b.get("type") == "tool_result" and b.get("is_error") for b in m["content"]
        )
        for m in repair_msgs
    )


def test_parse_raises_after_max_repair_attempts_exhausted() -> None:
    bad = {"result": {"kind": "clarification", "question": "", "reason": "x"}}
    # parse_user_message tries up to MAX_REPAIR_ATTEMPTS + 1 = 3 times.
    client = FakeClient([bad, bad, bad])
    with pytest.raises(ParseFailedError, match="Failed to obtain a valid ParseResult"):
        parse_user_message(
            "anything",
            client=client,
            current_spec=None,
            universe_metadata={"tickers": ["AAA"]},
        )
