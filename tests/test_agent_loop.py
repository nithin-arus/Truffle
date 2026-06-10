"""Chat-loop integration tests with a fake LLMClient.

Golden path:
    user message -> FreshSpec parse -> render echo -> confirm 'y' ->
    real Clarabel solve on the bundled prices -> grounded explain ->
    SolutionReport carried through TurnResult.

Repair-loop test: parse fails once then succeeds.

Amendment test: a SpecPatch removes the Box and replaces the objective;
the re-render shows the diff and the new spec is solvable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent.loop import ChatSession, TurnResult
from agent.schema import ParseEnvelope
from core.exceptions import ParseFailedError

EXAMPLES = Path(__file__).parent.parent / "examples"


class FakeClient:
    """Returns canned tool/text responses in order."""

    def __init__(self, *, tool: list[dict], text: list[str]) -> None:
        self._tool = list(tool)
        self._text = list(text)
        self.tool_calls = 0
        self.text_calls = 0

    def call_tool(self, **_kwargs):
        self.tool_calls += 1
        if not self._tool:
            raise AssertionError("FakeClient ran out of tool responses.")
        return self._tool.pop(0)

    def call_text(self, **_kwargs):
        self.text_calls += 1
        if not self._text:
            raise AssertionError("FakeClient ran out of text responses.")
        return self._text.pop(0)


@pytest.fixture
def prices() -> pd.DataFrame:
    return pd.read_csv(EXAMPLES / "prices_sample.csv", parse_dates=[0], index_col=0)


def _fresh_spec_payload() -> dict:
    return {
        "result": {
            "kind": "fresh_spec",
            "spec": {
                "universe": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                "objective": {"kind": "min_variance"},
                "constraints": [
                    {"kind": "budget", "id": "b", "total": 1.0},
                    {"kind": "long_only", "id": "lo"},
                    {
                        "kind": "box",
                        "id": "cap_aaa",
                        "lower": 0.0,
                        "upper": 0.35,
                        "tickers": ["AAA"],
                    },
                ],
            },
        }
    }


def _grounded_text(report) -> str:
    """Template-style narration that will pass verify()."""
    # Use exact objective_value and binders so we know verify() passes.
    lines = [
        f"Minimum-variance solve, objective value {report.objective_value:.6f}, "
        f"status {report.status}, {report.nonzero_names} of {report.n_assets} names nonzero.",
    ]
    if report.binding:
        b0 = report.binding[0]
        lines.append(
            f"The most binding constraint is {b0.human_name} with shadow price "
            f"{b0.shadow_price:.6f}."
        )
    return " ".join(lines)


def test_golden_path_message_to_grounded_solve(prices: pd.DataFrame) -> None:
    # 1. parse -> FreshSpec; 2. user confirms; 3. solve+report; 4. grounded explain.
    # The grounded narration is built off the report we will produce, so we need
    # to know the report first. Trick: temporarily inject a placeholder, then
    # discover the real values by running solve_spec ourselves, then build a
    # session with a client that returns the right narration.
    from core.solve import solve_spec  # noqa: PLC0415

    spec_payload = _fresh_spec_payload()
    pre_spec = ParseEnvelope.model_validate(spec_payload).result.spec
    _, report_for_narration = solve_spec(pre_spec, prices)
    narration = _grounded_text(report_for_narration)

    client = FakeClient(tool=[spec_payload], text=[narration])
    session = ChatSession(client=client, prices=prices)

    echo_result = session.handle_user_message(
        "Minimum variance, long only, fully invested, cap AAA at 35 percent."
    )
    assert echo_result.kind == "echo"
    assert "Minimize variance" in echo_result.text
    assert "Problem class: CONVEX" in echo_result.text
    assert session.pending is not None

    solved = session.confirm_pending("y")
    assert solved.kind == "solved"
    assert solved.report is not None
    assert abs(solved.report.weights["AAA"] - 0.35) < 1e-6
    assert solved.explanation is not None
    assert f"{solved.report.objective_value:.6f}" in solved.explanation
    assert session.current_spec is not None
    assert session.pending is None


def test_clarification_does_not_advance(prices: pd.DataFrame) -> None:
    client = FakeClient(
        tool=[
            {
                "result": {
                    "kind": "clarification",
                    "question": "What is the maximum per-name weight?",
                    "reason": "vague_quantity",
                }
            }
        ],
        text=[],
    )
    session = ChatSession(client=client, prices=prices)
    result = session.handle_user_message("Minimize risk but not too concentrated.")
    assert result.kind == "clarification"
    assert session.pending is None
    assert session.current_spec is None


def test_user_can_discard_pending(prices: pd.DataFrame) -> None:
    client = FakeClient(tool=[_fresh_spec_payload()], text=[])
    session = ChatSession(client=client, prices=prices)
    session.handle_user_message("Set up a min-var portfolio.")
    assert session.pending is not None
    out = session.confirm_pending("n")
    assert out.kind == "info"
    assert session.pending is None
    assert session.current_spec is None  # not committed because user said no


def test_amendment_patches_and_diffs(prices: pd.DataFrame) -> None:
    # Two-turn session: first FreshSpec + confirm + solve, then a SpecPatch.
    from core.solve import solve_spec  # noqa: PLC0415

    spec_payload = _fresh_spec_payload()
    pre_spec = ParseEnvelope.model_validate(spec_payload).result.spec
    _, r1 = solve_spec(pre_spec, prices)
    n1 = _grounded_text(r1)

    # The patch replaces objective with CVaR and removes the AAA box.
    patch_payload = {
        "result": {
            "kind": "spec_patch",
            "remove_constraint_ids": ["cap_aaa"],
            "replace_objective": {"kind": "min_cvar", "cvar_alpha": 0.9},
        }
    }
    # Apply the patch by hand to get the report for narration #2.
    from agent.schema import SpecPatch, apply_patch  # noqa: PLC0415

    patch = SpecPatch.model_validate(patch_payload["result"])
    spec2 = apply_patch(pre_spec, patch)
    _, r2 = solve_spec(spec2, prices)
    n2 = _grounded_text(r2)

    client = FakeClient(tool=[spec_payload, patch_payload], text=[n1, n2])
    session = ChatSession(client=client, prices=prices)
    session.handle_user_message("Set up min-var, cap AAA at 35.")
    session.confirm_pending("y")
    assert session.current_spec is not None

    echo2 = session.handle_user_message("Switch to CVaR at 90 and drop the AAA cap.")
    assert echo2.kind == "echo"
    assert "Removed constraint: cap_aaa" in echo2.text
    assert "Minimize CVaR" in echo2.text
    solved2 = session.confirm_pending("y")
    assert solved2.kind == "solved"
    assert solved2.report.objective_kind == "min_cvar"
    assert solved2.report.var is not None


def test_repair_loop_succeeds_on_second_attempt(prices: pd.DataFrame) -> None:
    bad = {"result": {"kind": "clarification", "question": "", "reason": "x"}}
    good = {
        "result": {
            "kind": "clarification",
            "question": "What's the max per-name weight?",
            "reason": "vague_quantity",
        }
    }
    client = FakeClient(tool=[bad, good], text=[])
    session = ChatSession(client=client, prices=prices)
    result = session.handle_user_message("not too concentrated")
    assert result.kind == "clarification"
    assert "max per-name" in result.text


def test_repair_loop_raises_then_loop_returns_error(prices: pd.DataFrame) -> None:
    bad = {"result": {"kind": "clarification", "question": "", "reason": "x"}}
    client = FakeClient(tool=[bad, bad, bad], text=[])
    session = ChatSession(client=client, prices=prices)
    result = session.handle_user_message("anything")
    assert result.kind == "error"
    assert "parse" in result.text.lower()


def test_reset_clears_state(prices: pd.DataFrame) -> None:
    client = FakeClient(tool=[_fresh_spec_payload()], text=[])
    session = ChatSession(client=client, prices=prices)
    session.handle_user_message("min var").pending_spec  # noqa: B018
    assert session.pending is not None
    out: TurnResult = session.handle_user_message("start over")
    assert out.kind == "info"
    assert session.pending is None
    assert session.current_spec is None


def test_parse_failed_surfaces_cleanly_in_loop(prices: pd.DataFrame) -> None:
    class AlwaysRaises:
        def call_tool(self, **_kw):
            raise ParseFailedError("simulated")

        def call_text(self, **_kw):
            raise NotImplementedError

    session = ChatSession(client=AlwaysRaises(), prices=prices)
    out = session.handle_user_message("anything")
    assert out.kind == "error"
