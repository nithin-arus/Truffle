"""Tests for the Truffle IR (core/ir.py).

We exercise: (1) discriminated-union dispatch on the ``kind`` tag,
(2) auto-id generation + uniqueness checking, (3) semantic validations
(duplicate tickers, box references to unknown tickers, lower>upper),
(4) problem-class aggregation.
"""

from __future__ import annotations

import pytest

from core.ir import (
    Box,
    Budget,
    LongOnly,
    MeanVariance,
    MinVariance,
    PortfolioSpec,
)


def test_minvariance_spec_validates_and_aggregates_convex() -> None:
    spec = PortfolioSpec(
        universe=["AAPL", "MSFT", "NVDA"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Box(lower=0.0, upper=0.5)],
    )
    assert spec.problem_class == "convex"
    # Auto-generated ids are unique and prefixed.
    ids = [c.id for c in spec.constraints]
    assert len(set(ids)) == 3
    assert ids[0].startswith("budget_")
    assert ids[1].startswith("longonly_")
    assert ids[2].startswith("box_")


def test_mean_variance_requires_positive_risk_aversion() -> None:
    with pytest.raises(ValueError):
        MeanVariance(risk_aversion=0.0)
    with pytest.raises(ValueError):
        MeanVariance(risk_aversion=-1.0)
    mv = MeanVariance(risk_aversion=2.5)
    assert mv.kind == "mean_variance"


def test_box_rejects_lower_above_upper() -> None:
    with pytest.raises(ValueError):
        Box(lower=0.5, upper=0.1)


def test_box_tickers_must_be_in_universe() -> None:
    with pytest.raises(ValueError, match="not in universe"):
        PortfolioSpec(
            universe=["AAPL", "MSFT"],
            objective=MinVariance(),
            constraints=[Box(lower=0.0, upper=0.2, tickers=["GOOG"])],
        )


def test_universe_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate ticker"):
        PortfolioSpec(
            universe=["AAPL", "AAPL"],
            objective=MinVariance(),
        )


def test_duplicate_constraint_ids_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate constraint ids"):
        PortfolioSpec(
            universe=["AAPL", "MSFT"],
            objective=MinVariance(),
            constraints=[
                Budget(id="dup"),
                Box(id="dup", lower=0.0, upper=1.0),
            ],
        )


def test_explicit_ids_preserved() -> None:
    spec = PortfolioSpec(
        universe=["AAPL", "MSFT"],
        objective=MinVariance(),
        constraints=[Budget(id="b1"), LongOnly(id="lo1")],
    )
    assert [c.id for c in spec.constraints] == ["b1", "lo1"]


def test_discriminator_picks_correct_constraint_class() -> None:
    # Round-trip through dict to simulate YAML/JSON input the agent will produce.
    payload = {
        "universe": ["AAPL", "MSFT"],
        "objective": {"kind": "mean_variance", "risk_aversion": 1.0},
        "constraints": [
            {"kind": "budget", "total": 1.0},
            {"kind": "long_only"},
            {"kind": "box", "lower": 0.0, "upper": 0.4},
        ],
    }
    spec = PortfolioSpec.model_validate(payload)
    assert isinstance(spec.objective, MeanVariance)
    assert isinstance(spec.constraints[0], Budget)
    assert isinstance(spec.constraints[1], LongOnly)
    assert isinstance(spec.constraints[2], Box)


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValueError):
        PortfolioSpec.model_validate(
            {
                "universe": ["AAPL"],
                "objective": {"kind": "min_variance"},
                "constraints": [],
                "junk": True,
            }
        )
