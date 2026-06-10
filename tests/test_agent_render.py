"""Tests for agent/render.py: deterministic spec echo and patch diff."""

from __future__ import annotations

from agent.render import render_patch, render_spec
from agent.schema import SpecPatch, apply_patch
from core.ir import (
    Box,
    Budget,
    LongOnly,
    MeanVariance,
    MinCVaR,
    MinVariance,
    PortfolioSpec,
)


def test_render_spec_minvariance_shows_problem_class_convex() -> None:
    spec = PortfolioSpec(
        universe=["AAA", "BBB"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Box(lower=0.0, upper=0.5)],
    )
    out = render_spec(spec)
    assert "Minimize variance" in out
    assert "(budget)" in out
    assert "(long-only)" in out
    assert "(box)" in out
    assert "Problem class: CONVEX" in out


def test_render_spec_meanvariance_shows_lambda() -> None:
    spec = PortfolioSpec(
        universe=["A"],
        objective=MeanVariance(risk_aversion=2.5),
        constraints=[Budget()],
    )
    out = render_spec(spec)
    assert "λ = 2.5" in out


def test_render_spec_cvar_shows_alpha() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinCVaR(cvar_alpha=0.9),
        constraints=[Budget(), LongOnly()],
    )
    out = render_spec(spec)
    assert "Minimize CVaR at α = 0.9" in out


def test_render_spec_box_tickers_rendered() -> None:
    spec = PortfolioSpec(
        universe=["AAA", "BBB", "CCC"],
        objective=MinVariance(),
        constraints=[Box(id="cap_aaa", lower=0.0, upper=0.4, tickers=["AAA"])],
    )
    out = render_spec(spec)
    assert "on AAA" in out
    assert "[cap_aaa]" in out


def test_render_patch_shows_objective_swap_and_constraint_changes() -> None:
    before = PortfolioSpec(
        universe=["AAA", "BBB", "CCC"],
        objective=MinVariance(),
        constraints=[
            Budget(id="b"),
            LongOnly(id="lo"),
            Box(id="cap_aaa", lower=0.0, upper=0.35, tickers=["AAA"]),
        ],
    )
    patch = SpecPatch(
        remove_constraint_ids=["cap_aaa"],
        replace_objective=MinCVaR(cvar_alpha=0.9),
        add_constraints=[Box(id="cap_all", lower=0.0, upper=0.40)],
    )
    after = apply_patch(before, patch)
    diff = render_patch(patch, before, after)
    assert "Removed constraint: cap_aaa" in diff
    assert "Added constraint" in diff and "cap_all" in diff
    assert "Minimize variance" in diff and "Minimize CVaR" in diff
    assert "Problem class is now: CONVEX" in diff
