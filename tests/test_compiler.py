"""Tests for the IR → CVXPY compiler.

The headline test is the analytical 3-asset minimum-variance case. Closed
form derivation (see ``test_minvariance_three_asset_analytical`` docstring):

    For a diagonal Σ = diag(σ²₁, σ²₂, σ²₃) with the full-investment constraint
    Σwᵢ = 1 and no other constraints, the Lagrangian gives

        wᵢ* = (1/σ²ᵢ) / Σⱼ (1/σ²ⱼ)

    i.e. each weight is proportional to the inverse variance ("inverse-vol
    weighting" up to the σ² → σ choice — here σ², because the objective is
    variance, not vol).

We verify Truffle's solver lands on that closed-form optimum to 1e-6.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from core.compiler import compile_spec
from core.exceptions import CompilationError
from core.ir import Box, Budget, LongOnly, MeanVariance, MinVariance, PortfolioSpec


def _solve(spec: PortfolioSpec, mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, float]:
    compiled = compile_spec(spec, mu, sigma)
    compiled.problem.solve(solver=cp.CLARABEL)
    assert compiled.problem.status == "optimal"
    return np.asarray(compiled.weights.value), float(compiled.problem.value)


def test_minvariance_three_asset_analytical() -> None:
    r"""Σ = diag(0.04, 0.09, 0.16), budget = 1 ⇒ w* ∝ 1/σ²ᵢ.

    Inverse-variance: [1/0.04, 1/0.09, 1/0.16] = [25, 11.111…, 6.25].
    Normalize so they sum to 1: divide by 42.361…
    Expected w* ≈ [0.5902, 0.2624, 0.1475] (to 4 dp).
    """
    universe = ["A", "B", "C"]
    sigma = np.diag([0.04, 0.09, 0.16])
    mu = np.zeros(3)  # unused for min-variance

    spec = PortfolioSpec(
        universe=universe,
        objective=MinVariance(),
        constraints=[Budget(total=1.0)],
    )
    w, obj_value = _solve(spec, mu, sigma)

    inv = 1.0 / np.diag(sigma)
    expected = inv / inv.sum()
    np.testing.assert_allclose(w, expected, atol=1e-6)
    # Objective value matches wᵀ Σ w at the optimum.
    np.testing.assert_allclose(obj_value, float(expected @ sigma @ expected), atol=1e-9)


def test_meanvariance_respects_risk_aversion_direction() -> None:
    """Higher λ tilts mean-variance toward the higher-return asset."""
    universe = ["A", "B"]
    sigma = np.diag([0.04, 0.04])  # symmetric variance so mean direction dominates
    mu = np.array([0.10, 0.05])  # asset A has higher expected return

    spec_lo = PortfolioSpec(
        universe=universe,
        objective=MeanVariance(risk_aversion=0.1),
        constraints=[Budget(total=1.0), LongOnly()],
    )
    spec_hi = PortfolioSpec(
        universe=universe,
        objective=MeanVariance(risk_aversion=10.0),
        constraints=[Budget(total=1.0), LongOnly()],
    )
    w_lo, _ = _solve(spec_lo, mu, sigma)
    w_hi, _ = _solve(spec_hi, mu, sigma)
    # Higher λ → more weight in the higher-μ asset.
    assert w_hi[0] > w_lo[0] + 1e-3


def test_compile_rejects_shape_mismatch() -> None:
    spec = PortfolioSpec(
        universe=["A", "B", "C"],
        objective=MinVariance(),
        constraints=[Budget()],
    )
    with pytest.raises(CompilationError, match="does not match universe size"):
        compile_spec(spec, mu=np.zeros(3), sigma=np.eye(2))
    with pytest.raises(CompilationError, match="Expected-return vector"):
        compile_spec(spec, mu=np.zeros(2), sigma=np.eye(3))


def test_compile_rejects_asymmetric_sigma() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget()],
    )
    asym = np.array([[0.04, 0.01], [0.02, 0.04]])  # 0.01 vs 0.02 → not symmetric
    with pytest.raises(CompilationError, match="not symmetric"):
        compile_spec(spec, mu=np.zeros(2), sigma=asym)


def test_constraint_objs_keyed_by_ir_id() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[
            Budget(id="my_budget"),
            LongOnly(id="my_longonly"),
            Box(id="my_box", lower=0.0, upper=0.8),
        ],
    )
    compiled = compile_spec(spec, mu=np.zeros(2), sigma=np.eye(2))
    assert set(compiled.constraint_objs.keys()) == {"my_budget", "my_longonly", "my_box"}
