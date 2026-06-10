"""Tests for the min-CVaR objective and scenario plumbing.

Hand-checkable case (the spec from the kickoff prompt):

    A single asset whose return scenarios are equally likely and equal
    [+2, +1, 0, -1, -4] percent. The portfolio is forced to w = 1.0 by
    Budget + LongOnly, so portfolio losses ``L_s = -r_s`` are
    [-2, -1, 0, 1, 4] percent. At alpha = 0.8:

        VaR_0.8  = inf{x : P(L <= x) >= 0.8}.
                   P(L <= 1) = 4/5 = 0.8  ⇒  VaR = 1 percent.

        CVaR_0.8 = t* + (1/((1-alpha)S)) Σ max(L_s - t*, 0)
                 = 1 + (1/1) * max(4 - 1, 0)
                 = 4 percent.

We verify the LP recovers VaR = 1 and CVaR = 4 (in *return-fraction*
units; scenarios are passed in fractions, e.g. 0.04 for 4 percent).

We also check the monotonicity property: the CVaR objective value is
non-decreasing as alpha increases (a tighter tail confidence cannot
produce a lower CVaR).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from core.compiler import compile_spec
from core.exceptions import CompilationError
from core.ir import Budget, LongOnly, MinCVaR, PortfolioSpec
from data.scenarios import historical_scenarios


def test_cvar_hand_checkable_single_asset() -> None:
    universe = ["A"]
    scenarios = np.array([[0.02], [0.01], [0.00], [-0.01], [-0.04]])  # returns

    spec = PortfolioSpec(
        universe=universe,
        objective=MinCVaR(cvar_alpha=0.8),
        constraints=[Budget(total=1.0), LongOnly()],
    )
    compiled = compile_spec(
        spec,
        mu=np.zeros(1),
        sigma=np.eye(1),
        scenarios=scenarios,
    )
    compiled.problem.solve(solver=cp.CLARABEL)
    assert compiled.problem.status == "optimal"

    # Single asset + budget=1 + long-only => w = 1.
    np.testing.assert_allclose(compiled.weights.value, [1.0], atol=1e-6)
    # CVaR = 4% is the testable invariant.
    np.testing.assert_allclose(compiled.problem.value, 0.04, atol=1e-6)
    # When alpha lands on an atom of the empirical distribution, the LP has
    # a flat optimal face: any t in [VaR_inf, max_loss] gives the optimal
    # objective. With losses [-2,-1,0,1,4]% and alpha=0.8, the inf-quantile
    # VaR is 1% but the LP may pick any t in [0.01, 0.04]. The
    # alpha=0.7 case below tests the unique-VaR branch.
    t = float(compiled.extra_vars["t"].value)
    assert 0.01 - 1e-6 <= t <= 0.04 + 1e-6, t


def test_cvar_var_uniquely_recovered_at_non_atom_alpha() -> None:
    """At alpha = 0.7 the optimum is uniquely t* = VaR = 0.01 = 1 percent.

    Derivation: f(t) = t + (2/3) * Σ_s max(L_s - t, 0). For t in [0, 1]
    the slope is -1/3; for t in [1, 4] the slope is +1/3; so the unique
    optimum is t = 0.01 with f(0.01) = 0.03 = 3 percent CVaR.
    """
    scenarios = np.array([[0.02], [0.01], [0.00], [-0.01], [-0.04]])
    spec = PortfolioSpec(
        universe=["A"],
        objective=MinCVaR(cvar_alpha=0.7),
        constraints=[Budget(total=1.0), LongOnly()],
    )
    compiled = compile_spec(spec, mu=np.zeros(1), sigma=np.eye(1), scenarios=scenarios)
    compiled.problem.solve(solver=cp.CLARABEL)
    assert compiled.problem.status == "optimal"
    np.testing.assert_allclose(compiled.extra_vars["t"].value, 0.01, atol=1e-6)
    np.testing.assert_allclose(compiled.problem.value, 0.03, atol=1e-6)


def test_cvar_geq_var() -> None:
    """KKT property: CVaR >= VaR at the optimum, for any alpha and scenarios."""
    universe = ["A", "B"]
    rng = np.random.default_rng(11)
    scenarios = rng.normal(0.0, 0.02, size=(200, 2))

    spec = PortfolioSpec(
        universe=universe,
        objective=MinCVaR(cvar_alpha=0.9),
        constraints=[Budget(total=1.0), LongOnly()],
    )
    compiled = compile_spec(spec, mu=np.zeros(2), sigma=np.eye(2), scenarios=scenarios)
    compiled.problem.solve(solver=cp.CLARABEL)
    assert compiled.problem.status == "optimal"
    var = float(compiled.extra_vars["t"].value)
    cvar = float(compiled.problem.value)
    assert cvar >= var - 1e-9


def test_cvar_monotone_in_alpha() -> None:
    """A tighter tail confidence (higher alpha) cannot lower the optimal CVaR."""
    universe = ["A", "B", "C"]
    rng = np.random.default_rng(7)
    scenarios = rng.normal(0.0, 0.02, size=(200, 3))

    def cvar_at(alpha: float) -> float:
        spec = PortfolioSpec(
            universe=universe,
            objective=MinCVaR(cvar_alpha=alpha),
            constraints=[Budget(total=1.0), LongOnly()],
        )
        compiled = compile_spec(spec, mu=np.zeros(3), sigma=np.eye(3), scenarios=scenarios)
        compiled.problem.solve(solver=cp.CLARABEL)
        assert compiled.problem.status == "optimal"
        return float(compiled.problem.value)

    c80 = cvar_at(0.80)
    c90 = cvar_at(0.90)
    c95 = cvar_at(0.95)
    assert c80 <= c90 + 1e-9
    assert c90 <= c95 + 1e-9


def test_cvar_without_scenarios_raises() -> None:
    spec = PortfolioSpec(
        universe=["A"],
        objective=MinCVaR(cvar_alpha=0.95),
        constraints=[Budget(), LongOnly()],
    )
    with pytest.raises(CompilationError, match="requires a scenario matrix"):
        compile_spec(spec, mu=np.zeros(1), sigma=np.eye(1))


def test_cvar_scenarios_shape_mismatch_raises() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinCVaR(cvar_alpha=0.95),
        constraints=[Budget(), LongOnly()],
    )
    bad = np.zeros((50, 3))  # 3 cols but universe size 2
    with pytest.raises(CompilationError, match="Scenario matrix must have shape"):
        compile_spec(spec, mu=np.zeros(2), sigma=np.eye(2), scenarios=bad)


def test_historical_scenarios_shape_and_values() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 49.0, 50.47]})
    returns = historical_scenarios(prices)
    assert returns.shape == (2, 2)
    np.testing.assert_allclose(returns[0], [0.10, -0.02], atol=1e-9)
    np.testing.assert_allclose(returns[1], [0.10, 50.47 / 49.0 - 1.0], atol=1e-9)


def test_historical_scenarios_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        historical_scenarios(pd.DataFrame({"A": [100.0]}))
    with pytest.raises(ValueError, match="strictly positive"):
        historical_scenarios(pd.DataFrame({"A": [100.0, 0.0]}))
