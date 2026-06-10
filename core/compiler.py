"""IR → CVXPY compiler.

This module owns *all* of the math. The LLM layer will never construct CVXPY
expressions directly; it only emits IR, and :func:`compile_spec` deterministically
translates that IR into a CVXPY ``Problem``. No string-built expressions, no
``eval``, no LLM involvement.

The compiler also returns the dictionary ``constraint_objs`` that maps every
IR constraint's ``id`` to its CVXPY ``Constraint`` object. :mod:`core.duals`
walks that dict after the solve to lift shadow prices back into the IR's
naming, which is the foundation of the explanation layer (BLUEPRINT §5/§6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from core.exceptions import CompilationError
from core.ir import Box, Budget, LongOnly, MeanVariance, MinCVaR, MinVariance, PortfolioSpec


@dataclass(slots=True)
class CompiledProblem:
    """Container for everything the solver layer needs after compilation.

    Attributes:
        problem: The CVXPY ``Problem``. Solve it externally so the compiler
            stays a pure builder (easier to test, easier to reason about).
        weights: The ``cp.Variable`` representing the asset weight vector.
        constraint_objs: ``{ir_constraint_id -> cvxpy.Constraint}``. Used by
            :mod:`core.duals` to recover shadow prices and name them back to
            the user.
        spec: The originating ``PortfolioSpec`` (kept for downstream reporting).
        extra_vars: Objective-specific auxiliary variables. For ``min_cvar``
            this exposes ``{"t": <scalar Variable>, "z": <S-vector Variable>}``
            so the caller can read VaR (``= t.value``) after the solve.
    """

    problem: cp.Problem
    weights: cp.Variable
    constraint_objs: dict[str, cp.Constraint] = field(default_factory=dict)
    spec: PortfolioSpec | None = None
    extra_vars: dict[str, cp.Variable] = field(default_factory=dict)


def _validate_inputs(spec: PortfolioSpec, mu: np.ndarray, sigma: np.ndarray) -> None:
    n = len(spec.universe)
    if sigma.shape != (n, n):
        raise CompilationError(
            f"Covariance shape {sigma.shape} does not match universe size {n}."
        )
    if mu.shape != (n,):
        raise CompilationError(
            f"Expected-return vector shape {mu.shape} does not match universe size ({n},)."
        )
    # Symmetrize sigma defensively — CVXPY's `quad_form` insists on PSD, and
    # off-by-eps asymmetry from floating point is a common compile-time surprise.
    asym = float(np.max(np.abs(sigma - sigma.T))) if sigma.size else 0.0
    if asym > 1e-8:
        raise CompilationError(
            f"Covariance matrix is not symmetric (max |Σ − Σᵀ| = {asym:.2e})."
        )


def _build_quad_objective(
    spec: PortfolioSpec, w: cp.Variable, mu: np.ndarray, sigma: np.ndarray
) -> cp.Minimize:
    obj = spec.objective
    # `cp.psd_wrap` tells CVXPY to trust the matrix as PSD; the Ledoit–Wolf
    # estimator we use guarantees this, but a sample Σ on a tiny window can
    # be numerically indefinite, so wrapping is the safe contract.
    quad = cp.quad_form(w, cp.psd_wrap(sigma))
    if isinstance(obj, MinVariance):
        return cp.Minimize(quad)
    if isinstance(obj, MeanVariance):
        return cp.Minimize(quad - obj.risk_aversion * (mu @ w))
    raise CompilationError(f"_build_quad_objective called on non-quadratic: {type(obj).__name__}")


def _build_constraint(
    c: Budget | LongOnly | Box, w: cp.Variable, ticker_index: dict[str, int]
) -> cp.Constraint:
    if isinstance(c, Budget):
        # Σ w = total. Stays an equality so its dual is a free-sign multiplier
        # — duals on equalities can be negative; the explanation layer
        # interprets the sign per BLUEPRINT §5 "duals everywhere".
        return cp.sum(w) == c.total
    if isinstance(c, LongOnly):
        return w >= 0.0
    if isinstance(c, Box):
        if c.tickers is None:
            target = w
        else:
            idx = np.array([ticker_index[t] for t in c.tickers], dtype=int)
            target = w[idx]
        # Stack lower-side and upper-side slacks into a single non-negativity
        # constraint so this Box maps to *one* CVXPY Constraint (one id, one
        # dual vector of length 2k). First k entries are duals on the lower
        # bound, last k on the upper bound — :mod:`core.duals` documents this.
        slacks = cp.hstack([target - c.lower, c.upper - target])
        return slacks >= 0
    raise CompilationError(f"Unsupported constraint kind: {type(c).__name__}")


def compile_spec(
    spec: PortfolioSpec,
    mu: np.ndarray,
    sigma: np.ndarray,
    scenarios: np.ndarray | None = None,
) -> CompiledProblem:
    """Deterministically build a CVXPY problem from an IR spec.

    Args:
        spec: Validated ``PortfolioSpec``.
        mu: Expected-return vector, length ``len(spec.universe)``. Ignored by
            min-variance and min-CVaR objectives but the signature is uniform
            across kinds so callers don't branch.
        sigma: Annualized covariance matrix, shape ``(n, n)``, symmetric PSD.
            Ignored by min-CVaR but accepted for signature uniformity.
        scenarios: Per-period return matrix of shape ``(S, n)`` used by the
            CVaR objective only. ``None`` is allowed when the objective does
            not require scenarios; ``None`` with a ``min_cvar`` objective
            raises :class:`CompilationError`.

    Returns:
        ``CompiledProblem`` wrapping the unsolved CVXPY problem, the weight
        variable, the IR-id → cvxpy.Constraint map, and any objective-specific
        auxiliary variables (``t`` and ``z`` for CVaR).

    Raises:
        CompilationError: if shapes mismatch, Σ is not symmetric, scenarios
            are missing/malformed for CVaR, or the IR contains an
            objective/constraint kind the compiler does not understand.
    """
    _validate_inputs(spec, mu, sigma)

    n = len(spec.universe)
    w = cp.Variable(n, name="w")
    ticker_index = {t: i for i, t in enumerate(spec.universe)}

    constraint_objs: dict[str, cp.Constraint] = {}
    for c in spec.constraints:
        constraint_objs[c.id] = _build_constraint(c, w, ticker_index)

    extra_vars: dict[str, cp.Variable] = {}
    all_constraints = list(constraint_objs.values())

    obj = spec.objective
    if isinstance(obj, MinCVaR):
        if scenarios is None:
            raise CompilationError(
                "min_cvar objective requires a scenario matrix; got scenarios=None. "
                "Pass scenarios from data.scenarios.historical_scenarios(prices) (or another generator)."
            )
        scenarios = np.asarray(scenarios, dtype=float)
        if scenarios.ndim != 2 or scenarios.shape[1] != n:
            raise CompilationError(
                f"Scenario matrix must have shape (S, {n}); got {scenarios.shape}."
            )
        s = scenarios.shape[0]
        if s < 1:
            raise CompilationError("Scenario matrix must have at least one scenario row.")
        t_var = cp.Variable(name="t")  # = VaR at optimum
        z_var = cp.Variable(s, name="z", nonneg=True)
        # Rockafellar–Uryasev: losses are −r·w. We do NOT add `z >= 0` here
        # separately because `nonneg=True` already encodes it (more efficient).
        loss = -scenarios @ w
        all_constraints = all_constraints + [z_var >= loss - t_var]
        objective = cp.Minimize(t_var + cp.sum(z_var) / ((1.0 - obj.cvar_alpha) * s))
        extra_vars = {"t": t_var, "z": z_var}
    else:
        objective = _build_quad_objective(spec, w, mu, sigma)

    problem = cp.Problem(objective, all_constraints)

    return CompiledProblem(
        problem=problem,
        weights=w,
        constraint_objs=constraint_objs,
        spec=spec,
        extra_vars=extra_vars,
    )
