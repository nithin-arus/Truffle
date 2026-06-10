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
from core.ir import Box, Budget, LongOnly, MeanVariance, MinVariance, PortfolioSpec


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
    """

    problem: cp.Problem
    weights: cp.Variable
    constraint_objs: dict[str, cp.Constraint] = field(default_factory=dict)
    spec: PortfolioSpec | None = None


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


def _build_objective(
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
    raise CompilationError(f"Unsupported objective kind: {type(obj).__name__}")


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
    spec: PortfolioSpec, mu: np.ndarray, sigma: np.ndarray
) -> CompiledProblem:
    """Deterministically build a CVXPY problem from an IR spec.

    Args:
        spec: Validated ``PortfolioSpec``.
        mu: Expected-return vector, length ``len(spec.universe)``. Ignored by
            min-variance objectives but the signature is uniform across kinds
            so callers don't branch.
        sigma: Annualized covariance matrix, shape ``(n, n)``, symmetric PSD.

    Returns:
        ``CompiledProblem`` wrapping the unsolved CVXPY problem, the weight
        variable, and the IR-id → cvxpy.Constraint map.

    Raises:
        CompilationError: if shapes mismatch, Σ is not symmetric, or the IR
            contains an objective/constraint kind the compiler does not
            understand (a structural bug — the IR should have rejected it).
    """
    _validate_inputs(spec, mu, sigma)

    n = len(spec.universe)
    w = cp.Variable(n, name="w")
    ticker_index = {t: i for i, t in enumerate(spec.universe)}

    # Box constraints encode as vstacked >= / <= pairs (two scalar constraints
    # per Box), but CVXPY's `vstack` of constraints returns a single composite
    # Constraint object whose dual is a stacked vector. That's the right shape
    # for duals.harvest_duals: one IR id, one dual array.
    constraint_objs: dict[str, cp.Constraint] = {}
    for c in spec.constraints:
        constraint_objs[c.id] = _build_constraint(c, w, ticker_index)

    objective = _build_objective(spec, w, mu, sigma)
    problem = cp.Problem(objective, list(constraint_objs.values()))

    return CompiledProblem(
        problem=problem,
        weights=w,
        constraint_objs=constraint_objs,
        spec=spec,
    )
