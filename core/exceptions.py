"""Typed exceptions for the Truffle solver pipeline.

These are designed to be surfaced to the user (CLI / future agent tool output)
verbatim. Messages should be structured: state *what* failed, *which* spec
element caused it, and *what to try*. Avoid stack-trace-only diagnostics.
"""

from __future__ import annotations


class TruffleError(Exception):
    """Base class for all Truffle errors."""


class SpecValidationError(TruffleError):
    """Raised when a PortfolioSpec is structurally or semantically invalid."""


class CompilationError(TruffleError):
    """Raised when the IR cannot be turned into a well-formed CVXPY problem."""


class SolverError(TruffleError):
    """Raised when the solver fails for a reason other than infeasibility/unboundedness."""


class InfeasibleError(SolverError):
    """Raised when the solver reports the problem is infeasible.

    Sprint-3 will replace bare raises of this with an elastic relaxation
    diagnosis. For now we surface the solver status verbatim so the user
    knows which constraints to suspect.
    """


class UnboundedError(SolverError):
    """Raised when the solver reports the problem is unbounded."""


class DualsUnavailableError(TruffleError):
    """Raised when dual values are requested but the problem has not been solved
    (or was solved with a method that does not produce duals, e.g. MIP)."""
