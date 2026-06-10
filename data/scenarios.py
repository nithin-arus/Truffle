"""Scenario generation for CVaR (Rockafellar–Uryasev).

CVaR is evaluated as a *sample average over scenarios* — each row of the
returned matrix is one realization of asset returns. Sprint 2 ships the
simplest scenario source — historical returns from the price panel — which
is also the right baseline for evaluation. Block bootstrap and IID
bootstrap (BLUEPRINT §5) come later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_scenarios(prices: pd.DataFrame) -> np.ndarray:
    """Return the historical scenario matrix.

    Args:
        prices: ``DataFrame`` indexed by date with one column per asset,
            in the canonical universe order. Must have at least 2 rows.

    Returns:
        ``np.ndarray`` of shape ``(S, N)`` of per-period *simple* returns
        (``p_t / p_{t-1} − 1``). We use simple returns here, not log
        returns: portfolio P&L over one period equals ``w · r_simple``
        only for simple returns; using log returns would mis-state CVaR
        for any non-trivial holding.

    Raises:
        ValueError: if the panel is too small or contains non-positive prices.
    """
    if prices.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 price observations to build scenarios; got {prices.shape[0]}."
        )
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    arr = prices.to_numpy(dtype=float)
    returns = arr[1:] / arr[:-1] - 1.0
    return returns
