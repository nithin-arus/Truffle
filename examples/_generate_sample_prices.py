"""Generate the bundled examples/prices_sample.csv from a fixed seed.

We never call yfinance or any network API in examples or tests — Sprint 1
demos must be reproducible offline. Run this script once to refresh the
sample CSV when the universe changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate(out_path: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    n_days = 504  # ~2 trading years
    # Heterogeneous daily vols so min-variance has a non-trivial preference.
    daily_vols = np.array([0.010, 0.014, 0.018, 0.024, 0.030])
    daily_drifts = np.array([0.0003, 0.0004, 0.0005, 0.0006, 0.0007])

    # Mild positive cross-correlation so the covariance isn't diagonal.
    corr = np.full((5, 5), 0.25)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    z = rng.standard_normal((n_days, 5)) @ chol.T
    log_returns = daily_drifts[None, :] + daily_vols[None, :] * z
    log_prices = np.cumsum(log_returns, axis=0)
    prices = 100.0 * np.exp(log_prices)

    dates = pd.bdate_range(end="2025-12-31", periods=n_days)
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "date"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, float_format="%.6f")
    print(f"Wrote {out_path} ({df.shape[0]} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    generate(Path(__file__).parent / "prices_sample.csv")
