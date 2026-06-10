"""Truffle CLI.

Usage:
    python cli.py solve examples/spec_minvar.yaml --prices examples/prices_sample.csv

Loads a YAML spec into the IR, validates it, estimates mu/sigma from a CSV
price panel, compiles and solves with Clarabel, and pretty-prints the
results (weights table, objective, problem class, solver stats, shadow
prices) using Rich.
"""

from __future__ import annotations

import time
from pathlib import Path

import cvxpy as cp
import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from core.compiler import compile_spec
from core.duals import harvest_duals
from core.exceptions import InfeasibleError, SolverError, TruffleError, UnboundedError
from core.ir import PortfolioSpec
from data.estimation import estimate_moments

app = typer.Typer(add_completion=False, help="Truffle: typed portfolio optimization.")
console = Console()


@app.callback()
def _root() -> None:
    """Truffle: typed portfolio optimization.

    A no-op callback so Typer treats ``solve`` as an explicit subcommand
    rather than promoting it to the default entrypoint.
    """


def _load_spec(spec_path: Path) -> PortfolioSpec:
    with spec_path.open("r") as f:
        payload = yaml.safe_load(f)
    return PortfolioSpec.model_validate(payload)


def _load_prices(prices_path: Path, universe: list[str]) -> pd.DataFrame:
    df = pd.read_csv(prices_path, parse_dates=[0], index_col=0)
    missing = [t for t in universe if t not in df.columns]
    if missing:
        raise SystemExit(
            f"Prices CSV is missing columns for universe tickers: {missing}.\n"
            f"  CSV columns: {list(df.columns)}"
        )
    # Reorder to match the spec's universe — order must be canonical
    # because compile_spec indexes mu/sigma by position.
    return df[universe]


def _render_weights(spec: PortfolioSpec, weights: list[float]) -> Table:
    t = Table(title="Optimal weights", show_lines=False)
    t.add_column("Ticker", style="bold cyan")
    t.add_column("Weight", justify="right")
    t.add_column("Weight %", justify="right")
    for ticker, w in zip(spec.universe, weights, strict=True):
        t.add_row(ticker, f"{w:.6f}", f"{100.0 * w:.2f}%")
    return t


def _render_duals(spec: PortfolioSpec, duals: dict[str, float]) -> Table:
    t = Table(title="Shadow prices (binding constraints first)")
    t.add_column("Constraint id", style="bold magenta")
    t.add_column("Kind")
    t.add_column("Shadow price", justify="right")
    t.add_column("Binding?", justify="center")
    kinds = {c.id: c.kind for c in spec.constraints}
    rows = sorted(duals.items(), key=lambda kv: -abs(kv[1]))
    for cid, sp in rows:
        binding = "yes" if abs(sp) > 1e-6 else "no"
        t.add_row(cid, kinds.get(cid, "?"), f"{sp:.6f}", binding)
    return t


@app.command()
def solve(
    spec_path: Path = typer.Argument(..., exists=True, readable=True, help="YAML spec file."),
    prices: Path = typer.Option(
        ..., "--prices", exists=True, readable=True, help="CSV of historical prices."
    ),
) -> None:
    """Solve the portfolio problem described in ``spec_path`` against ``prices``."""
    try:
        spec = _load_spec(spec_path)
    except Exception as e:
        console.print(f"[red]Spec validation failed:[/red] {e}")
        raise typer.Exit(code=2) from None

    console.print(f"[bold]Spec loaded:[/bold] {spec_path}")
    console.print(
        f"  universe = {len(spec.universe)} tickers · "
        f"objective = [bold]{spec.objective.kind}[/bold] · "
        f"problem class = [bold]{spec.problem_class}[/bold]"
    )

    price_df = _load_prices(prices, spec.universe)
    mu, sigma = estimate_moments(price_df)

    compiled = compile_spec(spec, mu, sigma)
    start = time.perf_counter()
    try:
        compiled.problem.solve(solver=cp.CLARABEL)
    except cp.SolverError as e:
        raise SolverError(f"Clarabel failed: {e}") from e
    elapsed_ms = 1000.0 * (time.perf_counter() - start)

    status = compiled.problem.status
    if status in {"infeasible", "infeasible_inaccurate"}:
        console.print(f"[red]Problem is infeasible[/red] (status: {status}).")
        raise typer.Exit(code=3)
    if status in {"unbounded", "unbounded_inaccurate"}:
        console.print(f"[red]Problem is unbounded[/red] (status: {status}).")
        raise typer.Exit(code=4)

    weights = list(compiled.weights.value)
    console.print(_render_weights(spec, weights))

    console.print(
        f"\n[bold]Objective value:[/bold] {compiled.problem.value:.6f}"
        f"  ·  [bold]Solver:[/bold] Clarabel"
        f"  ·  [bold]Status:[/bold] {status}"
        f"  ·  [bold]Time:[/bold] {elapsed_ms:.1f} ms"
    )

    try:
        duals = harvest_duals(compiled)
    except (InfeasibleError, UnboundedError, TruffleError) as e:
        console.print(f"[yellow]Duals unavailable:[/yellow] {e}")
        return

    console.print(_render_duals(spec, duals))


if __name__ == "__main__":
    app()
