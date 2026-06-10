# Project Blueprint: Natural-Language Portfolio Optimization Agent

Working names to consider: `convexa`, `folioagent`, `allocd`. Pick something short and pip-able.

**One-line pitch:** Describe your portfolio problem in plain English; get a rigorously formulated, solved, backtested, and *explained* optimization — including which of your constraints cost you money and why.

---

## 1. Thesis and Novelty Claim

Free-form "LLM writes optimization code" fails because LLMs hallucinate constraints, produce non-convex formulations, and can't diagnose infeasibility. This project's core claim:

> **In a bounded domain, NL→optimization becomes reliable if the LLM only ever emits a typed intermediate representation (IR), and a deterministic compiler owns the mathematics.**

Four pillars, each individually defensible, together novel as an open-source system:

1. **Typed IR for portfolio problems.** A JSON schema (Pydantic models) covering ~95% of practitioner portfolio constraints. The LLM's only job is NL → IR.
2. **Deterministic compiler.** IR → CVXPY model. Hand-written, tested, provably convex (or declared MIP). Zero LLM involvement.
3. **Grounded explanation layer.** Dual variables, shadow prices, and sensitivity analysis are computed by the solver, then narrated by the LLM — which is only allowed to reference numbers that exist in the solver output. "Your 25% tech cap has a shadow price of 14bps of expected annual return."
4. **Automated infeasibility repair.** When constraints conflict, an elastic relaxation pass identifies the minimal conflicting set and the agent explains the conflict and proposes fixes in plain English. (This is a real OR research area — IIS computation — and almost nobody has shipped it with an NL front end.)

This is also a publishable angle if you want it later: "domain-restricted NL4Opt with verified compilation" — eval harness included (Section 9).

---

## 2. User Experience (what the demo looks like)

```
> I have $50k across these 25 tickers. Long-only. No single stock above 8%,
  tech sector capped at 30%. I care more about downside risk than variance —
  minimize CVaR at 95%. Keep me within 15 names max. Rebalance monthly,
  assume 10bps transaction costs.

[Agent] Here's how I interpreted your problem — confirm before I solve:

  Objective:    Minimize CVaR (α = 0.95) of portfolio returns
  Universe:     25 tickers (listed)
  Constraints:  fully invested (Σw = 1) · long-only (w ≥ 0)
                position cap w_i ≤ 0.08 · sector cap: Tech ≤ 0.30
                cardinality ≤ 15 names (this makes it a MIP)
  Backtest:     monthly rebalance, 10bps proportional costs, walk-forward

  Note: cardinality + CVaR means a mixed-integer LP. Solve time ~seconds
  at this universe size. Proceed?

> yes

[Agent] Solved (HiGHS, 2.3s, optimality gap 0.0%).
  → 13 names selected. Portfolio CVaR₉₅: -2.1% monthly.
  → Binding constraints: tech sector cap (shadow price ≈ 11bps/yr),
    position cap on NVDA.
  → If you relaxed the tech cap to 35%, expected CVaR improves ~0.15%.
  [allocation chart] [backtest tearsheet: Sharpe 1.1, MaxDD -14%, turnover 22%/yr]
```

The **spec echo** step (confirm interpretation before solving) is non-negotiable. It's the trust mechanism, the thing that makes this feel like working with a quant analyst rather than a slot machine.

---

## 3. System Architecture

```
 ┌────────────┐   NL    ┌─────────────┐  IR(JSON) ┌──────────────┐
 │   User     │ ──────► │ Agent Layer │ ────────► │  Validator   │
 │ (CLI/Web)  │ ◄────── │  (Claude +  │ ◄──────── │  (Pydantic + │
 └────────────┘ explain │  tool use)  │  errors   │  semantics)  │
                        └──────┬──────┘           └──────┬───────┘
                               │ tools                   │ valid IR
                ┌──────────────┼──────────────┐          ▼
                ▼              ▼              ▼   ┌──────────────┐
        ┌──────────────┐ ┌───────────┐ ┌────────┐│   Compiler   │
        │ Infeasibility│ │ Backtest  │ │ Explain││  IR → CVXPY  │
        │  Diagnoser   │ │  Engine   │ │ (duals)│└──────┬───────┘
        └──────────────┘ └───────────┘ └────────┘       ▼
                ▲              ▲              ▲   ┌──────────────┐
                └──────────────┴──────────────┴── │ Solver Layer │
                                                  │ HiGHS/Clarabel│
                                                  └──────┬───────┘
                                                         ▼
                                                  ┌──────────────┐
                                                  │  Data Layer  │
                                                  │ prices→μ,Σ,  │
                                                  │  scenarios   │
                                                  └──────────────┘
```

Key boundary: **everything below the Agent Layer is deterministic, typed, and unit-tested.** The LLM touches only (a) NL → IR, (b) clarifying questions, (c) narrating numbers the solver produced.

---

## 4. The Intermediate Representation (the heart of the project)

Pydantic schema, roughly:

```python
class PortfolioSpec(BaseModel):
    universe: list[str]                      # tickers
    capital: float | None
    objective: Objective                     # discriminated union
    constraints: list[Constraint]            # discriminated union
    estimation: EstimationConfig             # lookback, shrinkage, scenario gen
    backtest: BacktestConfig | None
    rebalance: RebalanceConfig | None

class Objective(BaseModel):
    kind: Literal["min_variance", "mean_variance", "max_sharpe",
                  "min_cvar", "risk_parity", "min_tracking_error"]
    params: dict          # e.g. risk_aversion λ, cvar_alpha, benchmark

# Constraint examples (each its own typed model):
#  Budget(total=1.0) · LongOnly() · Box(lower, upper, tickers?)
#  GroupCap(group="Technology", max_weight=0.30, mapping=...)
#  Cardinality(max_names=15) · TurnoverCap(max_turnover=0.25)
#  TransactionCost(bps=10) · FactorExposure(factor, min, max)
#  CVaRLimit(alpha, max_cvar) · TrackingErrorCap(benchmark, max_te)
```

Design rules:
- Every constraint type declares whether it preserves convexity or forces MIP. The compiler aggregates this and tells the user what problem class they've built.
- The schema *is* the product roadmap: v0.1 ships ~8 constraint types; growth = adding types.
- Semantic validation beyond types: caps that sum to < 1 with full-investment, cardinality > universe size, conflicting box bounds — catch these *before* the solver does, with named errors the agent can relay.

---

## 5. The Modeling Core (your OR education, operationalized)

Each objective is a small, separately tested CVXPY builder. The math you'll implement and therefore actually learn:

**Mean-variance (Markowitz).** `min wᵀΣw − λμᵀw`. QP. The hello-world; also where you implement covariance estimation properly — **Ledoit–Wolf shrinkage**, not sample covariance (sample Σ is garbage when N assets ≈ T observations; this detail alone signals you know what you're doing).

**Min-CVaR (Rockafellar–Uryasev).** The crown jewel and your differentiation from PyPortfolioOpt-style libraries. CVaR at level α over S return scenarios linearizes to an LP:

```
min  t + (1/(1−α)S) Σₛ zₛ
s.t. zₛ ≥ −rₛᵀw − t,  zₛ ≥ 0   ∀s
```

Scenario generation lives in the data layer: historical, IID bootstrap, and block bootstrap (preserves autocorrelation — mention this in the README, it's a sophistication marker).

**Max-Sharpe.** Non-convex as stated; implement via the standard variable-substitution transform to a convex program. Document the transform in the code — recruiters and professors read code comments.

**Cardinality / position limits.** Big-M MIP with binary selection variables `yᵢ ∈ {0,1}`, `w ≤ M·y`, `Σy ≤ K`. HiGHS handles this fine at ≤ 100 assets. Enforce universe-size guardrails so demo solve times stay in seconds.

**Risk parity.** Different solution technique (fixed-point or convex reformulation) — good "breadth" exhibit.

**Transaction costs & turnover.** Proportional costs as a linear penalty `κ‖w − w_prev‖₁` (linearized with auxiliary vars). This is the bridge toward multi-period later.

**Duals everywhere.** After every solve, harvest dual values on every named constraint and map them back to IR constraint IDs. This mapping (constraint ID → shadow price → NL name) is what powers the explanation layer and it's the kind of plumbing nobody else bothers to build.

---

## 6. Agent Layer (Claude + tool use)

A thin orchestration loop using the Anthropic SDK with tools:

- `parse_request(nl) → PortfolioSpec` — the main extraction call. Use structured output against the Pydantic JSON schema. Temperature low. Include few-shot examples covering tricky phrasings ("I don't want to be too concentrated" → ask a clarifying question, don't guess a number).
- `validate(spec) → ok | errors` — deterministic.
- `solve(spec) → SolutionReport` — weights, objective value, duals, problem class, solver stats.
- `diagnose(spec) → ConflictReport` — elastic relaxation (Section 7).
- `backtest(spec) → Tearsheet` — Section 8.
- `explain(solution) → grounded narration` — system prompt hard rule: *every number in the explanation must appear in the SolutionReport.* Add a post-hoc check that regex-extracts numerals from the explanation and verifies membership; flag and retry otherwise. (Ship this check — it's a great "we take hallucination seriously" README section.)

Agent conversational policy: ambiguity → ask exactly one clarifying question; vague risk language → propose a concrete interpretation and confirm; always spec-echo before first solve; never silently change the spec.

---

## 7. Infeasibility Diagnosis (the feature that gets you noticed)

When the model is infeasible, re-solve an **elastic version**: add slack variables to every (soft-able) constraint with weighted penalties, minimize total weighted violation. Constraints with positive slack in the elastic optimum form the conflict set. Then the agent says:

> "Your constraints conflict: a 30% tech cap is impossible because 6 of your 25 tickers are tech and your 15-name cardinality + 8% position caps force ≥ 34% tech exposure. Options: raise the tech cap to 34%, raise position caps to 10%, or drop cardinality to 12. Want me to try one?"

Deterministic conflict detection, LLM narration, actionable repair options. No open-source portfolio tool does this. It is also genuinely educational for you — elastic programming and IIS are real OR techniques you'll discuss credibly with Purdue faculty.

---

## 8. Backtest Engine

Walk-forward, no lookahead: at each rebalance date, estimate μ/Σ/scenarios from trailing window only, solve, hold, account for transaction costs on the weight change. Output a tearsheet: cumulative return vs equal-weight and SPY baselines, annualized vol, Sharpe, max drawdown, realized CVaR vs the optimized CVaR (honest model-vs-reality comparison — put this chart in the README), turnover, cost drag.

Scope guard: this is a *decision-support and research tool*, not an alpha engine. Say so in the README disclaimer. Expected returns are the weakest input in all of portfolio optimization; default the agent toward min-risk objectives unless the user supplies views.

Data: `yfinance` for v0 (free, fine for demo). Abstract behind a `DataProvider` interface so Polygon/Tiingo can slot in later.

---

## 9. Eval Harness (what makes it research-grade)

Build a benchmark of **75–100 NL prompts with hand-verified ground-truth IR**, spanning: simple specs, multi-constraint specs, ambiguous phrasings (correct answer = clarifying question), adversarial phrasings ("at least 8%... wait no, at most"), and deliberately infeasible specs. Metrics: exact-match IR accuracy, per-constraint precision/recall, clarification-when-appropriate rate. Run on every commit (cheap — it's ~100 short API calls).

This turns "I made an LLM wrapper" into "I built and measured a verified NL→optimization pipeline; parse accuracy is 94% on a public benchmark." It's also a blog post, a launch asset, and possibly a workshop paper.

---

## 10. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | ecosystem |
| Modeling | CVXPY | declarative, dual extraction, solver-agnostic |
| Solvers | Clarabel (conic/QP), HiGHS (LP/MIP) | free, fast, no license friction for users |
| Schema | Pydantic v2 | IR validation + JSON schema for structured output |
| Agent | Anthropic SDK, tool use | you live in this ecosystem already |
| Data | yfinance behind DataProvider interface | free now, swappable later |
| Tests | pytest + the eval harness | credibility |
| CLI | Typer + Rich | pretty terminal output for the demo GIF |
| Web demo | Streamlit v1 → Next.js + FastAPI v2 | Streamlit ships in a day; Next.js is your home turf for the polished version |
| CI | GitHub Actions: tests, lint, eval | green badges matter |

## 11. Repository Layout

```
convexa/
├── core/
│   ├── ir.py              # Pydantic IR schema
│   ├── compiler.py        # IR → CVXPY
│   ├── objectives/        # markowitz.py, cvar.py, risk_parity.py, ...
│   ├── constraints/       # one module per constraint type
│   ├── diagnose.py        # elastic relaxation / conflict sets
│   └── duals.py           # constraint-ID → shadow-price mapping
├── data/                  # providers, estimation (ledoit_wolf), scenarios
├── backtest/              # engine.py, tearsheet.py
├── agent/                 # claude_client.py, tools.py, prompts/, grounding.py
├── eval/                  # benchmark.jsonl, run_eval.py, report.py
├── cli.py
├── examples/              # notebooks that double as docs
├── tests/
└── README.md              # GIF, architecture diagram, benchmark numbers, disclaimer
```

## 12. Milestones (built for your burst style — four sprints, each independently shippable)

**Sprint 1 — The Core (no LLM yet).** IR schema, compiler, Markowitz + min-variance + CVaR, budget/long-only/box/group-cap constraints, Ledoit–Wolf, CLI that solves from a YAML spec, duals harvested, tests. *Exit: solve a real 25-ticker CVaR problem from the terminal.* This sprint is also your OR bootcamp — do the math by hand alongside the code.

**Sprint 2 — The Agent.** NL → IR with structured output, validator, spec echo, clarifying-question policy, grounded explain with the numeral check, conversational CLI. *Exit: the Section-2 demo transcript works live.*

**Sprint 3 — Proof.** Backtest engine + tearsheet, cardinality MIP, turnover/costs, infeasibility diagnosis. *Exit: the "constraints conflict, here are your options" moment works.*

**Sprint 4 — Launch.** Eval harness + benchmark numbers, Streamlit demo deployed, README with GIF + architecture diagram, one technical blog post ("Why the LLM never touches the solver"), post to r/algotrading, r/quant, HN Show, optimization Discords. *Exit: public, demoable, measurable.*

Stretch (post-launch, pick by energy): multi-period optimization via stochastic programming on scenario trees (serious OR flex), factor-model risk (Fama-French exposures as constraints), Black-Litterman view blending ("I think NVDA outperforms by 5%" → posterior μ — extremely natural fit for an NL agent), robust optimization (uncertainty sets on μ).

## 13. Risks and Guards

- **LLM extraction errors** → mitigated by IR + validator + spec echo + eval harness; never auto-solve without confirmation.
- **MIP blowup** → universe-size limits, solver time limits, agent warns when a constraint forces MIP.
- **Garbage-in expected returns** → default to risk-based objectives; document loudly.
- **"Is this financial advice?"** → README + agent disclaimer: research/education tool.
- **Scope creep back to horizontal** → the IR is the fence. New capability = new constraint type in the schema, nothing else.

## 14. Resume Translation

Finance: CVaR, shadow prices, backtesting, transaction costs, covariance shrinkage — quant-interview vocabulary you'll have implemented, not just read. ML/AI: structured-output agentic system with grounding checks and a measured eval benchmark — the exact skillset AI-product teams hire for in 2026. SWE: typed schema design, compiler pattern, CI, packaging, deployed demo. OR: convex optimization, LP/QP/MIP, duality, elastic programming, stochastic scenarios — and a concrete artifact to show Purdue professors when asking about research.
