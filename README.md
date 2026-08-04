# Options Market Making Simulator

> **⚠️ READ `docs/FINAL_NUMBERS.md` BEFORE QUOTING ANY NUMBER FROM THIS REPO.**
> A blind audit found 21 defects, including a P&L identity that overstated returns by 48% and a
> load-bearing test that asserted an accounting identity against its own definition. Much of the
> prose below predates that audit. Corrected sections are marked inline; `docs/FINAL_NUMBERS.md`
> is authoritative on what is claimable, what is retired, and what remains open.

A production-quality options market making simulator in Python. The system quotes a multi-strike, multi-expiry options book on a Heston stochastic-volatility underlying, models adverse selection from informed traders, delta-hedges in real time, and decomposes daily P&L into eight economic components — all validated by 88 unit tests.

---

## What This Is

This project simulates how a market maker runs a book of equity options. The MM posts bid and ask prices on 6 options simultaneously, earns the spread when traders hit the quotes, and continuously delta-hedges to stay roughly flat on direction. After 30 trading days, the P&L is decomposed into its economic causes using a second-order Greek expansion plus an explicit adverse-selection term.

The simulation is genuinely risky: a Heston stochastic-vol model drives the underlying, meaning implied vol clusters and spikes unpredictably. Informed traders trade against the MM whenever their edge exceeds the half-spread, so the MM is adversely selected and can lose money. The spread carries a Glosten-Milgrom **break-even charge**, derived from the informed share and the expected informed move. The quoter itself is **not** a Glosten-Milgrom maker: it has no informed-arrival probability, no conditional expectation given order direction, and no Bayesian belief update — its mid is always the stale fair regardless of what just traded. It is a symmetric risk-loaded quoter. See `audit/FINDINGS.md` finding 13.

---

## Architecture

```
configs/
  default.py          ← single source of truth for all parameters

src/
  pricing/
    black_scholes.py  ← analytical BS pricer (calls & puts)
    binomial.py       ← CRR binomial tree pricer
    monte_carlo.py    ← antithetic-variates MC pricer
  greeks/
    analytical.py     ← closed-form Delta, Gamma, Vega, Theta, Vanna, Volga
    numerical.py      ← finite-difference Greeks (cross-validation)
    portfolio.py      ← portfolio-level Greek aggregation
  market/
    underlying.py     ← Heston SV simulator (Euler-Maruyama; max(v,0) in drift and
                        diffusion but max(v+dv,1e-8) carried — absorption, not textbook
                        full truncation. Immaterial here: Feller is satisfied)
    order_flow.py     ← two-population order flow (noise + informed); optional
                        counterparty reservation price, OFF by default
  mm/
    quoter.py         ← half-spread = base + gamma carry + vega carry + GM adverse selection
    inventory.py      ← position tracking (options + underlying hedge)
    hedger.py         ← threshold-based delta rebalancing
  risk/
    limits.py         ← directional per-leg and portfolio Gamma/Vega caps (bid/ask sized
                        separately so the book can always quote its way toward flat)
  pnl/
    attribution.py    ← DEAD CODE. The engine inlines its own attribution and never
                        imports this module (FINAL_NUMBERS.md OPEN #11)
  backtest/
    engine.py         ← main simulation loop (30 days × 78 steps)
    report.py         ← summary table + 5-panel matplotlib visualization
    data.py           ← yfinance SPY IV surface loader + Heston calibration
    multi_seed.py     ← 20-seed aggregate statistics → results/multi_seed.csv
    sensitivity.py    ← 27-combo parameter grid search → results/sensitivity.csv

tests/               ← 88 pytest tests (run: python3 -m pytest -q)
run_backtest.py      ← entry point
```

---

## Quick Start

```bash
pip install numpy scipy pandas matplotlib pytest yfinance
python3 -m pytest -q                   # 88 tests, all should pass
python3 run_backtest.py                # 30-day simulation, saves backtest_results.png
python3 src/backtest/multi_seed.py     # 20-seed stats, saves results/multi_seed.csv
python3 src/backtest/sensitivity.py    # parameter grid search, saves results/sensitivity.csv
```

---

## Sample Output

```
============================================================
OPTIONS MARKET MAKER — BACKTEST SUMMARY
============================================================
  Total P&L:          $ 255353.47
  P&L signal/noise:        9.315   (NOT a Sharpe ratio — no capital base)
  Win Rate (days):         70.0%
  Max Drawdown:       $  18441.33

  P&L Attribution (cumulative):
    spread_capture         $1256938.06
    adverse_selection      $-421284.69
    ...
============================================================
```

> **⚠️ The P&L LEVEL above is not a result.** With noise flow inelastic to quote width, P&L rises
> monotonically with the spread we choose — the level is not identified without a model of
> competition. See `docs/FINAL_NUMBERS.md`. The previous sample output in this README showed
> **Total P&L $33,837.79 / Sharpe 1.849 / residual 21.67%**; all three are retired. That P&L was
> overstated by 48% by an accounting defect, 89% of it was a volatility-initialization artifact,
> and "Sharpe" is not a Sharpe ratio.

The simulation saves `backtest_results.png` with five panels: cumulative P&L, daily attribution stacked bar, Heston price path, spread capture vs. hedge cost, and Gamma vs. Theta P&L.

---

## How It Works

### 1. Underlying: Heston Stochastic Volatility

The stock price follows the Heston model — two coupled SDEs with a mean-reverting variance process:

```
dS = r·S·dt + √v·S·dW_S
dv = κ(θ − v)dt + ξ·√v·dW_v
corr(dW_S, dW_v) = ρ = −0.7   (negative: vol spikes when price drops)
```

Parameters: `S₀=450`, `v₀=0.04` (20% vol), `κ=2` (mean reversion), `θ=0.04` (long-run var), `ξ=0.3` (vol-of-vol), `ρ=−0.7` (leverage effect). Full truncation (`v⁺ = max(v, 0)`) prevents negative variance in discrete simulation.

### 2. Option Universe

The MM quotes 6 options simultaneously:

| Strike | Expiry | Type |
|--------|--------|------|
| −5% OTM | 30 days | put |
| ATM | 30 days | call |
| ATM | 30 days | put |
| +5% OTM | 30 days | call |
| −5% OTM | 60 days | put |
| ATM | 60 days | call |

### 3. Pricing

Three independent pricers, all converging to the same answer — used for cross-validation:

- **Black-Scholes**: closed-form, used for live quoting (instant)
- **CRR Binomial tree**: 500-step recursive tree, validated against BS (within $0.05)
- **Monte Carlo**: 50,000 paths with antithetic variates (pairs each path with its mirror to halve variance), validated against BS (within $0.10)

The MM prices using BS with `σ_implied` estimated from a rolling 10-step log-return window.

### 4. Greeks Engine

Closed-form analytical Greeks (Delta, Gamma, Vega, Theta) cross-validated against finite-difference approximations to 4 decimal places. Portfolio Greeks aggregate across all positions weighted by `quantity × contract_size`. A short position flips the sign — a short call earns positive theta.

### 5. Order Flow: Adverse Selection

Two populations of counterparties arrive each 5-minute step:

**Noise traders** arrive Poisson(λ=8/day × dt) with random direction and size 1–5 contracts. They provide the spread revenue.

**Informed traders** exploit quote staleness. The MM prices using `S_stale` (price 2 steps ago); the true price is `S_true`. When `|S_true − S_stale| / S_stale > 0.2%`, an informed trader hits the profitable side of the book *for that option type* — buying calls and selling puts when S rises, and the reverse when S falls. (Before the audit the side was chosen from the sign of the underlying move alone, which is correct for calls and backwards for puts.)

This is **not** the Glosten-Milgrom mechanism, despite earlier claims in this document. In Glosten-Milgrom the maker sets bid and ask each equal to the conditional expectation of value given the order direction, so adverse selection is priced into the quote and beliefs update after every trade. `Quoter` has no informed-arrival probability, no conditional expectation, and no Bayesian update; its mid is always the stale fair regardless of what just traded. It is a symmetric risk-loaded quoter. See `audit/FINDINGS.md` finding 13.

### 6. Spread Formula

```
half_spread = base_spread
            + 0.5 × |Gamma| × S² × σ² × τ          (gamma carry over the holding horizon)
            + |Vega| × (ξ/2) × √τ                  (vol move over τ; from dv = ξ√v dW)
            + |Delta| × S × E[|m| | informed] × informed_share   (Glosten-Milgrom break-even)
```

Every term is the cost of something specific, and **no coefficient is fitted**. τ is the
inventory holding horizon, derived from the arrival rate as `steps_per_day / λ` — not the hedge
interval, since delta hedging removes delta risk only while gamma and vega persist for the life
of the position. The adverse-selection term is the Glosten-Milgrom break-even charge: the maker
pays the informed trader's edge on informed volume and collects the half-spread on all volume.

This yields **1.13%–3.17% of premium** across the six legs. The previous formula —
`base + γ_coeff × |Gamma| × contract_size + ν_coeff × |Vega| × σ_uncertainty` — built a dollar
charge from a dimensionless second derivative with no horizon or position size in it, and
produced half-spreads of **10–51% of premium**, of which 97%+ came from that one term
(`docs/FINAL_NUMBERS.md` defect #9).

### 7. Delta Hedging

After each step, if `|portfolio_delta| > 25 shares`, the hedger trades the underlying to flatten back to zero. Each hedge trade pays a 0.1% transaction cost. Delta is `option_delta × quantity × 100`.

> **⚠️ The 0.1% (10bps) transaction cost is roughly 20–100× realistic.** Institutional equity
> execution in a name like SPY is low single-digit basis points all-in. This assumption accounts
> for **−$525,702** of the median P&L. It is documented rather than corrected — see
> `docs/FINAL_NUMBERS.md` defect #21 and OPEN #4 for why. The full-flatten policy (no dead band)
> also drives gratuitous turnover: hedging fires on roughly half of all 5-minute bars.

### 8. Risk Limits

Before quoting each option, three limits gate the quote size:
- **Portfolio Gamma cap**: 124 Gamma units — scale down the side that *increases* exposure
- **Portfolio Vega cap**: 756,000 Vega units — same, directionally
- **Per-leg position cap**: 20 contracts per strike/expiry/type — blocks only the increasing side

The Greek caps are **derived from** the position cap, not chosen independently: 20 contracts ×
6 legs is a declared capacity of 120 contracts, which carries 755,957 vega and 123.7 gamma at
σ=0.20. The previous values (800 / 50,000) contradicted the position limit and each other by
~100×: vega bound at 6.6% of declared capacity while gamma bound at 646% of it.

All three limits are **directional**. Sizing on `abs(position)` refused the inventory-reducing
side as hard as the increasing side, so a book at the cap froze and could only exit via expiry
(`docs/FINAL_NUMBERS.md` defect #7).

### 9. P&L Attribution

Daily P&L decomposes into five economic components:

| Component | Formula | Meaning |
|-----------|---------|---------|
| `spread_capture` | Σ (ask − fair_stale or fair_stale − bid) × fills × 100 | **Gross quoted** spread, measured at the quote-time mark |
| `adverse_selection` | Σ (fair_now − fair_stale) × signed_qty × 100 | Cost of quoting off a stale mark. **Sums with `spread_capture` to give the edge against the contemporaneous fair** — read the two together, never `spread_capture` alone |
| `theta_pnl` | Σ_steps portfolio_theta × dt | Time decay (positive when short options) |
| `gamma_pnl` | Σ_steps ½ × Γ × S² × (σ²_realized − σ²_implied) × dt | Variance differential: profit if realized vol > implied |
| `vega_pnl` | portfolio_vega_EOD × (σ_EOD − σ_SOD) | P&L from shifts in implied vol level |
| `vanna_pnl` | −portfolio_vanna_EOD × ΔS × Δσ | Cross-gamma: P&L from joint spot-vol moves |
| `volga_pnl` | −½ × portfolio_volga_EOD × Δσ² | Vol convexity: P&L from curvature in vol |
| `hedge_cost` | −Σ transaction costs | Always ≤ 0 |
| `residual` | mtm_pnl − all of the above | Higher-order terms. **A plug** — it closes by definition, so a bug anywhere lands here silently. Currently 2.50% of gross flow |

The second-order terms carry a **minus** sign because `port_eod` Greeks are evaluated at σ_EOD, the endpoint of the move, so the Taylor expansion runs backwards from it. Using `+½volga` made the approximation worse on every move tested — vol terms totalled $44,038 against an exact reprice of $2,966.70 (`docs/FINAL_NUMBERS.md` defect #8).

The accounting identity `components + residual ≡ mtm_pnl` holds to machine precision. After the second-order vol terms and the adverse-selection term are included, the residual is **2.50% of gross flow** (median, seeds 0–19, ex day 0; per-seed range 0.05%–6.90%), verified closing on all 20 seeds in both configurations. The remaining residual reflects third-order effects and discrete-hedging approximation error.

Residuals are quoted against **gross** flow, not net. The net denominator is unstable — seed 1 shows 2264% of net against 4.00% of gross — so any residual figure stated as a percentage of net is uninterpretable.

**Why vega/vanna/volga use EOD net moves, not intraday accumulation:**

Theta and gamma are accumulated step-by-step because their P&L depends on the realized price path (actual log-returns and variance). Vega, vanna, and volga use the net SOD→EOD sigma change only. The reason: `sigma_implied` is estimated from a 10-step rolling log-return window. With only 10 samples, step-to-step changes in this estimator are large and noisy — they reflect sampling variance in a tiny window, not actual changes in implied vol. Summing `½ × volga × Δσ_step²` across 78 steps accumulates estimator noise squared, producing a spurious term (~10× the daily MTM P&L) with no relation to the actual book mark. The MTM book is repriced only at SOD and EOD, so only the net `σ_EOD − σ_SOD` move propagates to realized P&L. The intraday oscillations cancel in the book mark.

The claim that once stood here — that the 21.67% residual was "a floor given this vol estimator, not an implementation gap" — was wrong, and the audit refuted it. Two implementation defects were inflating the attribution in opposite directions and partly cancelling: the P&L identity substituted an inception-to-date realized gain for a cash flow, and the second-order vol terms carried the wrong sign for Greeks evaluated at the endpoint σ. Measured against an exact reprice of the EOD book, the old expansion erred by $41,071 against a true vol revaluation of $2,967. Both are fixed; see `audit/FINDINGS.md`.

MTM P&L is `(EOD book value − SOD book value) + net cash flow from fills + underlying position P&L`, where book value = BS price × quantity × 100. The middle term must be a **cash flow** — substituting realized P&L omits opening premium entirely.

---

## Performance Metrics

**There is no P&L level to report, and that is the terminal finding of this project.**

With no competition model, the simulator's P&L level is not identified. Neither configuration
yields a claimable performance number:

| configuration | median P&L | median Sharpe\* | why it is not claimable |
|---|---:|---:|---|
| `use_reservation_price=False` (repo default) | +$290,930 | +9.752 | Unbounded in quote width. Noise flow is perfectly inelastic, so P&L rises monotonically with width — the level is a function of a width *we chose*, not of anything the model determines. |
| `use_reservation_price=True` (ITEM 13) | −$1,132,830 | −42.342 | Artifact of the mis-anchored reservation scale: p_fill ≈ 1.33% at the derived spread, screening out 99% of benign flow, because the scale is anchored to the MM's own inventory horizon τ (circular — λ sets τ sets the scale). |

The default is `False` because a repo default must not be a configuration already identified as
an artifact — **not** because it is more trustworthy. Unbounded is a different defect from
wrong, not a smaller one. Do not quote either P&L figure, either Sharpe\*, or the positive-seed
counts (19/20 and 0/20) as a performance result.

### \* Every Sharpe figure in this project carries this caveat

**There is no capital base anywhere in this repo.** `compute_sharpe`
(`src/backtest/report.py:8-13`) computes `sqrt(252) × mean(daily DOLLAR P&L) /
population-std(daily DOLLAR P&L)` — a **signal-to-noise ratio of a dollar P&L stream**, not a
risk-adjusted return. It is invariant to leverage and book size: double every position and it
does not move. It also subtracts a daily rate from a dollar series (dimensionally incoherent,
worth 1.3e-07) and uses `ddof=0`. A percentage drawdown is undefinable here for the same reason.
`grep -rniE "capital|equity|nav|aum|account_value|initial_cash|book_size"` returns zero hits in
`src/`, `configs/` or `bench/`.

### What IS claimable

| claim | value | measured on |
|---|---|---|
| The attribution identity closes | residual median **2.50%** of gross flow; per-seed range 0.05%–6.90% | seeds 0–19, ex day 0, closing on **all 20 seeds** in both configurations |
| The backtest is bit-reproducible | byte-identical daily P&L, attribution and price path across separate processes | seed 42, re-verified after every commit |
| Derived adverse-selection share matches the measured one, independently | derived **0.8868** from Heston dynamics + arrival model; measured **~89%** from fill counts | two independent routes, no shared inputs |
| Fill probability was completely independent of quote width | **5,335 fills / 19,388 contracts at every width** across a 16× sweep | seed 42 |
| The half-spread derivation carries no fitted coefficient | 1.13%–3.17% of premium across the six legs | analytic, at σ=0.20 |

### Multi-Seed Dispersion (20 seeds, default params)

```bash
python3 src/backtest/multi_seed.py   # saves results/multi_seed.csv
```

Reported as **dispersion only** — the levels inherit the identification problem above.

| Metric | Value |
|--------|-------|
| Mean Sharpe\* | −0.144 |
| Std Sharpe\* | 3.072 |

Any single-seed comparison is noise: with a 20-seed mean of −0.14 against a std of 3.07, a ±1
move on one seed carries no information. The 30-day window is short enough that path-dependent
vol clustering dominates parameter skill.

Full breakdown of what can and cannot be claimed: [`docs/FINAL_NUMBERS.md`](docs/FINAL_NUMBERS.md).

---

## Configuration

All parameters are in [configs/default.py](configs/default.py). Nothing is hardcoded in the simulation:

```python
HESTON = dict(S0=450.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.02)

OPTION_UNIVERSE = [
    (-0.05, 30, "put"), (0.00, 30, "call"), (0.00, 30, "put"),
    (0.05, 30, "call"), (-0.05, 60, "put"), (0.00, 60, "call"),
]

BACKTEST = dict(
    n_days=30, steps_per_day=78,        # 5-minute bars in a 6.5-hour day
    sigma_uncertainty_window=10,         # rolling log-return window for IV
    quote_staleness_steps=2,             # MM sees price 2 steps late
    default_sigma=0.20, risk_free_rate=0.02, desired_quote_size=5,
)
```

---

## Tests

55 pytest tests covering every layer:

| Module | Tests | What's verified |
|--------|-------|----------------|
| Black-Scholes | 5 | Put-call parity, known ATM value, deep ITM, expiry, invalid type |
| Binomial | 2 | Convergence to BS at 500 steps, put-call parity |
| Monte Carlo | 2 | Accuracy within $0.10, antithetic variance reduction |
| Analytical Greeks | 8 | Delta range, delta sum, gamma symmetry, theta sign, ATM delta; vanna/volga sign and zero-at-expiry |
| Numerical Greeks | 6 | FD agrees with analytical to 4 decimal places for all 6 Greeks including vanna/volga |
| Portfolio Greeks | 3 | Aggregation, linearity, empty book; includes vanna/volga keys |
| Heston Simulator | 4 | Output shape, variance positivity, price positivity, seed independence |
| Order Flow | 4 | No informed when prices equal, correct side, fields, size ordering |
| Quoter | 4 | bid < ask, symmetric, widens with gamma, widens with vol uncertainty |
| Inventory | 4 | Fill direction, buy/sell, underlying tracking, realized P&L on close |
| Hedger | 4 | Below threshold, above threshold, negative delta, inventory update |
| Risk Limits | 4 | Full size, gamma scale-down, position cap, at-limit zero |
| P&L Attributor | 5 | Fields present, identity holds to 1e-10, short-call theta sign, zeros, non-zero vanna/volga pass-through |
| Backtest Engine | 3 | Smoke test, attribution identity each day (includes vanna/volga), total sums daily |
| Sensitivity | 3 | CSV created with correct columns, returns DataFrame, best combo has valid floats |

```bash
pytest tests/ -v   # all 66 pass
```

---

## Parameter Sensitivity Analysis

> **⚠️ RETIRED RESULT. The ranking carries no demonstrated information.**
>
> The grid searches 27 combinations against 5 shared seeds and prints a "Best combo". Under
> the null that every combination has identical true performance, `E[max of 27]` exceeds the
> truth by **+1.319** (ρ=0) to **+0.591** (ρ=0.8) given the measured per-combo standard error
> of 0.661. The observed best-minus-mean was **+0.670**, and the across-combo standard
> deviation (0.4913) is *smaller* than the single-combo standard error (0.6606).
>
> **The entire observed spread from best to worst is consistent with no effect at all.** Acting
> on the ranking would import roughly +0.7 to +1.3 of apparent performance that does not exist,
> validated against a benchmark whose own standard error is ±2.9. Do not quote the "Best combo"
> line, and do not tune parameters to it. See `docs/FINAL_NUMBERS.md` defect #18.
>
> `results/sensitivity.csv` is a pre-audit artifact: its ranking is retired and its column names
> predate the `sharpe` → `pnl_snr` rename. Kept for provenance only.

The grid remains useful as a *robustness* check — showing that results do not depend on a
knife-edge parameter choice — not as an optimizer.

```bash
python3 src/backtest/sensitivity.py   # ~25 min, saves results/sensitivity.csv
```

Grid:
- `hedge_threshold`: [10, 25, 50] shares — how aggressively to delta-hedge
- `base_spread_bps`: [10, 20, 50] bps — minimum quote width
- `informed_threshold`: [0.001, 0.002, 0.005] — staleness gap before informed traders arrive

Each combo is averaged across 5 random seeds. Ranked output is printed but must not be read as
a recommendation.

---

## What This Work Actually Demonstrates

> Audited a stochastic options market-making simulator and found 21 defects that a green
> 70-test suite had not caught, including a P&L identity that overstated returns by 48%, a
> volatility-initialization artifact responsible for 89% of reported P&L, and a load-bearing
> test that asserted an accounting identity against its own definition. Rebuilt the P&L
> attribution to close at 2.5% of gross flow, derived the quoted spread from inventory carry
> cost and a Glosten-Milgrom break-even condition with no fitted coefficients, and established
> that the simulator's P&L level is not identified without a model of competition.

Superseded summary claims from earlier revisions are recorded, with their corrections, in the
RETIRED section of `docs/FINAL_NUMBERS.md`.
