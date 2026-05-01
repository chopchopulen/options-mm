# Options Market Making Simulator

A production-quality options market making simulator in Python. The system quotes a multi-strike, multi-expiry options book on a Heston stochastic-volatility underlying, models adverse selection from informed traders, delta-hedges in real time, and decomposes daily P&L into five economic components — all validated by 55 unit tests.

---

## What This Is

This project simulates how a market maker runs a book of equity options. The MM posts bid and ask prices on 6 options simultaneously, earns the spread when traders hit the quotes, and continuously delta-hedges to stay roughly flat on direction. After 30 trading days, the P&L is decomposed into its economic causes using a first-order Greek approximation.

The simulation is genuinely risky: a Heston stochastic-vol model drives the underlying, meaning implied vol clusters and spikes unpredictably. Informed traders exploit quote staleness whenever the MM's pricing lags the true price — this is the Glosten-Milgrom adverse selection mechanism. The MM can lose money.

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
    analytical.py     ← closed-form Delta, Gamma, Vega, Theta
    numerical.py      ← finite-difference Greeks (cross-validation)
    portfolio.py      ← portfolio-level Greek aggregation
  market/
    underlying.py     ← Heston SV simulator (Euler-Maruyama, full truncation)
    order_flow.py     ← two-population order flow (noise + informed traders)
  mm/
    quoter.py         ← spread widening: f(Gamma, Vega, sigma_uncertainty)
    inventory.py      ← position tracking (options + underlying hedge)
    hedger.py         ← threshold-based delta rebalancing
  risk/
    limits.py         ← per-leg and portfolio Gamma/Vega caps
  pnl/
    attribution.py    ← 5-component P&L decomposition + closure check
  backtest/
    engine.py         ← main simulation loop (30 days × 78 steps)
    report.py         ← summary table + 5-panel matplotlib visualization
    data.py           ← yfinance SPY IV surface loader + Heston calibration

tests/               ← 55 pytest tests (run: pytest tests/ -v)
run_backtest.py      ← entry point
```

---

## Quick Start

```bash
pip install numpy scipy pandas matplotlib pytest yfinance
pytest tests/ -v          # 55 tests, all should pass
python3 run_backtest.py   # 30-day simulation, saves backtest_results.png
```

---

## Sample Output

```
============================================================
OPTIONS MARKET MAKER — BACKTEST SUMMARY
============================================================
  Total P&L:          $  33837.79
  Sharpe Ratio:            1.849
  Win Rate (days):         46.7%
  Max Drawdown:       $  35111.78

  P&L Attribution (cumulative):
    spread_capture         $  22315.22
    theta_pnl              $ -12002.57
    gamma_pnl              $   2006.81
    vega_pnl               $  21597.79
    hedge_cost             $ -29851.80
    residual               $  29772.34
============================================================
```

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

**Informed traders** exploit quote staleness. The MM prices using `S_stale` (price 2 steps ago); the true price is `S_true`. When `|S_true − S_stale| / S_stale > 0.2%`, an informed trader hits the profitable side of the book (buying cheap calls or selling rich puts). This is the Glosten-Milgrom adverse selection mechanism — the MM takes real losses on informed flow.

### 6. Spread Formula

```
half_spread = base_spread + γ_coeff × |Gamma| × contract_size + ν_coeff × |Vega| × σ_uncertainty
```

The MM widens quotes when portfolio Gamma is high (convexity risk) or vol uncertainty is high (pricing risk).

### 7. Delta Hedging

After each step, if `|portfolio_delta| > 25 shares`, the hedger trades the underlying to flatten back to zero. Each hedge trade pays a 0.1% transaction cost. Delta is `option_delta × quantity × 100`.

### 8. Risk Limits

Before quoting each option, three limits gate the quote size:
- **Portfolio Gamma cap**: 800 Gamma units — scale down if near limit
- **Portfolio Vega cap**: 50,000 Vega units — scale down if near limit
- **Per-leg position cap**: 20 contracts per strike/expiry/type — stop quoting at limit

### 9. P&L Attribution

Daily P&L decomposes into five economic components:

| Component | Formula | Meaning |
|-----------|---------|---------|
| `spread_capture` | Σ (ask − fair or fair − bid) × fills × 100 | Revenue from liquidity provision |
| `theta_pnl` | Σ_steps portfolio_theta × dt | Time decay (positive when short options) |
| `gamma_pnl` | Σ_steps ½ × Γ × S² × (σ²_realized − σ²_implied) × dt | Variance differential: profit if realized vol > implied |
| `vega_pnl` | portfolio_vega_EOD × (σ_EOD − σ_SOD) | P&L from shifts in implied vol level |
| `hedge_cost` | −Σ transaction costs | Always ≤ 0 |
| `residual` | mtm_pnl − all of the above | What the first-order model cannot explain |

The accounting identity `components + residual ≡ mtm_pnl` holds to machine precision by construction. The residual (~88% here) is structural — it reflects discrete hedging error, rolling-window IV estimation lagging Heston vol spikes, and second-order cross-Greek effects (vanna, volga) that a first-order Taylor expansion cannot capture. Large residuals are expected and honest in a Heston simulation with 5-minute bars.

MTM P&L is `(EOD book value − SOD book value) + realized P&L from closes + underlying position P&L`, where book value = BS price × quantity × 100.

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total P&L | $33,837.79 |
| Sharpe Ratio (annualized) | 1.849 |
| Win Rate | 46.7% of days |
| Max Drawdown | $35,111.78 |

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
| Analytical Greeks | 8 | Delta range, delta sum, gamma symmetry, theta sign, ATM delta |
| Numerical Greeks | 4 | FD agrees with analytical to 4 decimal places |
| Portfolio Greeks | 3 | Aggregation, linearity, empty book |
| Heston Simulator | 4 | Output shape, variance positivity, price positivity, seed independence |
| Order Flow | 4 | No informed when prices equal, correct side, fields, size ordering |
| Quoter | 4 | bid < ask, symmetric, widens with gamma, widens with vol uncertainty |
| Inventory | 4 | Fill direction, buy/sell, underlying tracking, realized P&L on close |
| Hedger | 4 | Below threshold, above threshold, negative delta, inventory update |
| Risk Limits | 4 | Full size, gamma scale-down, position cap, at-limit zero |
| P&L Attributor | 4 | Fields present, identity holds to 1e-10, short-call theta sign, zeros |
| Backtest Engine | 3 | Smoke test, attribution identity each day, total sums daily |

```bash
pytest tests/ -v   # all 55 pass
```
