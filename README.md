# Options Market Making Simulator

A production-quality options market making simulator in Python. The system quotes a multi-strike, multi-expiry options book on a Heston stochastic-volatility underlying, models adverse selection from informed traders, delta-hedges in real time, and decomposes daily P&L into five economic components — all validated by 55 unit tests.

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
    underlying.py     ← Heston SV simulator (Euler-Maruyama)
    order_flow.py     ← two-population order flow (noise + informed)
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

tests/               ← 55 pytest tests (run: pytest tests/ -v)
run_backtest.py      ← entry point
```

---

## Quick Start

```bash
pip install numpy scipy pandas matplotlib pytest
pytest tests/ -v          # 55 tests, all should pass
python3 run_backtest.py   # 30-day simulation, saves backtest_results.png
```

---

## Output Explained

The simulation prints a summary table after 30 × 78 = 2,340 steps:

| Field | Meaning |
|---|---|
| **Total P&L** | Net dollar P&L across all 30 trading days |
| **Sharpe Ratio** | Annualised Sharpe of daily P&L (√252 × mean/std) |
| **Win Rate** | Fraction of days with positive P&L |
| **Max Drawdown** | Largest peak-to-trough in cumulative P&L |

### P&L Attribution Components

| Component | Meaning |
|---|---|
| `spread_capture` | Half-spread earned on each fill (bid-ask profit) |
| `theta_pnl` | Time decay collected from the option book |
| `gamma_pnl` | Profit from realized vs. implied variance differential (Γ/2 × S² × (σ²_realized − σ²_implied) × dt) |
| `vega_pnl` | P&L from changes in implied volatility (Vega × Δσ) |
| `hedge_cost` | Transaction costs paid on delta-hedging trades (always ≤ 0) |
| `residual` | `MTM − (spread + θ + Γ + ν + hedge)` — always closes to zero by construction; nonzero components reflect first-order approximation error from discrete hedging and rolling-window IV estimation |

The closure identity `components + residual = total` holds to machine precision by construction. The residual quantifies how much of the MTM P&L the first-order Taylor expansion cannot explain — large residuals indicate nonlinear effects (vanna, volga, vol-of-vol), and are expected in simulations with Heston stochastic vol and rolling-window implied vol estimation.

The saved `backtest_results.png` contains five panels: cumulative P&L, daily attribution stacked bar, underlying Heston price path, spread capture vs. hedge cost, and Gamma vs. Theta P&L.

---

## Resume Bullets

> Built a production-quality options market making simulator in Python: implemented Black-Scholes, binomial tree, and Monte Carlo (antithetic variates) pricers validated against put-call parity and inter-model convergence tests.

> Implemented a real-time Greeks engine (Delta, Gamma, Vega, Theta) analytically and via finite differences, with agreement verified to 4 decimal places; aggregated portfolio-level Greeks across a multi-strike, multi-expiry option book.

> Modeled realistic adverse selection using a two-population order flow model (Glosten-Milgrom-inspired): informed traders exploit quote staleness from a Heston stochastic-vol underlying, making the simulation genuinely risky rather than trivially spread-collecting.

> Built P&L attribution that decomposes daily returns into spread capture, delta hedge cost, Gamma P&L (realized vs implied variance), Theta decay, and Vega P&L — with a hard closure test asserting components sum to mark-to-market P&L to machine precision.

> Ran a 30-day market making backtest with configurable risk limits (Gamma/Vega caps, position limits per leg) and delta hedging rebalancing; reported Sharpe ratio, win rate, drawdown, and a 5-panel attribution visualization.
