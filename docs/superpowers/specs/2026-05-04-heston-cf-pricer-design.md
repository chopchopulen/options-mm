# Heston Characteristic Function Pricer — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fourth pricer (Heston CF via Gil-Pelaez + Carr-Madan FFT), a cross-pricer comparison module, and three analysis plots.

**Architecture:** Two new source files (`characteristic_function.py`, `comparison.py`), one new test file. No changes to existing files.

**Tech Stack:** numpy, scipy (integrate.quad, optimize.brentq), matplotlib; all already in requirements.

---

## 1. `src/pricing/characteristic_function.py`

Exports two public functions.

### 1a. `heston_price(S, K, T, r, v0, kappa, theta, xi, rho, option_type) -> float`

Single-strike pricer via **Gil-Pelaez inversion** using `scipy.integrate.quad`.

**Characteristic function — little-trap formulation (Albrecher et al. 2007):**

```
b    = kappa - rho·xi·iu
d    = sqrt(b² + xi²·(u² + iu))
r_m  = b - d
r_p  = b + d
h    = exp(-d·T)

A = iu·(ln S + r·T) + kappa·theta/xi² · (r_m·T - 2·ln[(r_m·h - r_p)/(-2d)])
B = r_p/xi² · (h - 1)·r_m / (r_m·h - r_p)

phi(u) = exp(A + B·v0)
```

The trap vs. original Heston (1993): identical formula except `log_arg = (r_m·h − r_p)/(−2d)` instead of `(1 − g·h)/(1−g)`. Mathematically equivalent but numerically stable on the principal Riemann sheet for all `u ∈ [0, ∞)`. Never computes `exp(+dT)`, avoiding overflow for large T.

**Verified pre-implementation:**
- `phi(0) = 1.0` (exact)
- `phi(-i) = S·exp(rT)` (martingale condition)
- Trap matches original Heston to machine precision at T=1, S=K=100
- `xi→0` limit matches BS to `<1e-4`
- Vol smile monotonically decreasing for `rho=-0.7` (negative skew)

**Gil-Pelaez inversion:**
```
P1 = 0.5 + 1/π · ∫₀^∞ Re[exp(-iu·ln K) · phi(u-i) / (iu·F)] du
P2 = 0.5 + 1/π · ∫₀^∞ Re[exp(-iu·ln K) · phi(u)   /  iu   ] du
call = exp(-rT) · (F·P1 - K·P2)
put  = call - (S - K·exp(-rT))   [put-call parity]
```

Use `scipy.integrate.quad` with `limit=500`, upper bound `500`. Handle `T <= 0` as intrinsic value.

### 1b. `heston_price_grid(S, strikes, T, r, v0, kappa, theta, xi, rho) -> np.ndarray`

Full strike grid via **Carr-Madan FFT** (prices calls; convert to puts via parity).

```
N     = 4096
eta   = 0.25          # log-strike spacing
alpha = 1.5           # damping exponent (must be > 0, avoid integers)
lambda_ = 2π/(N·eta)  # FFT output spacing in log-strike space
b     = N·lambda_/2   # center the log-strike grid at ln(F)

# Modified CF for Carr-Madan:
psi(u) = exp(-rT) · phi(u - (alpha+1)i) / (alpha² + alpha - u² + i·(2α+1)·u)

# FFT input: Simpson weights × psi × exp(-i·b·u_j·eta)
# Output: call prices on log-strike grid via interpolation
```

Steps:
1. Build `u_j = j·eta` for `j=0..N-1`
2. Evaluate `psi(u_j)` vectorized (phi accepts complex u)
3. Apply Simpson quadrature weights (1, 4, 2, 4, ..., 2, 4, 1)/3 × eta
4. FFT → call prices on log-strike grid `k_m = -b + m·lambda_` for `m=0..N-1`
5. Cubic-spline interpolate to exact `strikes` in log-strike space
6. Clip to `max(intrinsic, price)` to remove numerical negatives

---

## 2. `src/pricing/comparison.py`

Standalone script. Run: `python3 src/pricing/comparison.py`

**Grid:** 9 strikes (80%–120% of spot in 5% steps) × 3 expiries (30, 60, 90 days) = 27 points.

**Params:**
- S=100, r=0.05, sigma_bs=0.2 (flat vol for BS/binomial/MC)
- Heston: v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7 (same as simulator)

For each (K, T) point: price with all 4 methods, record differences vs Heston-CF benchmark.

**Saves:** `results/pricing_comparison.csv` with columns:
`strike_pct, expiry_days, K, T_years, option_type, bs_price, binomial_price, mc_price, heston_cf_price, bs_vs_cf, mc_vs_cf, binomial_vs_cf`

**Option type rule:** puts for K ≤ S, calls for K > S (match existing convention).

### Plot 1: `results/pricing_comparison.png`

3×3 subplot grid (rows = expiry 30/60/90d, cols = moneyness OTM put / ATM / OTM call). Each panel: grouped bar chart with 4 bars (BS, Binomial, MC, Heston-CF). Highlight bars where `|BS − CF| > 0.05` in a distinct color. Title each panel with strike%, expiry.

### Plot 2: `results/vol_skew.png`

Implied vol smile extracted from Heston-CF call prices via `scipy.optimize.brentq` on BS inverse.
- X-axis: strike as % of spot (80–120%)
- Y-axis: implied vol
- Three lines: 30d, 60d, 90d
- Horizontal dashed line at `sigma_bs = 0.20` (flat BS assumption)
- Label each line with expiry.

### Plot 3: `results/convergence.png`

MC convergence benchmark. Benchmark = `heston_price` (Gil-Pelaez) at ATM (S=K=100, T=60/252).
- X-axis: N paths (100, 500, 1k, 5k, 10k, 50k) — log scale
- Y-axis: mean absolute error vs Heston-CF — log scale
- Two lines: standard MC (`antithetic=False`), antithetic MC (`antithetic=True`)
- Each point = mean of 20 runs (different seeds) to average out randomness
- Theoretical O(1/√N) line anchored at standard MC's first point
- Log-log slope annotation: print achieved convergence rate

---

## 3. `tests/test_characteristic_function.py`

Four tests:

| Test | What it checks |
|------|---------------|
| `test_heston_cf_reduces_to_bs_when_xi_zero` | `heston_price(xi=1e-6) ≈ bs_price` to within `0.01` |
| `test_put_call_parity` | `call - put ≈ S - K·exp(-rT)` to within `1e-6` |
| `test_prices_positive_finite` | All 27 grid points: price > 0, finite, not NaN |
| `test_vol_smile_negative_skew` | Implied vol at 85% > implied vol at 100% > implied vol at 115% (for rho=-0.7) |

---

## Summary of What Changes

| File | Action |
|------|--------|
| `src/pricing/characteristic_function.py` | Create |
| `src/pricing/comparison.py` | Create |
| `tests/test_characteristic_function.py` | Create |
| `results/pricing_comparison.csv` | Generated by running comparison.py |
| `results/pricing_comparison.png` | Generated |
| `results/vol_skew.png` | Generated |
| `results/convergence.png` | Generated |

No changes to existing source files. The new pricer is standalone — it is not wired into the backtest engine (which prices with BS for speed). The comparison module is a standalone analysis script.

---

## Key Insight to Preserve in Comments

Black-Scholes prices every option using the same flat vol (`sigma = 0.2`). Heston-CF shows the true implied vol smile — OTM puts are priced with higher implied vol than ATM because Heston captures the negative spot-vol correlation (`rho = -0.7`): when spot falls, vol spikes, making puts more expensive than BS predicts. This is the vol skew observed in real equity options markets, which BS systematically misses.
