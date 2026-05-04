# Heston CF Pricer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Heston characteristic function pricer (Gil-Pelaez + Carr-Madan FFT), a cross-pricer comparison script, and three analysis plots.

**Architecture:** Three new files only — `src/pricing/characteristic_function.py`, `src/pricing/comparison.py`, `tests/test_characteristic_function.py`. No existing files are modified.

**Tech Stack:** numpy, scipy (integrate.quad, optimize.brentq, interpolate.CubicSpline), matplotlib — all already installed.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/pricing/characteristic_function.py` | Create | `heston_price()` (Gil-Pelaez) and `heston_price_grid()` (Carr-Madan FFT) |
| `src/pricing/comparison.py` | Create | Run all 4 pricers on 27-point grid, save CSV + 3 plots |
| `tests/test_characteristic_function.py` | Create | 4 correctness tests for `heston_price` |

---

## Task 1: Gil-Pelaez pricer (`heston_price`)

**Files:**
- Create: `src/pricing/characteristic_function.py`
- Test: `tests/test_characteristic_function.py`

### Background the implementer needs

The Heston model has a known characteristic function (CF) — the Fourier transform of the log-spot distribution. Given the CF, the Gil-Pelaez theorem recovers option prices via numerical integration.

**Little-trap CF (Albrecher et al. 2007) — verified numerically to be correct:**

```python
def _heston_cf(u, S, T, r, v0, kappa, theta, xi, rho):
    """Heston characteristic function, little-trap formulation."""
    iu  = 1j * u
    b   = kappa - rho * xi * iu
    d   = np.sqrt(b**2 + xi**2 * (u**2 + iu))
    r_m = b - d
    r_p = b + d
    h   = np.exp(-d * T)
    # log_arg stays on principal Riemann sheet for all u in [0, inf)
    log_arg  = (r_m * h - r_p) / (-2 * d)
    A = iu * (np.log(S) + r * T) + kappa * theta / xi**2 * (r_m * T - 2 * np.log(log_arg))
    B = r_p / xi**2 * (h - 1) * r_m / (r_m * h - r_p)
    return np.exp(A + B * v0)
```

**Gil-Pelaez inversion:**
```
F  = S * exp(r*T)
P1 = 0.5 + 1/π * ∫₀^∞ Re[exp(-iu*ln(K)) * phi(u-i) / (iu * F)] du
P2 = 0.5 + 1/π * ∫₀^∞ Re[exp(-iu*ln(K)) * phi(u)   / (iu)    ] du
call = exp(-r*T) * (F*P1 - K*P2)
put  = call - (S - K*exp(-r*T))   ← put-call parity
```

Use `scipy.integrate.quad(fn, 1e-10, 500, limit=500, epsabs=1e-9)` for each integral (start at 1e-10 not 0 to avoid the integrand singularity at u=0).

- [ ] **Step 1: Write all four failing tests**

Create `tests/test_characteristic_function.py`:

```python
import numpy as np
import pytest
from scipy.optimize import brentq
from src.pricing.black_scholes import bs_price
from src.pricing.characteristic_function import heston_price

# Shared Heston params matching the simulator's default config
HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
S, K, T, r = 100.0, 100.0, 1.0, 0.05


def test_heston_cf_reduces_to_bs_when_xi_zero():
    # When xi (vol-of-vol) → 0, Heston reduces to BS with sigma = sqrt(v0) = 0.2
    heston_p = heston_price(S, K, T, r, v0=0.04, kappa=2.0, theta=0.04, xi=1e-6, rho=0.0, option_type="call")
    bs_p     = bs_price(S, K, T, r, sigma=0.2, option_type="call")
    assert abs(heston_p - bs_p) < 0.01


def test_put_call_parity():
    call = heston_price(S, K, T, r, **HESTON, option_type="call")
    put  = heston_price(S, K, T, r, **HESTON, option_type="put")
    parity_rhs = S - K * np.exp(-r * T)
    assert abs((call - put) - parity_rhs) < 1e-6


def test_prices_positive_finite():
    # All 27 grid points must be positive and finite
    spot_pcts = np.arange(0.80, 1.21, 0.05)
    expiries  = [30, 60, 90]
    for T_days in expiries:
        T_yr = T_days / 252
        for pct in spot_pcts:
            K_i   = S * pct
            otype = "put" if pct <= 1.0 else "call"
            p = heston_price(S, K_i, T_yr, r, **HESTON, option_type=otype)
            assert np.isfinite(p) and p > 0, f"Bad price at K={K_i}, T={T_days}d: {p}"


def test_vol_smile_negative_skew():
    # rho=-0.7 → negative skew → OTM put IV > ATM IV > OTM call IV
    T_yr = 1.0

    def implied_vol(price, K_iv, otype):
        f = lambda s: bs_price(S, K_iv, T_yr, r, s, otype) - price
        return brentq(f, 0.001, 5.0, xtol=1e-8)

    call_atm  = heston_price(S, S * 1.00, T_yr, r, **HESTON, option_type="call")
    call_otmc = heston_price(S, S * 1.15, T_yr, r, **HESTON, option_type="call")
    put_otmp  = heston_price(S, S * 0.85, T_yr, r, **HESTON, option_type="put")

    iv_atm   = implied_vol(call_atm,  S * 1.00, "call")
    iv_otmc  = implied_vol(call_otmc, S * 1.15, "call")
    iv_otmp  = implied_vol(put_otmp,  S * 0.85, "put")

    assert iv_otmp > iv_atm > iv_otmc, (
        f"Expected negative skew: IV(85%)={iv_otmp:.4f} > IV(100%)={iv_atm:.4f} "
        f"> IV(115%)={iv_otmc:.4f}"
    )
```

- [ ] **Step 2: Run tests to confirm they all fail with ImportError**

```bash
cd /path/to/options-mm
pytest tests/test_characteristic_function.py -v
```

Expected: 4 errors — `ModuleNotFoundError: No module named 'src.pricing.characteristic_function'`

- [ ] **Step 3: Implement `heston_price` in `characteristic_function.py`**

Create `src/pricing/characteristic_function.py`:

```python
import numpy as np
from scipy import integrate, optimize


def _heston_cf(u, S, T, r, v0, kappa, theta, xi, rho):
    """
    Heston (1993) characteristic function, little-trap formulation (Albrecher et al. 2007).

    The trap avoids Riemann sheet discontinuities by choosing log_arg = (r_m*h - r_p)/(-2d)
    instead of the original (1 - g*h)/(1-g). Mathematically equivalent, numerically stable
    for all u in [0, inf) without computing exp(+dT).
    """
    iu  = 1j * u
    b   = kappa - rho * xi * iu
    d   = np.sqrt(b**2 + xi**2 * (u**2 + iu))
    r_m = b - d
    r_p = b + d
    h   = np.exp(-d * T)
    log_arg = (r_m * h - r_p) / (-2 * d)
    A = iu * (np.log(S) + r * T) + kappa * theta / xi**2 * (r_m * T - 2 * np.log(log_arg))
    B = r_p / xi**2 * (h - 1) * r_m / (r_m * h - r_p)
    return np.exp(A + B * v0)


def heston_price(S: float, K: float, T: float, r: float,
                 v0: float, kappa: float, theta: float, xi: float, rho: float,
                 option_type: str) -> float:
    """
    Price a European option under the Heston stochastic volatility model
    using Gil-Pelaez Fourier inversion.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        v0: Initial variance (sigma^2 at t=0)
        kappa: Mean-reversion speed of variance
        theta: Long-run variance (sigma^2 long-run mean)
        xi: Vol-of-vol (volatility of variance process)
        rho: Correlation between spot and variance Brownian motions
        option_type: "call" or "put"

    Returns:
        Option price
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    F   = S * np.exp(r * T)
    lnK = np.log(K)

    def integrand_P1(u):
        phi = _heston_cf(u - 1j, S, T, r, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-1j * u * lnK) * phi / (1j * u * F))

    def integrand_P2(u):
        phi = _heston_cf(u, S, T, r, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-1j * u * lnK) * phi / (1j * u))

    P1 = 0.5 + 1 / np.pi * integrate.quad(integrand_P1, 1e-10, 500, limit=500, epsabs=1e-9)[0]
    P2 = 0.5 + 1 / np.pi * integrate.quad(integrand_P2, 1e-10, 500, limit=500, epsabs=1e-9)[0]

    call = np.exp(-r * T) * (F * P1 - K * P2)
    if option_type == "call":
        return float(call)
    # Put via put-call parity (avoids second integration)
    return float(call - (S - K * np.exp(-r * T)))
```

- [ ] **Step 4: Run tests — all four should pass**

```bash
pytest tests/test_characteristic_function.py -v
```

Expected output:
```
test_characteristic_function.py::test_heston_cf_reduces_to_bs_when_xi_zero PASSED
test_characteristic_function.py::test_put_call_parity PASSED
test_characteristic_function.py::test_prices_positive_finite PASSED
test_characteristic_function.py::test_vol_smile_negative_skew PASSED
4 passed
```

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
pytest tests/ -q
```

Expected: `66 passed` (no new failures)

- [ ] **Step 6: Commit**

```bash
git add src/pricing/characteristic_function.py tests/test_characteristic_function.py
git commit -m "feat: add Heston CF pricer via Gil-Pelaez inversion (little-trap formulation)"
```

---

## Task 2: Carr-Madan FFT grid pricer (`heston_price_grid`)

**Files:**
- Modify: `src/pricing/characteristic_function.py` (add `heston_price_grid`)

### Background the implementer needs

Carr-Madan (1999) prices a full grid of strikes with a single FFT call. The trick: introduce a damping factor `exp(alpha * k)` to make the call pricing function square-integrable, then take the FFT, then divide out the damping.

**Algorithm:**

```
N      = 4096           # FFT size — must be power of 2
eta    = 0.25           # spacing in u-space (frequency domain)
alpha  = 1.5            # damping parameter
lam    = 2π / (N * eta) # spacing in log-strike space
b      = N * lam / 2    # half-width → log-strike grid centered near ln(F)

# u-space grid
u_j = j * eta  for j = 0, ..., N-1

# Modified integrand (Carr-Madan eq 24):
# psi(u) = exp(-r*T) * phi(u - (alpha+1)*i) / (alpha^2 + alpha - u^2 + i*(2*alpha+1)*u)
# where phi is the Heston CF of ln(S_T)

# Simpson weights
w = eta/3 * [1, 4, 2, 4, 2, ..., 2, 4, 1]  (standard Simpson rule)

# FFT input
x_j = exp(-i * b * u_j) * psi(u_j) * w_j

# FFT output → call prices (after damping removal)
fft_out = real(FFT(x_j))
call_m  = exp(-alpha * k_m) / pi * fft_out_m
where k_m = -b + m * lam  (log-strike grid)

# Interpolate to desired log-strikes
Use scipy.interpolate.CubicSpline on (k_m, call_m) arrays.
Clip output to max(intrinsic, price).
```

Note: `_heston_cf` already accepts complex `u`, so `u - (alpha+1)*1j` works directly.

- [ ] **Step 1: Add `heston_price_grid` to `characteristic_function.py`**

Append to `src/pricing/characteristic_function.py` (after `heston_price`):

```python
def heston_price_grid(S: float, strikes: np.ndarray, T: float, r: float,
                      v0: float, kappa: float, theta: float, xi: float,
                      rho: float) -> np.ndarray:
    """
    Price European calls across a full strike grid under Heston using Carr-Madan FFT.

    Returns call prices. Convert to puts via: put = call - (S - K*exp(-rT)).

    Args:
        S: Spot price
        strikes: 1-D array of strike prices
        T: Time to maturity in years
        r: Risk-free rate
        v0, kappa, theta, xi, rho: Heston parameters (same as heston_price)

    Returns:
        1-D numpy array of call prices, same length as strikes.
    """
    from scipy.interpolate import CubicSpline

    N     = 4096
    eta   = 0.25
    alpha = 1.5
    lam   = 2 * np.pi / (N * eta)
    b     = N * lam / 2

    # u-space grid
    j   = np.arange(N)
    u_j = j * eta

    # Carr-Madan modified integrand (vectorized)
    denom = alpha**2 + alpha - u_j**2 + 1j * (2 * alpha + 1) * u_j
    phi_u = _heston_cf(u_j - 1j * (alpha + 1), S, T, r, v0, kappa, theta, xi, rho)
    psi   = np.exp(-r * T) * phi_u / denom

    # Simpson weights
    w      = np.ones(N)
    w[1::2] = 4
    w[2::2] = 2
    w[-1]   = 1
    w      *= eta / 3

    # FFT
    x       = np.exp(-1j * b * u_j) * psi * w
    fft_out = np.real(np.fft.fft(x))

    # Log-strike grid
    k_m    = -b + j * lam
    call_m = np.exp(-alpha * k_m) / np.pi * fft_out

    # Interpolate to requested strikes
    ln_strikes = np.log(strikes)
    cs         = CubicSpline(k_m, call_m)
    calls      = cs(ln_strikes)

    # Clip: call price >= max(intrinsic, 0)
    F         = S * np.exp(r * T)
    intrinsic = np.maximum(F * np.exp(ln_strikes) / S - K, 0)  # rough lower bound
    # Exact intrinsic per strike
    disc = np.exp(-r * T)
    intrinsic = np.maximum(strikes - S * np.exp(r * T) * disc, 0) * 0  # calls: max(S-K*disc... no)
    # Simplest: just clip to 0
    calls = np.maximum(calls, 0.0)
    return calls
```

Wait — the intrinsic clipping logic above is garbled. Use this clean version instead:

```python
def heston_price_grid(S: float, strikes: np.ndarray, T: float, r: float,
                      v0: float, kappa: float, theta: float, xi: float,
                      rho: float) -> np.ndarray:
    """
    Price European calls across a full strike grid under Heston using Carr-Madan FFT.

    Returns call prices. Convert to puts via: put = call - (S - K*exp(-rT)).

    Args:
        S: Spot price
        strikes: 1-D array of strike prices
        T: Time to maturity in years
        r: Risk-free rate
        v0, kappa, theta, xi, rho: Heston parameters (same as heston_price)

    Returns:
        1-D numpy array of call prices, same length as strikes.
    """
    from scipy.interpolate import CubicSpline

    strikes = np.asarray(strikes, dtype=float)
    N     = 4096
    eta   = 0.25
    alpha = 1.5
    lam   = 2 * np.pi / (N * eta)
    b     = N * lam / 2

    j   = np.arange(N)
    u_j = j * eta

    denom = alpha**2 + alpha - u_j**2 + 1j * (2 * alpha + 1) * u_j
    phi_u = _heston_cf(u_j - 1j * (alpha + 1), S, T, r, v0, kappa, theta, xi, rho)
    psi   = np.exp(-r * T) * phi_u / denom

    w      = np.ones(N)
    w[1::2] = 4
    w[2::2] = 2
    w[-1]   = 1
    w      *= eta / 3

    x       = np.exp(-1j * b * u_j) * psi * w
    fft_out = np.real(np.fft.fft(x))

    k_m    = -b + j * lam
    call_m = np.exp(-alpha * k_m) / np.pi * fft_out

    ln_strikes = np.log(strikes)
    cs    = CubicSpline(k_m, call_m)
    calls = cs(ln_strikes)

    # Call price lower bound: max(S - K*exp(-rT), 0) — clip numerical negatives
    lower = np.maximum(S - strikes * np.exp(-r * T), 0.0)
    calls = np.maximum(calls, lower)
    return calls
```

- [ ] **Step 2: Write a quick validation script (not a permanent test) to confirm FFT matches Gil-Pelaez**

```bash
python3 - <<'EOF'
import numpy as np
from src.pricing.characteristic_function import heston_price, heston_price_grid

S, r = 100.0, 0.05
HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
strikes = np.array([80., 85., 90., 95., 100., 105., 110., 115., 120.])
T = 60 / 252

fft_calls = heston_price_grid(S, strikes, T, r, **HESTON)

print(f"{'K':>6}  {'FFT call':>10}  {'GP call':>10}  {'diff':>10}")
for K, fft_c in zip(strikes, fft_calls):
    otype = "put" if K <= S else "call"
    gp_c = heston_price(S, K, T, r, **HESTON, option_type="call")
    print(f"{K:>6.0f}  {fft_c:>10.4f}  {gp_c:>10.4f}  {abs(fft_c-gp_c):>10.4f}")
EOF
```

Expected: FFT vs Gil-Pelaez differences all below `0.05` across the strike grid.

- [ ] **Step 3: Run full test suite — still 66 passing, no regressions**

```bash
pytest tests/ -q
```

- [ ] **Step 4: Commit**

```bash
git add src/pricing/characteristic_function.py
git commit -m "feat: add Carr-Madan FFT grid pricer (heston_price_grid)"
```

---

## Task 3: Comparison script — CSV + Plot 1

**Files:**
- Create: `src/pricing/comparison.py`

### Background the implementer needs

This standalone script runs all four pricers on a 27-point grid and saves a CSV plus three plots. Run it with: `python3 src/pricing/comparison.py` from the project root.

Existing pricers to import:
- `from src.pricing.black_scholes import bs_price`
- `from src.pricing.binomial import binomial_price`
- `from src.pricing.monte_carlo import mc_price`
- `from src.pricing.characteristic_function import heston_price, heston_price_grid`

**Grid params:**
- `S = 100.0`, `r = 0.05`, `sigma_bs = 0.20` (flat vol for BS/binomial/MC)
- Heston: `v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7`
- Strikes: `[80, 85, 90, 95, 100, 105, 110, 115, 120]` (9 strikes, 80%–120% of S)
- Expiries: `[30, 60, 90]` days → `[30/252, 60/252, 90/252]` years
- Option type: `"put"` if `K <= S` else `"call"`

Use `heston_price_grid` per expiry (one FFT call per expiry, fast). Use `heston_price` as a fallback only if you need single-point calls elsewhere.

MC: use `n_paths=10_000`, `rng=np.random.default_rng(42)` (fixed seed for reproducibility).
Binomial: use `n_steps=200` (fast enough, close to BS).

**CSV columns:** `strike_pct, expiry_days, K, T_years, option_type, bs_price, binomial_price, mc_price, heston_cf_price, bs_vs_cf, mc_vs_cf, binomial_vs_cf`

where `bs_vs_cf = bs_price - heston_cf_price` (signed difference, positive = BS overprices).

**Plot 1 layout:** 3 rows (expiry 30/60/90d) × 3 cols (K=80% / K=100% / K=120%). Each panel: grouped bar with 4 bars. BS bar is orange if `|bs_vs_cf| > 0.05`, else default blue. X-tick labels: `["BS", "Binomial", "MC", "Heston-CF"]`.

- [ ] **Step 1: Create `src/pricing/comparison.py` with the data-collection function**

```python
"""
Cross-pricer comparison: BS, Binomial, Monte Carlo, Heston-CF on a 27-point grid.

Run: python3 src/pricing/comparison.py
Saves: results/pricing_comparison.csv, results/pricing_comparison.png,
       results/vol_skew.png, results/convergence.png
"""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq

from src.pricing.black_scholes import bs_price
from src.pricing.binomial import binomial_price
from src.pricing.monte_carlo import mc_price
from src.pricing.characteristic_function import heston_price, heston_price_grid

S        = 100.0
r        = 0.05
SIGMA_BS = 0.20
HESTON   = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
STRIKES  = np.array([80., 85., 90., 95., 100., 105., 110., 115., 120.])
EXPIRIES = [30, 60, 90]   # days
TRADING_DAYS = 252


def _option_type(K):
    return "put" if K <= S else "call"


def build_comparison_df() -> pd.DataFrame:
    """Run all four pricers on the 27-point grid. Returns a DataFrame."""
    rng = np.random.default_rng(42)
    rows = []

    for T_days in EXPIRIES:
        T_yr = T_days / TRADING_DAYS

        # Heston FFT: one call per expiry prices all strikes at once
        cf_calls = heston_price_grid(S, STRIKES, T_yr, r, **HESTON)

        for i, K in enumerate(STRIKES):
            otype  = _option_type(K)

            bs_p   = bs_price(S, K, T_yr, r, SIGMA_BS, otype)
            bin_p  = binomial_price(S, K, T_yr, r, SIGMA_BS, otype, n_steps=200)
            mc_p   = mc_price(S, K, T_yr, r, SIGMA_BS, otype, n_paths=10_000, rng=rng)

            # Convert FFT call to put if needed
            cf_call = float(cf_calls[i])
            if otype == "put":
                cf_p = cf_call - (S - K * np.exp(-r * T_yr))
            else:
                cf_p = cf_call

            rows.append(dict(
                strike_pct   = round(K / S * 100, 1),
                expiry_days  = T_days,
                K            = K,
                T_years      = round(T_yr, 6),
                option_type  = otype,
                bs_price     = round(bs_p,  4),
                binomial_price = round(bin_p, 4),
                mc_price     = round(mc_p,  4),
                heston_cf_price = round(cf_p, 4),
                bs_vs_cf     = round(bs_p  - cf_p, 4),
                mc_vs_cf     = round(mc_p  - cf_p, 4),
                binomial_vs_cf = round(bin_p - cf_p, 4),
            ))

    return pd.DataFrame(rows)
```

- [ ] **Step 2: Add `plot_comparison` (Plot 1) to `comparison.py`**

Append to `src/pricing/comparison.py`:

```python
def plot_comparison(df: pd.DataFrame, out_path: str) -> None:
    """
    Plot 1: 3×3 grouped bar chart.
    Rows = expiry (30/60/90d). Cols = K=80%, K=100%, K=120%.
    BS bar highlighted orange when |bs_vs_cf| > 0.05.
    """
    col_strikes = [80., 100., 120.]
    col_labels  = ["80% (OTM put)", "100% (ATM)", "120% (OTM call)"]
    row_expiries = [30, 60, 90]
    pricer_labels = ["BS", "Binomial", "MC", "Heston-CF"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("Option Prices: All Four Pricers vs. Heston-CF Benchmark", fontsize=14)

    x = np.arange(4)  # 4 bars per panel
    width = 0.6

    for ri, T_days in enumerate(row_expiries):
        for ci, K_pct in enumerate(col_strikes):
            ax = axes[ri][ci]
            row = df[(df["expiry_days"] == T_days) & (df["K"] == K_pct)]
            if row.empty:
                ax.set_visible(False)
                continue
            row = row.iloc[0]

            prices = [row["bs_price"], row["binomial_price"],
                      row["mc_price"],  row["heston_cf_price"]]
            colors = ["steelblue", "steelblue", "steelblue", "steelblue"]
            if abs(row["bs_vs_cf"]) > 0.05:
                colors[0] = "darkorange"  # flag BS divergence

            bars = ax.bar(x, prices, width=width, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(pricer_labels, fontsize=8)
            ax.set_title(f"K={K_pct:.0f}% | T={T_days}d\n{row['option_type']}", fontsize=9)
            ax.set_ylabel("Price ($)", fontsize=8)

            # Annotate BS-vs-CF diff on the BS bar
            diff = row["bs_vs_cf"]
            sign = "+" if diff >= 0 else ""
            ax.text(0, prices[0] + 0.02, f"{sign}{diff:.2f}", ha="center", fontsize=7,
                    color="darkorange" if abs(diff) > 0.05 else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
```

- [ ] **Step 3: Run the script so far to verify CSV and Plot 1 are produced**

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, '.')
from src.pricing.comparison import build_comparison_df, plot_comparison
from pathlib import Path

Path("results").mkdir(exist_ok=True)
df = build_comparison_df()
df.to_csv("results/pricing_comparison.csv", index=False)
print(df[["K", "expiry_days", "option_type", "bs_price", "heston_cf_price", "bs_vs_cf"]].to_string())
plot_comparison(df, "results/pricing_comparison.png")
EOF
```

Expected: CSV with 27 rows, plot saved. BS prices should deviate from Heston-CF most at OTM strikes (K=80, K=120).

- [ ] **Step 4: Commit**

```bash
git add src/pricing/comparison.py results/pricing_comparison.csv results/pricing_comparison.png
git commit -m "feat: add comparison script, pricing grid CSV, and Plot 1 (grouped bar chart)"
```

---

## Task 4: Vol skew plot (Plot 2)

**Files:**
- Modify: `src/pricing/comparison.py` (add `plot_vol_skew`)

### Background the implementer needs

Extract implied vol from Heston-CF call prices by inverting BS via `scipy.optimize.brentq`. For each strike: get Heston-CF call price, find the BS sigma that matches it.

Use calls throughout (even OTM puts should be converted to calls via parity before inverting) to avoid put-specific BS formula edge cases:
```python
# From Heston-CF: get call price regardless of option_type
cf_call = heston_price_grid(S, strikes, T_yr, r, **HESTON)[i]
# BS call IV
iv = brentq(lambda s: bs_price(S, K, T_yr, r, s, "call") - cf_call, 0.001, 5.0)
```

- [ ] **Step 1: Add `plot_vol_skew` to `comparison.py`**

Append to `src/pricing/comparison.py`:

```python
def plot_vol_skew(out_path: str) -> None:
    """
    Plot 2: Implied vol smile extracted from Heston-CF prices.
    Three lines for 30d, 60d, 90d expiries.
    Horizontal dashed line at flat BS vol (0.20).

    Black-Scholes assumes flat vol across strikes. Heston-CF reveals the true
    implied vol smile: OTM puts carry higher IV due to negative spot-vol correlation
    (rho=-0.7) — exactly the skew observed in real equity options markets.
    """
    strike_pcts = STRIKES / S * 100   # x-axis values
    colors = ["steelblue", "darkorange", "forestgreen"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(SIGMA_BS, color="gray", linestyle="--", linewidth=1.2,
               label=f"BS flat vol ({SIGMA_BS:.0%})")

    for color, T_days in zip(colors, EXPIRIES):
        T_yr    = T_days / TRADING_DAYS
        cf_calls = heston_price_grid(S, STRIKES, T_yr, r, **HESTON)
        ivs = []
        for K, cf_call in zip(STRIKES, cf_calls):
            try:
                iv = brentq(
                    lambda s, K_=K, p_=float(cf_call): bs_price(S, K_, T_yr, r, s, "call") - p_,
                    0.001, 5.0, xtol=1e-8
                )
            except ValueError:
                iv = np.nan
            ivs.append(iv)

        ax.plot(strike_pcts, ivs, marker="o", color=color, label=f"{T_days}d")

    ax.set_xlabel("Strike (% of spot)")
    ax.set_ylabel("Implied Volatility")
    ax.set_title("Heston-CF Implied Vol Smile vs. BS Flat Vol\n"
                 r"$\rho=-0.7$: negative skew — OTM puts more expensive than BS predicts")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
```

- [ ] **Step 2: Run to verify vol_skew.png is produced**

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, '.')
from src.pricing.comparison import plot_vol_skew
plot_vol_skew("results/vol_skew.png")
EOF
```

Expected: plot saved. The three curves should all slope downward left-to-right (higher IV at low strikes, lower IV at high strikes). All curves above or crossing the 20% flat line near ATM.

- [ ] **Step 3: Commit**

```bash
git add src/pricing/comparison.py results/vol_skew.png
git commit -m "feat: add vol skew plot (Plot 2)"
```

---

## Task 5: MC convergence plot (Plot 3)

**Files:**
- Modify: `src/pricing/comparison.py` (add `plot_convergence`)

### Background the implementer needs

Benchmark = `heston_price(S=100, K=100, T=60/252, r=0.05, **HESTON, option_type="call")` — single Gil-Pelaez call (accurate to ~1e-9).

For each N in `[100, 500, 1_000, 5_000, 10_000, 50_000]`:
- Run 20 trials with different seeds (seed=0..19)
- Each trial: `mc_price(S, K, T, r, SIGMA_BS, "call", n_paths=N, rng=np.random.default_rng(seed), antithetic=False/True)`
- Record `|mc_price - benchmark|` for each trial
- Plot mean of the 20 absolute errors

Theoretical O(1/√N) line: anchor at the standard MC's mean error at N=100, then scale as `error_at_100 * sqrt(100) / sqrt(N)`.

Log-log slope: fit `np.polyfit(np.log(N_values), np.log(mean_errors_standard), 1)` — should be ~-0.5.

- [ ] **Step 1: Add `plot_convergence` to `comparison.py`**

Append to `src/pricing/comparison.py`:

```python
def plot_convergence(out_path: str) -> None:
    """
    Plot 3: MC convergence — absolute error vs. Heston-CF benchmark.
    Standard MC vs. antithetic MC. Theoretical O(1/sqrt(N)) reference line.
    """
    T_yr      = 60 / TRADING_DAYS
    benchmark = heston_price(S, S, T_yr, r, **HESTON, option_type="call")  # ATM call

    n_values = [100, 500, 1_000, 5_000, 10_000, 50_000]
    n_trials  = 20

    mean_std  = []
    mean_anti = []

    for N in n_values:
        errs_std  = []
        errs_anti = []
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            p_std  = mc_price(S, S, T_yr, r, SIGMA_BS, "call", n_paths=N, rng=rng, antithetic=False)
            rng = np.random.default_rng(seed)
            p_anti = mc_price(S, S, T_yr, r, SIGMA_BS, "call", n_paths=N, rng=rng, antithetic=True)
            errs_std.append(abs(p_std  - benchmark))
            errs_anti.append(abs(p_anti - benchmark))
        mean_std.append(np.mean(errs_std))
        mean_anti.append(np.mean(errs_anti))

    # Theoretical O(1/sqrt(N)) anchored at first std MC point
    theory = [mean_std[0] * np.sqrt(n_values[0] / N) for N in n_values]

    # Fit log-log slope
    slope_std,  _ = np.polyfit(np.log(n_values), np.log(mean_std),  1)
    slope_anti, _ = np.polyfit(np.log(n_values), np.log(mean_anti), 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(n_values, mean_std,  "o-", color="steelblue",
              label=f"Standard MC (slope={slope_std:.2f})")
    ax.loglog(n_values, mean_anti, "s-", color="darkorange",
              label=f"Antithetic MC (slope={slope_anti:.2f})")
    ax.loglog(n_values, theory, "--", color="gray",
              label="Theoretical O(1/√N) slope=-0.50")

    ax.set_xlabel("Number of MC paths (N)")
    ax.set_ylabel("Mean absolute error vs. Heston-CF (log scale)")
    ax.set_title("Monte Carlo Convergence to Heston-CF Benchmark\n"
                 f"ATM call, T=60d  |  benchmark={benchmark:.4f}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    print(f"Achieved convergence rate: standard MC={slope_std:.3f}, "
          f"antithetic MC={slope_anti:.3f} (theory: -0.500)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
```

- [ ] **Step 2: Run to verify convergence.png is produced and rates are reported**

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, '.')
from src.pricing.comparison import plot_convergence
plot_convergence("results/convergence.png")
EOF
```

Expected output (approximate):
```
Achieved convergence rate: standard MC=-0.48X, antithetic MC=-0.4XX (theory: -0.500)
Saved results/convergence.png
```

Both slopes should be in range `[-0.40, -0.55]`. Antithetic mean errors should be smaller than standard at every N.

- [ ] **Step 3: Commit**

```bash
git add src/pricing/comparison.py results/convergence.png
git commit -m "feat: add MC convergence plot (Plot 3)"
```

---

## Task 6: Wire up `__main__` and run full comparison

**Files:**
- Modify: `src/pricing/comparison.py` (add `__main__` block and `run_all`)
- Generate: `results/pricing_comparison.csv`, all three plots

- [ ] **Step 1: Add `run_all` and `__main__` block to `comparison.py`**

Append to `src/pricing/comparison.py`:

```python
def run_all() -> pd.DataFrame:
    """Run full comparison: save CSV and all three plots. Returns the comparison DataFrame."""
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)

    print("Building pricing comparison grid (27 points × 4 pricers)...")
    df = build_comparison_df()
    df.to_csv("results/pricing_comparison.csv", index=False)
    print(f"Saved results/pricing_comparison.csv ({len(df)} rows)")

    max_bs_diff = df["bs_vs_cf"].abs().max()
    print(f"\nMax |BS - Heston-CF| across grid: ${max_bs_diff:.4f}")
    print("(Largest divergence occurs at OTM strikes where Heston skew differs from flat vol)\n")

    print("Generating plots...")
    plot_comparison(df, "results/pricing_comparison.png")
    plot_vol_skew("results/vol_skew.png")
    plot_convergence("results/convergence.png")

    return df


if __name__ == "__main__":
    run_all()
```

- [ ] **Step 2: Run the full script and capture the reported metrics**

```bash
python3 src/pricing/comparison.py
```

Expected output:
```
Building pricing comparison grid (27 points × 4 pricers)...
Saved results/pricing_comparison.csv (27 rows)

Max |BS - Heston-CF| across grid: $X.XXXX
(Largest divergence occurs at OTM strikes where Heston skew differs from flat vol)

Generating plots...
Saved results/pricing_comparison.png
Saved results/vol_skew.png
Achieved convergence rate: standard MC=-0.4XX, antithetic MC=-0.4XX (theory: -0.500)
Saved results/convergence.png
```

Record the three output numbers to report back: max BS-CF diff, vol smile range (from the vol skew plot — check `results/vol_skew.png`), and convergence rates.

- [ ] **Step 3: Run full test suite — should still be 66 passing (4 new = 70 total)**

```bash
pytest tests/ -q
```

Expected: `70 passed`

- [ ] **Step 4: Final commit**

```bash
git add src/pricing/comparison.py results/pricing_comparison.csv \
        results/pricing_comparison.png results/vol_skew.png results/convergence.png
git commit -m "feat: wire up comparison __main__, generate all outputs

Run: python3 src/pricing/comparison.py"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| `heston_price` (Gil-Pelaez, little-trap) | Task 1 |
| `heston_price_grid` (Carr-Madan FFT) | Task 2 |
| 27-point comparison grid + CSV | Task 3 |
| Plot 1: grouped bar chart | Task 3 |
| Plot 2: vol skew smile | Task 4 |
| Plot 3: MC convergence | Task 5 |
| `__main__` entry point | Task 6 |
| Test: xi=0 BS limit | Task 1 |
| Test: put-call parity | Task 1 |
| Test: positive finite prices | Task 1 |
| Test: negative skew direction | Task 1 |
| Report max BS-CF diff | Task 6 |
| Report vol smile range | Task 6 |
| Report convergence rate | Task 5 |

All spec requirements are covered.

**Type/signature consistency check:**
- `_heston_cf(u, S, T, r, v0, kappa, theta, xi, rho)` — used internally by both `heston_price` and `heston_price_grid` ✓
- `heston_price(..., option_type)` — consistent with `bs_price`, `mc_price`, `binomial_price` signatures ✓
- `heston_price_grid(S, strikes, T, r, v0, kappa, theta, xi, rho)` — no `option_type` (always returns calls) ✓
- `HESTON = dict(v0=..., kappa=..., theta=..., xi=..., rho=...)` — keyword-unpacked as `**HESTON` in all calls ✓
