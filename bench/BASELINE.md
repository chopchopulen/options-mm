# BASELINE — options-mm

Frozen reference numbers. **Nothing in this file may be edited except by re-running the
reproduction commands below on an unchanged working tree.** Every future claim of
"improvement" must be a before/after against this file, across multiple seeds.

## Provenance

| field | value |
|---|---|
| git commit | `dbdc9cf915855fb1944e11c3c553b77a5340e4a1` |
| branch | `main` |
| working tree | clean except untracked `docs/superpowers/plans/`, `.DS_Store`, modified `backtest_results.png` (no `.py` modified) |
| date captured | 2026-07-26 |
| python | `python3` (system, macOS darwin 25.5.0) |
| config | `configs/default.py`, unmodified |
| primary seed | **42** |

## Reproducibility gate — PASSED

A full 30-day backtest is **bit-reproducible** given a fixed seed, across separate processes.

```bash
# run twice, byte-compare full daily P&L + attribution + price path
for i in 1 2; do python3 -c "
import json, configs.default as cfg
from src.backtest.engine import BacktestEngine
r = BacktestEngine(cfg, seed=42).run()
json.dump({'daily_pnl': r['daily_pnl'], 'attr': r['daily_attribution'],
           'total': r['total_pnl'], 'prices': [float(p) for p in r['prices']]},
          open(f'/tmp/run$i.json','w'), sort_keys=True)
"; done
cmp /tmp/run1.json /tmp/run2.json   # -> identical
```

Result: **BIT-IDENTICAL**. `total_pnl` = `33837.792857164866` both runs.

### Randomness inventory (every stochastic source in the repo)

| source | file:line | seeded? | on backtest path? |
|---|---|---|---|
| Heston price/variance path | `src/market/underlying.py:15` `default_rng(seed)` | yes — `seed` | yes |
| Poisson noise arrivals, noise side, noise size, informed size | `src/market/order_flow.py:14` `default_rng(seed)` | yes — `seed + 1` | yes |
| Monte-Carlo pricer | `src/pricing/monte_carlo.py:24-25` — `rng=None` default falls back to **unseeded** `default_rng()` | **NO by default** | **no** — `engine.py` imports only `bs_price`; `mc_price` is used solely by `src/pricing/comparison.py` (which passes a seeded `rng`) and by tests |
| `comparison.py` MC studies | `src/pricing/comparison.py:40,192,195` | yes — explicit `default_rng(42)` / `default_rng(seed)` | no |
| `data.py` live SPY chain | `src/backtest/data.py:27` `yfinance` network fetch | n/a — nondeterministic, network | no |

**Nothing on the backtest path is unseeded.** The one latent hazard is
`mc_price(..., rng=None)` defaulting to a fresh unseeded generator — harmless today, a
silent reproducibility break the moment anyone wires the MC pricer into the engine.
`sensitivity.py` and `multi_seed.py` both thread explicit seeds into `BacktestEngine`.

## Reproduction commands

```bash
cd /Users/harry/Desktop/options-mm

# single-seed baseline (seed 42) — prints the summary block below
MPLBACKEND=Agg python3 -c "
import configs.default as cfg
from src.backtest.engine import BacktestEngine
from src.backtest.report import print_summary
print_summary(BacktestEngine(cfg, seed=42).run())"

# 20-seed distribution
MPLBACKEND=Agg python3 -m src.backtest.multi_seed

# test suite
python3 -m pytest -q
```

## Baseline results — seed 42, 30 days, 78 steps/day

```
Total P&L:          $  33837.79
Sharpe Ratio:            1.849
Win Rate (days):         46.7%
Max Drawdown:       $  35111.78
```

Exact values (do not round when comparing):

| metric | value |
|---|---|
| `total_pnl` | `33837.792857164866` |
| Sharpe (as coded, population std) | `1.8491533028063232` |
| Sharpe (sample std, ddof=1) | `1.8180728824876304` |
| win rate (days > 0) | `0.4666666666666667` (14/30) |
| max drawdown | `35111.77889298844` |
| daily mean P&L | `1127.9264285721622` |
| daily std (population) | `9682.9590472678` |
| daily std (sample) | `9848.491705510844` |
| n observations | `30` |

### Full attribution (cumulative, seed 42)

| component | $ | % of net | % of gross flow |
|---|---:|---:|---:|
| `spread_capture` | `22315.217747` | 65.9% | 8.95% |
| `theta_pnl` | `-12002.568024` | -35.5% | 4.81% |
| `gamma_pnl` | `2006.811254` | 5.9% | 0.81% |
| `vega_pnl` | `21597.785531` | 63.8% | 8.66% |
| `vanna_pnl` | `-562.076079` | -1.7% | 0.23% |
| `volga_pnl` | `23002.630750` | 68.0% | 9.23% |
| `hedge_cost` | `-29851.796156` | -88.2% | 11.97% |
| **`residual`** | **`7331.787834`** | **21.67%** | **2.94%** |
| **`total`** | **`33837.792857`** | 100% | — |

Gross flow denominator = sum over days of Σ|component| = `249315.35278394131`.

Note: the summary printer reports residual **only** as a fraction of net
(`report.py:41`), giving 21.67% and a `✓`. Against gross flow it is 2.94%. Both
denominators are recorded here so neither can be quoted selectively later.

### Daily P&L series (seed 42)

```
[30221.7, -8940.6, 998.0, -1661.5, -2406.9, -5138.1, 2342.1, 1374.4, 6439.3, -4736.7,
 -2594.0, -3240.8, 21165.0, 3611.2, 8446.6, -15580.8, -5724.5, -3040.4, 4518.3, -1561.7,
 -2865.7, 7233.5, -18090.4, 13417.3, -6451.7, 850.3, 6684.6, -4412.9, -1713.9, 14696.2]
```

Price path: `S0 = 450.0` → `S_T = 417.573057765501`, min `405.4626`, max `455.2404`.

## 20-seed distribution (seeds 0–19) — the honest denominator

```
Median Sharpe:        1.2232
Mean Sharpe:         -0.1437
Std Sharpe:           3.0721
Min Sharpe:          -5.1455
Max Sharpe:           4.5055
Median Win Rate:      50.0%
Median Max Drawdown:  $45,797.46
Median Total P&L:     $23,413.57
```

Per-seed P&L spans `-$311,592` (seed 5) to `+$180,958` (seed 13). Saved to
`results/multi_seed.csv`.

**Seed 42's Sharpe of 1.849 sits well inside a distribution whose mean is negative and
whose standard deviation is 3.07.** Any single-seed comparison against this baseline is
uninformative; multi-seed is mandatory.

## Test suite

`python3 -m pytest -q` → **70 passed**, 2 `IntegrationWarning`s from
`src/pricing/characteristic_function.py:65-66` (scipy `quad` roundoff on the Heston CF
integrals). No failures. Runtime ~47s.
