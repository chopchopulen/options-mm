"""
Parameter sensitivity analysis for the options market-making backtest.

Grid search over:
  - hedge_threshold   : [10, 25, 50] shares
  - base_spread_bps   : [10, 20, 50] bps  (converted → dollars via S0=450)
  - informed_threshold: [0.001, 0.002, 0.005]

For each of the 27 combos, runs 5 seeds and averages key metrics.
"""

import os
import sys
import types
import itertools
import copy
from pathlib import Path

# Ensure project root is on sys.path when the file is executed directly
# (no-op when imported normally via conftest.py or pytest).
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.report import compute_sharpe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S0 = 450.0

HEDGE_THRESHOLDS    = [10, 25, 50]
BASE_SPREAD_BPS     = [10, 20, 50]
INFORMED_THRESHOLDS = [0.001, 0.002, 0.005]
DEFAULT_SEEDS       = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------
def _make_cfg(hedge_threshold: float, base_spread: float, informed_threshold: float):
    """Return a SimpleNamespace config with the given parameters."""
    cfg = types.SimpleNamespace()
    cfg.HESTON = dict(S0=S0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.02)
    cfg.OPTION_UNIVERSE = [
        (-0.05, 30, "put"),
        (0.00,  30, "call"),
        (0.00,  30, "put"),
        (0.05,  30, "call"),
        (-0.05, 60, "put"),
        (0.00,  60, "call"),
    ]
    cfg.ORDER_FLOW = dict(
        lambda_noise=8.0,
        max_noise_size=5,
        min_informed_size=3,
        max_informed_size=12,
        staleness_threshold=informed_threshold,
    )
    cfg.QUOTER = dict(
        base_spread=base_spread,
        gamma_coeff=2.0,
        vega_coeff=0.002,
        contract_size=100,
    )
    cfg.HEDGER = dict(
        delta_threshold=hedge_threshold,
        transaction_cost=0.001,
    )
    cfg.RISK = dict(
        max_gamma=800.0,
        max_vega=50000.0,
        max_contracts_per_leg=20,
    )
    cfg.BACKTEST = dict(
        n_days=30,
        steps_per_day=78,
        sigma_uncertainty_window=10,
        quote_staleness_steps=2,
        default_sigma=0.20,
        risk_free_rate=0.02,
        desired_quote_size=5,
    )
    return cfg


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------
def _build_grid():
    """Return list of (hedge_threshold, base_spread_bps, informed_threshold) tuples."""
    return list(itertools.product(HEDGE_THRESHOLDS, BASE_SPREAD_BPS, INFORMED_THRESHOLDS))


# ---------------------------------------------------------------------------
# Single combo runner
# ---------------------------------------------------------------------------
def _run_combo(hedge_threshold: float, base_spread_bps: float, informed_threshold: float,
               seeds: list) -> dict:
    """Run one parameter combo over multiple seeds; return averaged metrics."""
    base_spread = S0 * base_spread_bps / 10_000  # convert bps → dollars

    sharpes        = []
    total_pnls     = []
    hedge_costs    = []
    spread_captures = []

    for seed in seeds:
        cfg = _make_cfg(hedge_threshold, base_spread, informed_threshold)
        results = BacktestEngine(cfg, seed=seed).run()

        sharpe = compute_sharpe(results["daily_pnl"])
        total_pnl = results["total_pnl"]
        hedge_cost = sum(day["hedge_cost"] for day in results["daily_attribution"])
        spread_capture = sum(day["spread_capture"] for day in results["daily_attribution"])

        sharpes.append(sharpe)
        total_pnls.append(total_pnl)
        hedge_costs.append(hedge_cost)
        spread_captures.append(spread_capture)

    return {
        "hedge_threshold":    hedge_threshold,
        "base_spread_bps":    base_spread_bps,
        "informed_threshold": informed_threshold,
        "mean_sharpe":        float(np.mean(sharpes)),
        "std_sharpe":         float(np.std(sharpes)),
        "mean_total_pnl":     float(np.mean(total_pnls)),
        "mean_hedge_cost":    float(np.mean(hedge_costs)),
        "mean_spread_capture": float(np.mean(spread_captures)),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_sensitivity(seeds=None, grid_override=None) -> pd.DataFrame:
    """
    Run the full sensitivity grid search.

    Parameters
    ----------
    seeds : list[int] | None
        Seeds to average over. Defaults to [42, 43, 44, 45, 46].
    grid_override : list[tuple] | None
        If provided, replaces the default 27-combo grid.
        Each element is (hedge_threshold, base_spread_bps, informed_threshold).

    Returns
    -------
    pd.DataFrame sorted by mean_sharpe descending.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    grid = grid_override if grid_override is not None else _build_grid()

    rows = []
    for i, (ht, bps, it) in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] hedge_threshold={ht}, base_spread_bps={bps}, "
              f"informed_threshold={it} ...")
        row = _run_combo(ht, bps, it, seeds)
        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "hedge_threshold",
        "base_spread_bps",
        "informed_threshold",
        "mean_sharpe",
        "std_sharpe",
        "mean_total_pnl",
        "mean_hedge_cost",
        "mean_spread_capture",
    ])
    df = df.sort_values("mean_sharpe", ascending=False).reset_index(drop=True)

    # Save
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # Print ranked table
    print("\n" + "=" * 85)
    print("SENSITIVITY ANALYSIS — RANKED BY MEAN SHARPE")
    print("=" * 85)
    header = (f"{'Rank':>4}  {'HgThr':>6}  {'BpsBps':>7}  {'InfThr':>8}  "
              f"{'Sharpe':>8}  {'StdShr':>7}  {'TotalPnL':>10}  "
              f"{'HedgeCost':>11}  {'SprdCap':>10}")
    print(header)
    print("-" * 85)
    for rank, row in df.iterrows():
        print(
            f"{rank+1:>4}  {row.hedge_threshold:>6.0f}  {row.base_spread_bps:>7.0f}  "
            f"{row.informed_threshold:>8.4f}  {row.mean_sharpe:>8.3f}  "
            f"{row.std_sharpe:>7.3f}  {row.mean_total_pnl:>10.2f}  "
            f"{row.mean_hedge_cost:>11.2f}  {row.mean_spread_capture:>10.2f}"
        )
    print("=" * 85)

    best = df.iloc[0]
    print(
        f"\nBest combo: hedge_threshold={best.hedge_threshold:.0f}, "
        f"base_spread_bps={best.base_spread_bps:.0f}, "
        f"informed_threshold={best.informed_threshold:.4f}  "
        f"→  mean_sharpe={best.mean_sharpe:.3f}"
    )

    return df


if __name__ == "__main__":
    run_sensitivity()
