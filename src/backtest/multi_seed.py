"""
Multi-seed backtest: run the default 30-day simulation across 20 random seeds
and report aggregate statistics. Saves results/multi_seed.csv.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

import configs.default as _default_cfg_module
from src.backtest.engine import BacktestEngine
from src.backtest.report import compute_sharpe, _max_drawdown


SEEDS = list(range(20))


def run_multi_seed(seeds=None):
    if seeds is None:
        seeds = SEEDS

    rows = []
    for seed in seeds:
        print(f"  seed {seed:2d} ...", end="", flush=True)
        cfg = _default_cfg_module
        results = BacktestEngine(cfg, seed=seed).run()

        sharpe      = compute_sharpe(results["daily_pnl"])
        total_pnl   = results["total_pnl"]
        max_dd      = _max_drawdown(results["daily_pnl"])
        win_days    = sum(1 for p in results["daily_pnl"] if p > 0)
        win_rate    = win_days / len(results["daily_pnl"])

        rows.append(dict(
            seed=seed,
            sharpe=sharpe,
            total_pnl=total_pnl,
            max_drawdown=max_dd,
            win_rate=win_rate,
        ))
        print(f"  sharpe={sharpe:.3f}  pnl=${total_pnl:,.0f}")

    df = pd.DataFrame(rows)

    out_dir = Path(_project_root) / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "multi_seed.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    sharpes = df["sharpe"].values
    print("\n" + "="*50)
    print("MULTI-SEED SUMMARY (20 seeds, default params)")
    print("="*50)
    print(f"  Median Sharpe:        {np.median(sharpes):.4f}")
    print(f"  Mean Sharpe:          {np.mean(sharpes):.4f}")
    print(f"  Std Sharpe:           {np.std(sharpes):.4f}")
    print(f"  Min Sharpe:           {np.min(sharpes):.4f}")
    print(f"  Max Sharpe:           {np.max(sharpes):.4f}")
    print(f"  Median Win Rate:      {np.median(df['win_rate'].values)*100:.1f}%")
    print(f"  Median Max Drawdown:  ${np.median(df['max_drawdown'].values):,.2f}")
    print(f"  Median Total P&L:     ${np.median(df['total_pnl'].values):,.2f}")
    print("="*50)

    return df


if __name__ == "__main__":
    print("Running 20-seed backtest...\n")
    run_multi_seed()
