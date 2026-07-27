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
from src.backtest.report import compute_pnl_signal_to_noise, _max_drawdown


SEEDS = list(range(20))


def run_multi_seed(seeds=None):
    if seeds is None:
        seeds = SEEDS

    rows = []
    for seed in seeds:
        print(f"  seed {seed:2d} ...", end="", flush=True)
        cfg = _default_cfg_module
        results = BacktestEngine(cfg, seed=seed).run()

        pnl_snr     = compute_pnl_signal_to_noise(results["daily_pnl"])
        total_pnl   = results["total_pnl"]
        max_dd      = _max_drawdown(results["daily_pnl"])
        win_days    = sum(1 for p in results["daily_pnl"] if p > 0)
        win_rate    = win_days / len(results["daily_pnl"])

        rows.append(dict(
            seed=seed,
            pnl_snr=pnl_snr,
            total_pnl=total_pnl,
            max_drawdown=max_dd,
            win_rate=win_rate,
        ))
        print(f"  pnl_snr={pnl_snr:.3f}  pnl=${total_pnl:,.0f}")

    df = pd.DataFrame(rows)

    out_dir = Path(_project_root) / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "multi_seed.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    snrs = df["pnl_snr"].values
    print("\n" + "="*50)
    print("MULTI-SEED SUMMARY (20 seeds, default params)")
    print("  P&L snr is NOT a Sharpe ratio — no capital base. See docs/FINAL_NUMBERS.md.")
    print("="*50)
    print(f"  Median P&L snr:       {np.median(snrs):.4f}")
    print(f"  Mean P&L snr:         {np.mean(snrs):.4f}")
    print(f"  Std P&L snr:          {np.std(snrs):.4f}")
    print(f"  Min P&L snr:          {np.min(snrs):.4f}")
    print(f"  Max P&L snr:          {np.max(snrs):.4f}")
    print(f"  Median Win Rate:      {np.median(df['win_rate'].values)*100:.1f}%")
    print(f"  Median Max Drawdown:  ${np.median(df['max_drawdown'].values):,.2f}")
    print(f"  Median Total P&L:     ${np.median(df['total_pnl'].values):,.2f}")
    print("="*50)

    return df


if __name__ == "__main__":
    print("Running 20-seed backtest...\n")
    run_multi_seed()
