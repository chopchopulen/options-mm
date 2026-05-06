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

from src.pricing.black_scholes import bs_price
from src.pricing.binomial import binomial_price
from src.pricing.monte_carlo import mc_price
from src.pricing.characteristic_function import heston_price_grid

S        = 100.0
r        = 0.05
SIGMA_BS = 0.20
HESTON   = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
STRIKES  = np.array([80., 85., 90., 95., 100., 105., 110., 115., 120.])
EXPIRIES = [30, 60, 90]
TRADING_DAYS = 252


def _option_type(K):
    return "put" if K <= S else "call"


def build_comparison_df() -> pd.DataFrame:
    """Run all four pricers on the 27-point grid. Returns a DataFrame."""
    rng = np.random.default_rng(42)
    rows = []

    for T_days in EXPIRIES:
        T_yr = T_days / TRADING_DAYS

        cf_calls = heston_price_grid(S, STRIKES, T_yr, r, **HESTON)

        for i, K in enumerate(STRIKES):
            otype = _option_type(K)

            bs_p  = bs_price(S, K, T_yr, r, SIGMA_BS, otype)
            bin_p = binomial_price(S, K, T_yr, r, SIGMA_BS, otype, n_steps=200)
            mc_p  = mc_price(S, K, T_yr, r, SIGMA_BS, otype, n_paths=10_000, rng=rng)

            cf_call = float(cf_calls[i])
            cf_p = cf_call - (S - K * np.exp(-r * T_yr)) if otype == "put" else cf_call

            rows.append(dict(
                strike_pct      = round(K / S * 100, 1),
                expiry_days     = T_days,
                K               = K,
                T_years         = round(T_yr, 6),
                option_type     = otype,
                bs_price        = round(bs_p,   4),
                binomial_price  = round(bin_p,  4),
                mc_price        = round(mc_p,   4),
                heston_cf_price = round(cf_p,   4),
                bs_vs_cf        = round(bs_p  - cf_p, 4),
                mc_vs_cf        = round(mc_p  - cf_p, 4),
                binomial_vs_cf  = round(bin_p - cf_p, 4),
            ))

    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, out_path: str) -> None:
    """
    Plot 1: 3x3 grouped bar chart.
    Rows = expiry (30/60/90d). Cols = K=80%, K=100%, K=120%.
    BS bar highlighted orange when |bs_vs_cf| > 0.05.
    """
    col_strikes  = [80., 100., 120.]
    row_expiries = [30, 60, 90]
    pricer_labels = ["BS", "Binomial", "MC", "Heston-CF"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("Option Prices: All Four Pricers vs. Heston-CF Benchmark", fontsize=14)

    x = np.arange(4)

    for ri, T_days in enumerate(row_expiries):
        for ci, K_val in enumerate(col_strikes):
            ax = axes[ri][ci]
            row = df[(df["expiry_days"] == T_days) & (df["K"] == K_val)]
            if row.empty:
                ax.set_visible(False)
                continue
            row = row.iloc[0]

            prices = [row["bs_price"], row["binomial_price"],
                      row["mc_price"],  row["heston_cf_price"]]
            colors = ["steelblue"] * 4
            if abs(row["bs_vs_cf"]) > 0.05:
                colors[0] = "darkorange"

            ax.bar(x, prices, width=0.6, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(pricer_labels, fontsize=8)
            ax.set_title(f"K={K_val:.0f}% | T={T_days}d\n{row['option_type']}", fontsize=9)
            ax.set_ylabel("Price ($)", fontsize=8)

            diff = row["bs_vs_cf"]
            sign = "+" if diff >= 0 else ""
            ax.text(0, prices[0] + 0.02, f"{sign}{diff:.2f}", ha="center", fontsize=7,
                    color="darkorange" if abs(diff) > 0.05 else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
