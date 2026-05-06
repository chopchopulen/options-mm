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


def plot_vol_skew(out_path: str) -> None:
    """
    Plot 2: Implied vol smile extracted from Heston-CF prices.
    Three lines for 30d, 60d, 90d expiries.
    Horizontal dashed line at flat BS vol (0.20).

    Black-Scholes assumes flat vol across strikes. Heston-CF reveals the true
    implied vol smile: OTM puts carry higher IV due to negative spot-vol correlation
    (rho=-0.7) — exactly the skew observed in real equity options markets.
    """
    strike_pcts = STRIKES / S * 100
    colors = ["steelblue", "darkorange", "forestgreen"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(SIGMA_BS, color="gray", linestyle="--", linewidth=1.2,
               label=f"BS flat vol ({SIGMA_BS:.0%})")

    for color, T_days in zip(colors, EXPIRIES):
        T_yr     = T_days / TRADING_DAYS
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


def plot_convergence(out_path: str) -> None:
    """
    Plot 3: MC convergence to Heston-CF benchmark at ATM (S=K=100, T=60/252).
    Each point = mean absolute error over 20 seeds.
    Two lines: standard MC and antithetic MC.
    O(1/sqrt(N)) reference line anchored at standard MC first point.
    """
    T_yr      = 60 / TRADING_DAYS
    K_atm     = 100.0
    benchmark = heston_price(S, K_atm, T_yr, r, **HESTON, option_type="call")
    print(f"  Heston-CF benchmark (ATM, 60d call): {benchmark:.6f}")

    n_paths_list = [100, 500, 1_000, 5_000, 10_000, 50_000]
    n_seeds      = 20

    mae_std  = []
    mae_anti = []

    for n_paths in n_paths_list:
        errors_std  = []
        errors_anti = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            p_std  = mc_price(S, K_atm, T_yr, r, SIGMA_BS, "call",
                              n_paths=n_paths, rng=rng, antithetic=False)
            rng = np.random.default_rng(seed)
            p_anti = mc_price(S, K_atm, T_yr, r, SIGMA_BS, "call",
                              n_paths=n_paths, rng=rng, antithetic=True)
            errors_std.append(abs(p_std  - benchmark))
            errors_anti.append(abs(p_anti - benchmark))
        mae_std.append(np.mean(errors_std))
        mae_anti.append(np.mean(errors_anti))
        print(f"  N={n_paths:6d}: std MAE={mae_std[-1]:.4f}  anti MAE={mae_anti[-1]:.4f}")

    # O(1/sqrt(N)) reference anchored at standard MC first point
    ns  = np.array(n_paths_list, dtype=float)
    ref = mae_std[0] * np.sqrt(n_paths_list[0] / ns)

    # Log-log slope for standard MC
    log_n  = np.log(ns)
    log_e  = np.log(np.array(mae_std))
    slope, _ = np.polyfit(log_n, log_e, 1)
    print(f"  Standard MC log-log convergence slope: {slope:.3f} (theory: -0.500)")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(ns, mae_std,  "o-", color="steelblue",  label="Standard MC")
    ax.loglog(ns, mae_anti, "s-", color="darkorange", label="Antithetic MC")
    ax.loglog(ns, ref,      "--", color="gray",        label=r"$O(1/\sqrt{N})$")
    ax.set_xlabel("Number of MC Paths")
    ax.set_ylabel("Mean Absolute Error vs. Heston-CF")
    ax.set_title(f"MC Convergence to Heston-CF Benchmark (ATM 60d Call)\n"
                 f"Standard MC slope: {slope:.3f} (theory −0.500), 20 seeds each")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def run_all() -> None:
    out_dir = Path(_root) / "results"
    out_dir.mkdir(exist_ok=True)

    print("Building comparison DataFrame (27 grid points)...")
    df = build_comparison_df()
    csv_path = str(out_dir / "pricing_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    max_bs_cf = df["bs_vs_cf"].abs().max()
    print(f"  Max |BS − CF| across grid: {max_bs_cf:.4f}")

    print("\nGenerating Plot 1: grouped bar chart...")
    plot_comparison(df, str(out_dir / "pricing_comparison.png"))

    print("\nGenerating Plot 2: vol skew...")
    plot_vol_skew(str(out_dir / "vol_skew.png"))

    print("\nGenerating Plot 3: MC convergence...")
    plot_convergence(str(out_dir / "convergence.png"))

    print("\n" + "=" * 55)
    print("HESTON CF vs BS COMPARISON SUMMARY")
    print("=" * 55)
    print(f"  Max |BS − CF| (all 27 points): ${max_bs_cf:.4f}")
    bs_cf_by_expiry = df.groupby("expiry_days")["bs_vs_cf"].apply(lambda x: x.abs().max())
    for T_days, v in bs_cf_by_expiry.items():
        print(f"    {T_days}d max |BS − CF|: ${v:.4f}")

    print("\n  Vol smile range (Heston-CF, 30d):")
    df_30 = df[df["expiry_days"] == 30].copy()
    for _, row in df_30.iterrows():
        T_yr    = row["T_years"]
        K       = row["K"]
        cf_call = row["heston_cf_price"] if row["option_type"] == "call" \
                  else row["heston_cf_price"] + (S - K * np.exp(-r * T_yr))
        try:
            iv = brentq(
                lambda s, K_=K, p_=cf_call: bs_price(S, K_, T_yr, r, s, "call") - p_,
                0.001, 5.0
            )
        except ValueError:
            iv = float("nan")
        print(f"    K={row['strike_pct']:.0f}%: IV={iv:.1%}")
    print("=" * 55)


if __name__ == "__main__":
    run_all()
