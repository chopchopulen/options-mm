import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List


def compute_sharpe(daily_pnl: List[float], risk_free_daily: float = 0.02 / 252) -> float:
    arr    = np.array(daily_pnl)
    excess = arr - risk_free_daily
    if np.std(excess) == 0:
        return 0.0
    return float(np.sqrt(252) * np.mean(excess) / np.std(excess))


def _max_drawdown(pnl: List[float]) -> float:
    cumulative = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = peak - cumulative
    return float(np.max(drawdowns))


def print_summary(results: Dict) -> None:
    attrs  = results["daily_attribution"]
    pnl    = results["daily_pnl"]
    sharpe = compute_sharpe(pnl)

    df = pd.DataFrame(attrs)
    print("\n" + "="*60)
    print("OPTIONS MARKET MAKER — BACKTEST SUMMARY")
    print("="*60)
    print(f"  Total P&L:          ${results['total_pnl']:>10.2f}")
    print(f"  Sharpe Ratio:       {sharpe:>10.3f}")
    print(f"  Win Rate (days):    {np.mean(np.array(pnl) > 0)*100:>9.1f}%")
    print(f"  Max Drawdown:       ${_max_drawdown(pnl):>10.2f}")
    print()
    print("  P&L Attribution (cumulative):")
    for col in ["spread_capture", "theta_pnl", "gamma_pnl", "vega_pnl", "vanna_pnl", "volga_pnl", "hedge_cost", "residual"]:
        print(f"    {col:<22} ${df[col].sum():>10.2f}")
    residual_total = df["residual"].sum()
    residual_pct   = abs(residual_total / results["total_pnl"]) * 100 if results["total_pnl"] != 0 else 0
    status = "✓" if residual_pct < 30.0 else "✗ RESIDUAL > 30%"
    print(f"\n  Residual: ${residual_total:.4f}  ({residual_pct:.2f}% of total)  {status}")
    print("="*60 + "\n")


def plot_results(results: Dict, save_path: str = None) -> None:
    attrs  = results["daily_attribution"]
    pnl    = results["daily_pnl"]
    prices = results["prices"]
    df     = pd.DataFrame(attrs)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig)

    # Cumulative P&L
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(np.cumsum(pnl), color="steelblue", linewidth=2)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_title("Cumulative P&L")
    ax1.set_ylabel("P&L ($)")
    ax1.set_xlabel("Trading Day")

    # Daily P&L Attribution stacked bar
    ax2 = fig.add_subplot(gs[1, 0])
    components = ["spread_capture", "theta_pnl", "gamma_pnl", "vega_pnl", "vanna_pnl", "volga_pnl", "hedge_cost"]
    colors     = ["green", "orange", "blue", "purple", "teal", "brown", "red"]
    bottom_pos = np.zeros(len(df))
    bottom_neg = np.zeros(len(df))
    for comp, color in zip(components, colors):
        vals = df[comp].values
        pos  = np.where(vals > 0, vals, 0)
        neg  = np.where(vals < 0, vals, 0)
        ax2.bar(range(len(df)), pos, bottom=bottom_pos, color=color, alpha=0.7, label=comp)
        ax2.bar(range(len(df)), neg, bottom=bottom_neg, color=color, alpha=0.7)
        bottom_pos += pos
        bottom_neg += neg
    ax2.set_title("Daily P&L Attribution")
    ax2.set_xlabel("Trading Day")
    ax2.legend(fontsize=6)

    # Underlying price (downsample to ~252 points)
    ax3 = fig.add_subplot(gs[1, 1])
    step = max(1, len(prices) // 252)
    ax3.plot(prices[::step], color="gray", linewidth=1)
    ax3.set_title("Underlying Price (Heston)")
    ax3.set_ylabel("Price ($)")

    # Spread capture vs hedge cost cumulative
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(np.cumsum(df["spread_capture"]), label="Spread Capture", color="green")
    ax4.plot(np.cumsum(df["hedge_cost"]),     label="Hedge Cost",     color="red")
    ax4.set_title("Spread Capture vs Hedge Cost (cumulative)")
    ax4.legend()

    # Gamma P&L vs Theta cumulative
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(np.cumsum(df["gamma_pnl"]), label="Gamma P&L", color="blue")
    ax5.plot(np.cumsum(df["theta_pnl"]), label="Theta P&L", color="orange")
    ax5.set_title("Gamma vs Theta P&L (cumulative)")
    ax5.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
