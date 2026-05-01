import pytest
from src.backtest.engine import BacktestEngine
import configs.default as cfg

def test_engine_runs_without_error():
    engine = BacktestEngine(cfg, seed=42)
    results = engine.run()
    assert "daily_pnl" in results
    assert len(results["daily_pnl"]) == cfg.BACKTEST["n_days"]

def test_attribution_residual_small_each_day():
    engine = BacktestEngine(cfg, seed=42)
    results = engine.run()
    for day in results["daily_attribution"]:
        # The accounting identity must always hold: components + residual = total
        component_sum = (day["spread_capture"] + day["theta_pnl"]
                         + day["gamma_pnl"] + day["vega_pnl"]
                         + day["vanna_pnl"] + day["volga_pnl"]
                         + day["hedge_cost"] + day["residual"])
        assert abs(component_sum - day["total"]) < 1e-8, f"Identity broken: {day}"
        # With proper mark-to-market P&L, theta/gamma/vega are non-trivial,
        # so verify that the greek terms are non-zero across the backtest.
    greek_terms = [
        abs(day["theta_pnl"]) + abs(day["gamma_pnl"]) + abs(day["vega_pnl"])
        for day in results["daily_attribution"]
    ]
    assert any(g > 1.0 for g in greek_terms), "All greek attribution terms are zero — mtm_pnl is not properly computed"

def test_total_pnl_sums_daily():
    engine = BacktestEngine(cfg, seed=42)
    results = engine.run()
    assert abs(sum(results["daily_pnl"]) - results["total_pnl"]) < 1e-4
