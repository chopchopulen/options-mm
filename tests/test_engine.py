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
        component_sum = (day["spread_capture"] + day["theta_pnl"]
                         + day["gamma_pnl"] + day["vega_pnl"]
                         + day["hedge_cost"] + day["residual"])
        assert abs(component_sum - day["total"]) < 1e-8, f"Identity broken: {day}"
        if abs(day["total"]) > 1.0:
            assert abs(day["residual"] / day["total"]) < 0.05, f"Residual > 5%: {day}"

def test_total_pnl_sums_daily():
    engine = BacktestEngine(cfg, seed=42)
    results = engine.run()
    assert abs(sum(results["daily_pnl"]) - results["total_pnl"]) < 1e-4
