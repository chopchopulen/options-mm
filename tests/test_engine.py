import pytest
from src.backtest.engine import BacktestEngine
import configs.default as cfg

def test_engine_runs_without_error():
    engine = BacktestEngine(cfg, seed=42)
    results = engine.run()
    assert "daily_pnl" in results
    assert len(results["daily_pnl"]) == cfg.BACKTEST["n_days"]

def test_daily_totals_reconcile_to_an_independent_ledger():
    """The daily attribution series must reconcile to a cash+mark ledger.

    This replaces an assertion that
        sum(components) + residual == total
    which is a tautology: the engine DEFINES residual as total - sum(components), so it
    held for any inputs and passed unchanged while the engine overstated P&L by 48%.

    Here the terminal portfolio value is rebuilt from fill-by-fill cash flows and marks,
    independently of the attribution arithmetic, and the daily 'total' series must sum
    to it.
    """
    from src.mm.inventory import Inventory

    captured = {}
    original_book_value = BacktestEngine._book_value
    original_init = Inventory.__init__

    def recording_book_value(options, inventory, S, sigma, r, days_elapsed, contract_size):
        value = original_book_value(options, inventory, S, sigma, r, days_elapsed, contract_size)
        captured["book"] = value
        return value

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["inventory"] = self

    BacktestEngine._book_value = staticmethod(recording_book_value)
    Inventory.__init__ = recording_init
    try:
        results = BacktestEngine(cfg, seed=42).run()
    finally:
        BacktestEngine._book_value = staticmethod(original_book_value)
        Inventory.__init__ = original_init

    inventory = captured["inventory"]
    final_spot = results["prices"][cfg.BACKTEST["n_days"] * cfg.BACKTEST["steps_per_day"]]
    ledger = captured["book"] + inventory.cash + inventory.underlying_position * final_spot

    daily_total = sum(day["total"] for day in results["daily_attribution"])
    assert abs(daily_total - ledger) < 1e-6, (
        f"daily attribution totals sum to {daily_total:.4f}, "
        f"independent cash+mark ledger says {ledger:.4f}"
    )

    greek_terms = [
        abs(day["theta_pnl"]) + abs(day["gamma_pnl"]) + abs(day["vega_pnl"])
        for day in results["daily_attribution"]
    ]
    assert any(g > 1.0 for g in greek_terms), "All greek attribution terms are zero — mtm_pnl is not properly computed"

def test_total_pnl_sums_daily():
    engine = BacktestEngine(cfg, seed=42)
    results = engine.run()
    assert abs(sum(results["daily_pnl"]) - results["total_pnl"]) < 1e-4
