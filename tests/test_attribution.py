from src.pnl.attribution import PnLAttributor


def test_attribution_fields_present():
    attr = PnLAttributor()
    result = attr.compute(
        spread_fills=[{"spread_captured": 0.10, "size": 2, "contract_size": 100}],
        portfolio_theta=-5.0,
        portfolio_gamma=0.02,
        portfolio_vega=100.0,
        portfolio_vanna=0.0,
        portfolio_volga=0.0,
        S=100.0,
        realized_variance=0.0004,
        implied_variance=0.0004,
        delta_sigma_implied=0.0,
        hedge_costs=[0.50],
        mtm_pnl=19.50,
        dt=1/252,
    )
    for key in ["spread_capture", "theta_pnl", "gamma_pnl", "vega_pnl", "vanna_pnl", "volga_pnl", "hedge_cost", "residual", "total"]:
        assert key in result


def test_engine_pnl_matches_independent_cash_and_mark_ledger():
    """The engine's reported P&L must equal an independently built cash+mark ledger.

    This replaces a tautological test that asserted
        sum(components) + residual == total
    where residual is DEFINED as total - sum(components). That identity holds for any
    inputs, including arbitrarily wrong ones, and it passed unchanged while the engine
    was overstating total P&L by 48%.

    This assertion has content: it reconstructs portfolio value from first principles as
    cash + option book mark + underlying mark, using only fill-by-fill cash flows, and
    compares it against what the engine reports. It exercises BacktestEngine, which is
    the code that actually runs -- src/pnl/attribution.py is not imported by the engine.
    """
    import configs.default as cfg
    from src.mm.inventory import Inventory
    from src.backtest.engine import BacktestEngine

    captured = {}
    original_book_value = BacktestEngine._book_value
    original_init = Inventory.__init__

    def recording_book_value(*args, **kwargs):
        # Signature-agnostic on purpose: this probe must keep reconciling when the marking
        # convention changes (it caught the per-leg vol surface change by failing loudly).
        value = original_book_value(*args, **kwargs)
        captured["book"] = value
        return value

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["inventory"] = self

    BacktestEngine._book_value = staticmethod(recording_book_value)
    Inventory.__init__ = recording_init
    try:
        for seed in (42, 3, 7):
            captured.clear()
            results = BacktestEngine(cfg, seed=seed).run()
            inventory = captured["inventory"]
            final_spot = results["prices"][cfg.BACKTEST["n_days"] * cfg.BACKTEST["steps_per_day"]]
            # Terminal portfolio value, built only from cash flows and marks.
            ledger = (captured["book"]
                      + inventory.cash
                      + inventory.underlying_position * final_spot)
            assert abs(results["total_pnl"] - ledger) < 1e-6, (
                f"seed {seed}: engine reported {results['total_pnl']:.4f}, "
                f"independent cash+mark ledger says {ledger:.4f}"
            )
    finally:
        BacktestEngine._book_value = staticmethod(original_book_value)
        Inventory.__init__ = original_init


def test_opening_trade_premium_enters_pnl():
    """Selling an option must move P&L by the premium received.

    The pre-fix engine substituted realized_pnl (an inception-to-date gain) for cash
    flow, so an opening trade contributed exactly zero and the premium vanished.
    """
    from src.mm.inventory import Inventory

    inventory = Inventory(contract_size=100)
    inventory.fill_option(450.0, 30, "call", "sell", 5, 3.00)
    assert inventory.cash == 1500.0, "premium received must land in the cash ledger"
    assert inventory.realized_pnl == 0.0, "no position was closed, so no realized gain"

    # Buying it back 10 cents lower nets $50 of cash across the round trip.
    inventory.fill_option(450.0, 30, "call", "buy", 5, 2.90)
    assert abs(inventory.cash - 50.0) < 1e-10
    assert inventory.get_option_position(450.0, 30, "call") == 0


def test_short_call_theta_pnl_positive():
    # Short call: quantity = -10, so portfolio_theta should be positive -> theta_pnl > 0
    from src.greeks.portfolio import portfolio_greeks
    positions = [{"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2,
                  "option_type": "call", "quantity": -10}]
    port = portfolio_greeks(positions, contract_size=100)
    assert port["theta"] > 0, "Short call portfolio theta should be positive"
    attr = PnLAttributor()
    result = attr.compute(
        spread_fills=[], portfolio_theta=port["theta"],
        portfolio_gamma=0.0, portfolio_vega=0.0,
        portfolio_vanna=0.0, portfolio_volga=0.0,
        S=100.0, realized_variance=0.04/252, implied_variance=0.04/252,
        delta_sigma_implied=0.0, hedge_costs=[], mtm_pnl=port["theta"] * (1/252),
        dt=1/252,
    )
    assert result["theta_pnl"] > 0


def test_no_activity_zero_pnl():
    attr = PnLAttributor()
    result = attr.compute(
        spread_fills=[],
        portfolio_theta=0.0, portfolio_gamma=0.0, portfolio_vega=0.0,
        portfolio_vanna=0.0, portfolio_volga=0.0,
        S=100.0, realized_variance=0.0004, implied_variance=0.0004,
        delta_sigma_implied=0.0, hedge_costs=[], mtm_pnl=0.0, dt=1/252,
    )
    assert result["total"] == 0.0 and result["residual"] == 0.0


def test_nonzero_vanna_volga_pass_through():
    attr = PnLAttributor()
    result = attr.compute(
        spread_fills=[],
        portfolio_theta=0.0, portfolio_gamma=0.0, portfolio_vega=0.0,
        portfolio_vanna=5.0, portfolio_volga=3.0,
        S=100.0, realized_variance=0.04/252, implied_variance=0.04/252,
        delta_sigma_implied=0.0, hedge_costs=[], mtm_pnl=8.0, dt=1/252,
    )
    assert abs(result["vanna_pnl"] - 5.0) < 1e-10
    assert abs(result["volga_pnl"] - 3.0) < 1e-10
    assert abs(result["residual"] - 0.0) < 1e-10
