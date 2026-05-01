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


def test_components_plus_residual_equal_total():
    # total is defined as mtm_pnl; residual closes the gap exactly
    attr = PnLAttributor()
    result = attr.compute(
        spread_fills=[{"spread_captured": 0.05, "size": 5, "contract_size": 100}],
        portfolio_theta=-3.0,
        portfolio_gamma=0.015,
        portfolio_vega=80.0,
        portfolio_vanna=0.0,
        portfolio_volga=0.0,
        S=100.0,
        realized_variance=0.0005,
        implied_variance=0.0004,
        delta_sigma_implied=0.001,
        hedge_costs=[1.20, 0.80],
        mtm_pnl=42.0,
        dt=1/252,
    )
    component_sum = (result["spread_capture"] + result["theta_pnl"]
                     + result["gamma_pnl"] + result["vega_pnl"]
                     + result["vanna_pnl"] + result["volga_pnl"]
                     + result["hedge_cost"] + result["residual"])
    assert abs(component_sum - result["total"]) < 1e-10


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
