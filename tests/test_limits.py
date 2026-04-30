from src.risk.limits import RiskLimits


def test_full_size_within_limits():
    rl = RiskLimits(max_gamma=500.0, max_vega=10000.0, max_contracts_per_leg=50)
    size = rl.adjusted_quote_size(
        desired_size=10, portfolio_gamma=100.0, portfolio_vega=2000.0, current_leg_position=5
    )
    assert size == 10


def test_gamma_limit_scales_down():
    rl = RiskLimits(max_gamma=500.0, max_vega=10000.0, max_contracts_per_leg=50)
    size = rl.adjusted_quote_size(
        desired_size=10, portfolio_gamma=490.0, portfolio_vega=0.0, current_leg_position=0
    )
    assert size < 10


def test_position_limit_caps():
    rl = RiskLimits(max_gamma=500.0, max_vega=10000.0, max_contracts_per_leg=50)
    size = rl.adjusted_quote_size(
        desired_size=10, portfolio_gamma=0.0, portfolio_vega=0.0, current_leg_position=47
    )
    assert size == 3  # only 3 contracts to reach the 50 limit


def test_at_limit_returns_zero():
    rl = RiskLimits(max_gamma=500.0, max_vega=10000.0, max_contracts_per_leg=50)
    size = rl.adjusted_quote_size(
        desired_size=10, portfolio_gamma=0.0, portfolio_vega=0.0, current_leg_position=50
    )
    assert size == 0
