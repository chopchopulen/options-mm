# All simulation thresholds — set once, never tuned per run.

HESTON = dict(
    S0=450.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.02
)

OPTION_UNIVERSE = [
    # (strike_offset_pct, expiry_days, option_type)
    (-0.05, 30, "put"),
    (0.00,  30, "call"),
    (0.00,  30, "put"),
    (0.05,  30, "call"),
    (-0.05, 60, "put"),
    (0.00,  60, "call"),
]

ORDER_FLOW = dict(
    lambda_noise=8.0,
    max_noise_size=5,
    min_informed_size=3,
    max_informed_size=12,
    staleness_threshold=0.002,
)

QUOTER = dict(
    base_spread=0.05,
    gamma_coeff=2.0,
    vega_coeff=0.002,
    contract_size=100,
)

HEDGER = dict(
    delta_threshold=25.0,
    transaction_cost=0.001,
)

RISK = dict(
    max_gamma=800.0,
    max_vega=50000.0,
    max_contracts_per_leg=20,
)

BACKTEST = dict(
    n_days=30,
    steps_per_day=78,       # ~5-minute bars in a 6.5-hour trading day
    sigma_uncertainty_window=10,
    quote_staleness_steps=2,  # MM sees price 2 steps late
    default_sigma=0.20,
    risk_free_rate=0.02,
    desired_quote_size=5,
)
