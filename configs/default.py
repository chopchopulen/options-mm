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
    # base_spread is a per-share half-spread. On the ATM 30d leg (~$12.91 premium) it is
    # 0.4% of premium, which is roughly the real SPY NBBO half-width — this parameter was
    # already correctly scaled; the coefficients that used to sit beside it were not.
    # The gamma and vega loadings are no longer free coefficients: they are derived in
    # the engine as the cost of carrying inventory over the holding horizon tau.
    base_spread=0.05,
    contract_size=100,
)

HEDGER = dict(
    delta_threshold=25.0,
    transaction_cost=0.001,
)

RISK = dict(
    # The POSITION LIMIT is the binding risk control for this book. The aggregate Greek
    # caps below are set at 100% of declared capacity, so for THIS universe they are
    # redundant with it — but not redundant in principle: they are derived from this
    # book's composition (6 legs, 30/60d, sigma=0.20). A longer-dated or larger universe
    # pushes per-contract vega up (60d ATM already carries 8,718 against 6,179 at 30d),
    # and the aggregate cap would then bind before the per-leg limit does. No fraction of
    # capacity is invented here; choosing one would require a capital base or risk budget
    # that does not exist in this repo.
    #
    # Greek caps are DERIVED from max_contracts_per_leg, not chosen independently.
    # The book's declared capacity is 20 contracts x 6 legs = 120 contracts. At
    # sigma=0.20 the six legs carry, per contract (vega/gamma x contract_size=100):
    #   427.5 30d put  4453.4 / 0.924      450.0 30d call 6179.4 / 1.282
    #   450.0 30d put  6179.4 / 1.282      472.5 30d call 5053.4 / 1.048
    #   427.5 60d put  7213.8 / 0.748      450.0 60d call 8718.3 / 0.904
    #   sum over legs  37797.8 / 6.187
    # A full book at declared capacity therefore carries |vega| 755,957 and
    # |gamma| 123.7. The previous values were mutually inconsistent by ~100x:
    # max_vega=50,000 bound at 6.6% of declared capacity while max_gamma=800
    # bound at 646% of it, so vega refused quotes the position limit permitted
    # and gamma never bound at all.
    # Cross-check: Heston xi=0.30 implies a 1-day 1-sigma move of 0.94 vol points,
    # costing a full-capacity book $7,143 in vega, and a 5.67-point spot move
    # costing $1,988 in gamma. Both are coherent magnitudes for this book.
    max_gamma=124.0,
    max_vega=756000.0,
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
