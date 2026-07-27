import numpy as np
from src.mm.quoter import Quoter
from src.pricing.black_scholes import bs_price
from src.greeks.analytical import delta as D, gamma as G, vega as V

# tau = expected inter-arrival at lambda=8/day over 78 steps, in years.
TAU = (78 / 8.0) / (252 * 78)
XI  = 0.30
FLOW = dict(staleness_steps=2, staleness_threshold=0.002, dt_step=1 / (252 * 78),
            steps_per_day=78, lambda_noise=8.0, mean_informed_size=7.5,
            mean_noise_size=3.0)
NO_FLOW = dict(FLOW, staleness_threshold=1e9)   # informed never arrive -> zero charge


def _q(base_spread=0.05, flow=None):
    return Quoter(base_spread=base_spread, holding_horizon=TAU, vol_of_vol=XI,
                  flow=FLOW if flow is None else flow, contract_size=100)


def test_bid_below_ask():
    bid, ask = _q().quote(fair_value=5.0, gamma=0.02, vega=10.0, delta=0.5, S=450.0, sigma=0.20)
    assert bid < ask


def test_symmetric_around_fair():
    # Documents the ABSENCE of inventory skew. The quoter is not passed the position, so
    # it cannot lean against inventory (audit finding 17, quote-formation half). If skew
    # is added this test must change -- it pins current behaviour, not desired behaviour.
    bid, ask = _q().quote(fair_value=5.0, gamma=0.02, vega=10.0, delta=0.5, S=450.0, sigma=0.20)
    assert abs((bid + ask) / 2 - 5.0) < 1e-10


def test_wider_with_higher_gamma():
    lo = _q().half_spread(gamma=0.01, vega=0.0, delta=0.0, S=450.0, sigma=0.20)
    hi = _q().half_spread(gamma=0.10, vega=0.0, delta=0.0, S=450.0, sigma=0.20)
    assert hi > lo


def test_wider_with_higher_vega():
    lo = _q().half_spread(gamma=0.0, vega=10.0, delta=0.0, S=450.0, sigma=0.20)
    hi = _q().half_spread(gamma=0.0, vega=100.0, delta=0.0, S=450.0, sigma=0.20)
    assert hi > lo


def test_gamma_charge_equals_carry_cost_over_tau():
    """The gamma loading must BE the gamma cost over tau, not a fitted coefficient."""
    g, S, sigma = 0.0128, 450.0, 0.20
    charge = _q(base_spread=0.0).half_spread(gamma=g, vega=0.0, delta=0.0, S=S, sigma=sigma)
    assert abs(charge - 0.5 * g * S**2 * sigma**2 * TAU) < 1e-12


def test_vega_charge_equals_heston_vol_move_over_tau():
    """dv = xi sqrt(v) dW  =>  dsigma over tau = (xi/2) sqrt(tau)."""
    v = 61.79
    charge = _q(base_spread=0.0).half_spread(gamma=0.0, vega=v, delta=0.0, S=450.0, sigma=0.20)
    assert abs(charge - v * (XI / 2.0) * np.sqrt(TAU)) < 1e-12


def test_half_spread_is_low_single_digit_percent_of_premium():
    """Regression guard on the defect this replaced: 10-51% of premium.

    Deliberately loose. It asserts the order of magnitude is sane, not a target value --
    a tight band here would be a P&L tuning knob wearing a test's clothing.
    """
    S, r, sigma = 450.0, 0.02, 0.20
    for K, days, otype in [(427.5, 30, "put"), (450.0, 30, "call"), (450.0, 30, "put"),
                           (472.5, 30, "call"), (427.5, 60, "put"), (450.0, 60, "call")]:
        T = days / 252
        premium = bs_price(S, K, T, r, sigma, otype)
        hs = _q().half_spread(G(S, K, T, r, sigma), V(S, K, T, r, sigma),
                              D(S, K, T, r, sigma, otype), S, sigma)
        pct = hs / premium
        assert pct < 0.20, f"{K} {days}d {otype}: half-spread is {pct:.1%} of premium"


def test_adverse_selection_charge_is_zero_without_informed_flow():
    """No informed arrivals -> no adverse-selection charge. Guards the derivation."""
    q = _q(base_spread=0.0, flow=NO_FLOW)
    assert q.adverse_selection_charge(delta=0.5, S=450.0, sigma=0.20) < 1e-9


def test_adverse_selection_charge_scales_with_delta():
    q = _q(base_spread=0.0)
    lo = q.adverse_selection_charge(delta=0.10, S=450.0, sigma=0.20)
    hi = q.adverse_selection_charge(delta=0.50, S=450.0, sigma=0.20)
    assert hi > lo > 0
    assert abs(hi / lo - 5.0) < 1e-9, "charge is linear in |delta|"


def test_informed_share_matches_closed_form():
    """The derived informed share must reproduce the hand calculation, ~0.887."""
    import scipy.stats as st
    s_m = 0.20 * np.sqrt(2 * (1 / (252 * 78)))
    k = 0.002 / s_m
    tail = 1 - st.norm.cdf(k)
    inf_vol = 2 * tail * 78 * 7.5
    share = inf_vol / (inf_vol + 8.0 * 3.0)
    assert abs(share - 0.8868) < 1e-3

    q = _q(base_spread=0.0)
    e_move = s_m * st.norm.pdf(k) / tail
    assert abs(q.adverse_selection_charge(1.0, 450.0, 0.20) - 450.0 * e_move * share) < 1e-12
