import numpy as np
from scipy.optimize import brentq
from src.pricing.black_scholes import bs_price
from src.pricing.characteristic_function import heston_price

# Shared Heston params matching the simulator's default config
HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
S, K, T, r = 100.0, 100.0, 1.0, 0.05


def test_heston_cf_reduces_to_bs_when_xi_zero():
    # When xi (vol-of-vol) → 0, Heston reduces to BS with sigma = sqrt(v0) = 0.2
    heston_p = heston_price(S, K, T, r, v0=0.04, kappa=2.0, theta=0.04, xi=1e-6, rho=0.0, option_type="call")
    bs_p     = bs_price(S, K, T, r, sigma=0.2, option_type="call")
    assert abs(heston_p - bs_p) < 0.01


def test_put_matches_bs_in_zero_volvol_limit():
    """The put branch needs an INDEPENDENT check, not a parity check.

    heston_price computes the put as `call - (S - K*exp(-rT))`, so asserting
    put-call parity on it is a tautology: it holds by construction for any call price,
    including a wrong one. The previous test did exactly that.

    Driving xi -> 0 with rho = 0 collapses Heston to Black-Scholes at sigma = sqrt(v0),
    where the put has a closed form derived by a completely separate route.
    """
    heston_put = heston_price(S, K, T, r, v0=0.04, kappa=2.0, theta=0.04,
                              xi=1e-6, rho=0.0, option_type="put")
    bs_put     = bs_price(S, K, T, r, sigma=0.2, option_type="put")
    assert abs(heston_put - bs_put) < 0.01, f"Heston put {heston_put} vs BS put {bs_put}"


def test_put_call_parity_holds_by_construction():
    # Documents, rather than tests, the parity relation: it is how the put is computed.
    # Kept so the construction is explicit and a future refactor to an independent put
    # integration is still required to satisfy it.
    call = heston_price(S, K, T, r, **HESTON, option_type="call")
    put  = heston_price(S, K, T, r, **HESTON, option_type="put")
    assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-6


def test_prices_positive_finite():
    # All 27 grid points must be positive and finite
    spot_pcts = np.arange(0.80, 1.21, 0.05)
    expiries  = [30, 60, 90]
    for T_days in expiries:
        T_yr = T_days / 252
        for pct in spot_pcts:
            K_i   = S * pct
            otype = "put" if pct <= 1.0 else "call"
            p = heston_price(S, K_i, T_yr, r, **HESTON, option_type=otype)
            assert np.isfinite(p) and p > 0, f"Bad price at K={K_i}, T={T_days}d: {p}"


def test_vol_smile_negative_skew():
    # rho=-0.7 → negative skew → OTM put IV > ATM IV > OTM call IV
    T_yr = 1.0

    def implied_vol(price, K_iv, otype):
        def obj(s):
            return bs_price(S, K_iv, T_yr, r, s, otype) - price
        return brentq(obj, 0.001, 5.0, xtol=1e-8)

    call_atm  = heston_price(S, S * 1.00, T_yr, r, **HESTON, option_type="call")
    call_otmc = heston_price(S, S * 1.15, T_yr, r, **HESTON, option_type="call")
    put_otmp  = heston_price(S, S * 0.85, T_yr, r, **HESTON, option_type="put")

    iv_atm   = implied_vol(call_atm,  S * 1.00, "call")
    iv_otmc  = implied_vol(call_otmc, S * 1.15, "call")
    iv_otmp  = implied_vol(put_otmp,  S * 0.85, "put")

    assert iv_otmp > iv_atm > iv_otmc, (
        f"Expected negative skew: IV(85%)={iv_otmp:.4f} > IV(100%)={iv_atm:.4f} "
        f"> IV(115%)={iv_otmc:.4f}"
    )
