import numpy as np
import pytest

from src.pricing.vol_surface import HestonVolSurface, heston_implied_vol, bs_implied_vol
from src.pricing.black_scholes import bs_price

# Config parameters. rho < 0 is the leverage effect and must produce negative skew.
P = dict(S_ref=450.0, r=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
T30 = 30 / 365


@pytest.fixture(scope="module")
def surface():
    return HestonVolSurface(**P)


def test_bs_implied_vol_round_trips():
    for sigma in (0.08, 0.20, 0.55, 1.2):
        price = bs_price(450.0, 460.0, T30, 0.02, sigma, "call")
        assert abs(bs_implied_vol(price, 450.0, 460.0, T30, 0.02, "call") - sigma) < 1e-6


def test_bs_implied_vol_returns_nan_below_intrinsic():
    # A price below intrinsic has no implied vol. The previous bisection implementation
    # returned the midpoint of its bracket (~2.5) instead of admitting failure.
    assert np.isnan(bs_implied_vol(0.01, 450.0, 400.0, T30, 0.02, "call"))


def test_skew_has_the_right_sign(surface):
    """rho = -0.7 must make OTM puts richer than OTM calls in vol terms.

    This is the defining feature of an equity index surface and the thing a single flat
    sigma cannot represent at all.
    """
    otm_put = surface.skew_adjustment(450.0, 427.5, T30)
    otm_call = surface.skew_adjustment(450.0, 472.5, T30)
    assert otm_put > 0 > otm_call, f"put {otm_put}, call {otm_call}"
    # Real SPY 30d 95-105 skew runs a few vol points; assert the order of magnitude only.
    assert 0.005 < (otm_put - otm_call) < 0.10


def test_skew_is_zero_at_the_money(surface):
    assert abs(surface.skew_adjustment(450.0, 450.0, T30)) < 1e-9


def test_quoted_iv_equals_the_level_at_the_money(surface):
    for level in (0.12, 0.20, 0.35):
        assert abs(surface.quoted_iv(450.0, 450.0, T30, level) - level) < 1e-9


def test_quoted_iv_carries_the_shape_at_any_level(surface):
    """Shape must be independent of level -- that separation is the module's design."""
    adj = surface.skew_adjustment(450.0, 427.5, T30)
    for level in (0.10, 0.25, 0.40):
        assert abs(surface.quoted_iv(450.0, 427.5, T30, level) - (level + adj)) < 1e-9


def test_skew_reverses_when_rho_is_positive():
    """A falsification check: the skew must come from rho, not from the construction."""
    flipped = HestonVolSurface(**{**P, "rho": +0.7})
    assert flipped.skew_adjustment(450.0, 427.5, T30) < 0
    assert flipped.skew_adjustment(450.0, 472.5, T30) > 0


def test_surface_flattens_to_sqrt_v0_without_vol_of_vol():
    """xi -> 0 and rho = 0 collapses Heston to Black-Scholes: the surface must go flat."""
    flat = HestonVolSurface(**{**P, "xi": 1e-6, "rho": 0.0})
    for K in (405.0, 427.5, 450.0, 472.5, 495.0):
        assert abs(flat.iv(450.0, K, T30) - np.sqrt(P["v0"])) < 5e-3
        assert abs(flat.skew_adjustment(450.0, K, T30)) < 5e-3


def test_interpolation_matches_direct_cf_inversion(surface):
    """The grid is a performance optimisation and must not be a source of error."""
    errors = []
    for K in (405.0, 427.5, 440.0, 450.0, 460.0, 472.5, 495.0):
        for T in (12 / 365, 20 / 365, 33 / 365, 55 / 365):
            direct = heston_implied_vol(450.0, K, T, P["r"], P["v0"], P["kappa"],
                                        P["theta"], P["xi"], P["rho"],
                                        "call" if K >= 450.0 else "put")
            if np.isfinite(direct):
                errors.append(abs(surface.iv(450.0, K, T) - direct))
    assert max(errors) < 2e-3, f"max interpolation error {max(errors):.2e}"


def test_atm_level_is_near_sqrt_v0(surface):
    # v0 == theta here, so the ATM term structure is nearly flat and sits just under
    # sqrt(v0) because of mean reversion and convexity.
    assert abs(surface.atm_iv(T30) - np.sqrt(P["v0"])) < 0.02
    assert abs(surface.describe()["term_slope_30_to_60"]) < 0.01
