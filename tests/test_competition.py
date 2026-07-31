"""Competition model tests.

These use a SYNTHETIC surface on purpose. The production path requires a real cached SPY
chain, but the mechanism must be testable without a network call or market hours — and a
synthetic distribution here is a test fixture, not a calibration.
"""

import numpy as np
import pandas as pd
import pytest

from src.market.competition import CompetitiveQuotes, MONEYNESS_EDGES


def synthetic_surface(n=600, seed=0):
    rng = np.random.default_rng(seed)
    moneyness = rng.uniform(0.88, 1.12, n)
    days = rng.choice([3, 14, 30, 60, 120], n)
    # Wider in percentage terms away from the money, as real chains are.
    base = 0.004 + 0.05 * np.abs(moneyness - 1.0)
    return pd.DataFrame({
        "moneyness": moneyness,
        "days_to_exp": days,
        "half_spread_pct": base * rng.lognormal(0.0, 0.25, n),
    })


@pytest.fixture(scope="module")
def comp():
    return CompetitiveQuotes.from_surface(synthetic_surface())


def test_rejects_a_surface_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        CompetitiveQuotes.from_surface(pd.DataFrame({"moneyness": [1.0]}))


def test_draws_come_from_the_observed_values(comp):
    """Sampling must be empirical -- every draw is an observation, not a fitted value."""
    surface = synthetic_surface()
    observed = set(np.round(surface["half_spread_pct"].to_numpy(), 12))
    rng = np.random.default_rng(1)
    for _ in range(200):
        assert round(comp.draw_half_spread_pct(1.0, 30, rng), 12) in observed


def test_a_tighter_quote_wins_more_flow(comp):
    """The property the whole module exists for: fill probability falls with width."""
    def win_rate(half_spread_pct):
        rng = np.random.default_rng(7)
        premium = 10.0
        return np.mean([comp.wins_flow(half_spread_pct * premium, premium, 1.0, 30, rng)
                        for _ in range(4000)])

    tight, mid, wide = win_rate(0.001), win_rate(0.005), win_rate(0.05)
    assert tight > mid > wide, f"{tight:.3f} {mid:.3f} {wide:.3f}"
    assert tight > 0.9, "an extremely tight quote should win nearly all flow"
    assert wide < 0.1, "an extremely wide quote should win almost none"


def test_zero_premium_never_wins(comp):
    rng = np.random.default_rng(3)
    assert comp.wins_flow(0.01, 0.0, 1.0, 30, rng) is False


def test_buckets_separate_by_moneyness(comp):
    """Wings must be sampled from wing observations, not from the ATM bucket."""
    rng = np.random.default_rng(11)
    atm = np.mean([comp.draw_half_spread_pct(1.00, 30, rng) for _ in range(2000)])
    wing = np.mean([comp.draw_half_spread_pct(1.11, 30, rng) for _ in range(2000)])
    assert wing > atm * 1.5, f"atm {atm:.5f} vs wing {wing:.5f}"


def test_unseen_bucket_falls_back_rather_than_raising(comp):
    rng = np.random.default_rng(5)
    assert comp.draw_half_spread_pct(3.0, 9999, rng) > 0


def test_describe_reports_the_distribution(comp):
    d = comp.describe()
    assert d["n_observations"] > 0 and d["n_buckets"] > 0
    assert d["half_spread_pct_p10"] < d["half_spread_pct_median"] < d["half_spread_pct_p90"]


def test_from_cache_raises_without_real_data(tmp_path):
    """The production path must never silently invent a competitive level."""
    from src.backtest.data import MarketDataUnavailable
    with pytest.raises(MarketDataUnavailable):
        CompetitiveQuotes.from_cache(tmp_path / "does_not_exist.csv")
