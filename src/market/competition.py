"""Competing quotes: the market maker is not the only liquidity provider.

WHY THIS EXISTS. Without competition, fill probability does not depend on quote width at
all. Measured directly: sweeping `base_spread` over a 16x range produced 5,335 fills and
19,388 contracts at EVERY width, identical to the unit, with spread capture exactly linear
in the spread. P&L was therefore unbounded and linear in a quantity the maker chooses --
quote wider, earn more, forever. That is not a market, and it means the P&L LEVEL is not
identified by anything in the model.

THE ANCHOR PROBLEM. A competition model can easily reintroduce the same defect one layer
deeper: invent a competitor spread, and the P&L level is still something we chose, just
laundered. The escape is that real quoted spreads are observable. This module samples the
competing half-spread from the EMPIRICAL distribution of SPY quotes -- the actual bid-ask
as a fraction of premium, bucketed by moneyness and maturity -- so the competitive level
is measured rather than assumed. There is no dispersion parameter to tune: the dispersion
IS the observed distribution.

MECHANISM. Each quote request draws a competing half-spread for that bucket. The maker
wins the flow only if its own quote is at least as tight. This gates ALL flow, informed
and noise alike, because a counterparty takes the best price available regardless of why
it is trading. Quote too wide and the flow goes elsewhere; quote too tight and there is
nothing left to earn. That is what produces an interior optimum in width, and it is why
the level becomes identified.

REQUIRES REAL DATA. Build it from a cached SPY surface produced during market hours by
`src.backtest.data.fetch_spy_chain`. Absent that cache this module raises rather than
falling back to an invented distribution.
"""

import numpy as np
import pandas as pd

# Bucket edges. Moneyness bins are tighter near the money, where most volume sits and
# where the percentage spread changes fastest.
MONEYNESS_EDGES = np.array([0.00, 0.90, 0.96, 0.99, 1.01, 1.04, 1.10, np.inf])
MATURITY_EDGES  = np.array([0, 7, 21, 45, 90, np.inf])
MIN_BUCKET_ROWS = 5


class CompetitiveQuotes:
    """Empirical distribution of competing half-spreads, as a fraction of premium."""

    def __init__(self, buckets: dict, fallback: np.ndarray, source: str):
        if not buckets and fallback.size == 0:
            raise ValueError("CompetitiveQuotes built with no observations.")
        self._buckets = buckets
        self._fallback = fallback
        self.source = source

    # ---- construction -------------------------------------------------------------
    @classmethod
    def from_surface(cls, surface: pd.DataFrame, source: str = "cached SPY surface"):
        """Bucket observed half-spread percentages by moneyness and maturity."""
        required = {"moneyness", "days_to_exp", "half_spread_pct"}
        missing = required - set(surface.columns)
        if missing:
            raise ValueError(f"surface is missing columns: {sorted(missing)}")

        df = surface[np.isfinite(surface["half_spread_pct"])
                     & (surface["half_spread_pct"] > 0)]
        m_idx = np.digitize(df["moneyness"], MONEYNESS_EDGES) - 1
        t_idx = np.digitize(df["days_to_exp"], MATURITY_EDGES) - 1

        buckets = {}
        for (mi, ti), grp in df.groupby([m_idx, t_idx]):
            if len(grp) >= MIN_BUCKET_ROWS:
                buckets[(int(mi), int(ti))] = grp["half_spread_pct"].to_numpy()
        return cls(buckets, df["half_spread_pct"].to_numpy(), source)

    @classmethod
    def from_cache(cls, cache_path=None):
        """Load the committed SPY surface. Raises if it does not exist."""
        from src.backtest.data import load_cached_surface, DEFAULT_CACHE
        path = cache_path or DEFAULT_CACHE
        return cls.from_surface(load_cached_surface(path), source=str(path))

    # ---- sampling -----------------------------------------------------------------
    def _bucket(self, moneyness: float, days_to_exp: float) -> np.ndarray:
        key = (int(np.digitize(moneyness, MONEYNESS_EDGES) - 1),
               int(np.digitize(days_to_exp, MATURITY_EDGES) - 1))
        return self._buckets.get(key, self._fallback)

    def draw_half_spread_pct(self, moneyness: float, days_to_exp: float,
                             rng: np.random.Generator) -> float:
        """Sample a competing half-spread as a fraction of premium.

        Sampling from the empirical values rather than fitting a distribution to them:
        any parametric fit would introduce shape assumptions the data does not require.
        """
        obs = self._bucket(moneyness, days_to_exp)
        return float(obs[rng.integers(0, len(obs))])

    def wins_flow(self, own_half_spread: float, premium: float, moneyness: float,
                  days_to_exp: float, rng: np.random.Generator) -> bool:
        """True if the maker's quote is at least as tight as the competing one.

        A counterparty takes the best price available, so this gates informed and noise
        flow alike.
        """
        if premium <= 0:
            return False
        own_pct = own_half_spread / premium
        return own_pct <= self.draw_half_spread_pct(moneyness, days_to_exp, rng)

    # ---- reporting ----------------------------------------------------------------
    def describe(self) -> dict:
        allobs = (np.concatenate(list(self._buckets.values()))
                  if self._buckets else self._fallback)
        return {
            "source": self.source,
            "n_buckets": len(self._buckets),
            "n_observations": int(allobs.size),
            "half_spread_pct_median": float(np.median(allobs)),
            "half_spread_pct_p10": float(np.percentile(allobs, 10)),
            "half_spread_pct_p90": float(np.percentile(allobs, 90)),
        }
