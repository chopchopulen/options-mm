"""Implied volatility surface: skew and term structure for quoting in vol space.

WHY THIS EXISTS. Before this module the market maker quoted every leg off a SINGLE
`sigma_implied` — one number for all six strikes and both expiries, estimated from a
10-sample rolling window of REALIZED volatility. That is not options market making. A real
options maker quotes an implied vol SURFACE: its edge is its surface against its forecast,
its inventory is managed in vega and skew, and the flow that hurts it usually knows
something about volatility rather than about spot direction. With one flat sigma there is
no skew to be picked off on, no term structure, and no relative-value risk at all. Strip
the options away and the same model describes a delta-one maker.

DESIGN. The surface separates LEVEL from SHAPE:

    quoted_iv(K, T) = atm_level_estimate(T) + [ surface_iv(K, T) - surface_atm_iv(T) ]
                      \_______ MM's own ______/  \______ shape, relative to ATM ______/

The SHAPE — skew and term structure — is derived from the Heston characteristic function
at the model's own parameters. It contains no fitted constants: rho=-0.7 produces the
negative skew, xi=0.3 produces the smile curvature, and kappa/theta produce the term
structure, all as consequences of dynamics already in the simulator.

The LEVEL remains the market maker's own estimate, so the staleness and adverse-selection
story is preserved: the maker still misjudges where volatility IS, it simply no longer
misjudges the SHAPE of the surface in a way no real maker would.

PERFORMANCE. Inverting the CF per leg per step would be ~14,000 quadrature inversions per
backtest. The shape is precomputed once onto a (log-moneyness x maturity) grid and
interpolated, which is exact to the grid resolution and thousands of times faster. The
shape does not depend on the level estimate, so it never needs recomputing.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

from src.pricing.black_scholes import bs_price
from src.pricing.characteristic_function import heston_price


def bs_implied_vol(price: float, S: float, K: float, T: float, r: float,
                   option_type: str) -> float:
    """Invert Black-Scholes for volatility; NaN where no root exists."""
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-12:
        return np.nan
    try:
        return float(brentq(lambda v: bs_price(S, K, T, r, v, option_type) - price,
                            1e-4, 5.0, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return np.nan


def heston_implied_vol(S: float, K: float, T: float, r: float,
                       v0: float, kappa: float, theta: float, xi: float, rho: float,
                       option_type: str = "call") -> float:
    """Black-Scholes implied vol of the Heston price. The surface's ground truth."""
    price = heston_price(S, K, T, r, v0, kappa, theta, xi, rho, option_type)
    return bs_implied_vol(price, S, K, T, r, option_type)


class HestonVolSurface:
    """Precomputed Heston IV shape, interpolated in (log-moneyness, maturity).

    Exposes the surface RELATIVE to its own ATM level, so a caller can supply its own
    view of the level and inherit only the shape.
    """

    def __init__(self, S_ref: float, r: float, v0: float, kappa: float, theta: float,
                 xi: float, rho: float,
                 log_m_range: tuple = (-0.25, 0.25), n_moneyness: int = 31,
                 T_grid: np.ndarray = None):
        self.S_ref = S_ref
        self.r = r
        self.params = dict(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)

        if T_grid is None:
            # Dense at the short end, where both skew and term structure move fastest.
            T_grid = np.array([1, 3, 5, 7, 10, 14, 18, 21, 25, 30, 35, 40, 45, 52,
                               60, 75, 90, 120, 180], dtype=float) / 365.0
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.log_m_grid = np.linspace(log_m_range[0], log_m_range[1], n_moneyness)

        iv = np.empty((len(self.log_m_grid), len(self.T_grid)))
        for i, lm in enumerate(self.log_m_grid):
            K = S_ref * np.exp(lm)
            # Price the OTM side: it carries the information and avoids the numerically
            # awkward deep-ITM inversion where vega is nearly zero.
            otype = "call" if lm >= 0 else "put"
            for j, T in enumerate(self.T_grid):
                val = heston_implied_vol(S_ref, K, float(T), r, option_type=otype,
                                         **self.params)
                iv[i, j] = val
        # Fill any inversion failures by interpolating along the moneyness axis.
        for j in range(iv.shape[1]):
            col = iv[:, j]
            bad = ~np.isfinite(col)
            if bad.any() and (~bad).sum() >= 2:
                col[bad] = np.interp(self.log_m_grid[bad], self.log_m_grid[~bad], col[~bad])
        if not np.isfinite(iv).all():
            raise ValueError("Heston surface construction produced non-finite implied vols.")

        self._iv = iv
        self._spline = RectBivariateSpline(self.log_m_grid, self.T_grid, iv,
                                           kx=min(3, len(self.log_m_grid) - 1),
                                           ky=min(3, len(self.T_grid) - 1))
        self._atm = np.array([float(self._spline(0.0, T)[0, 0]) for T in self.T_grid])

    # ---- raw surface -------------------------------------------------------------
    def iv(self, S: float, K: float, T: float) -> float:
        """Absolute Heston implied vol at this strike and maturity."""
        lm = float(np.clip(np.log(K / S), self.log_m_grid[0], self.log_m_grid[-1]))
        t = float(np.clip(T, self.T_grid[0], self.T_grid[-1]))
        return float(self._spline(lm, t)[0, 0])

    def atm_iv(self, T: float) -> float:
        t = float(np.clip(T, self.T_grid[0], self.T_grid[-1]))
        return float(self._spline(0.0, t)[0, 0])

    # ---- shape, which is what the quoter consumes --------------------------------
    def skew_adjustment(self, S: float, K: float, T: float) -> float:
        """IV at this strike MINUS the surface's own ATM IV at the same maturity.

        Negative for OTM calls and positive for OTM puts under rho < 0. This is the whole
        point: it is a pure shape, independent of where the level happens to sit.
        """
        return self.iv(S, K, T) - self.atm_iv(T)

    def quoted_iv(self, S: float, K: float, T: float, atm_level: float,
                  floor: float = 0.01) -> float:
        """The maker's quoted vol: its own ATM level, wearing the surface's shape."""
        return max(floor, atm_level + self.skew_adjustment(S, K, T))

    def term_structure_adjustment(self, T: float, T_ref: float) -> float:
        """ATM IV at T minus ATM IV at a reference maturity."""
        return self.atm_iv(T) - self.atm_iv(T_ref)

    def describe(self) -> dict:
        """Summary used by tests and reporting to assert the shape is real."""
        T30, T60 = 30 / 365, 60 / 365
        return {
            "atm_iv_30d": self.atm_iv(T30),
            "atm_iv_60d": self.atm_iv(T60),
            "term_slope_30_to_60": self.atm_iv(T60) - self.atm_iv(T30),
            "skew_30d_95_105": (self.skew_adjustment(100.0, 95.0, T30)
                                - self.skew_adjustment(100.0, 105.0, T30)),
            "otm_put_premium_30d": self.skew_adjustment(100.0, 95.0, T30),
            "otm_call_discount_30d": self.skew_adjustment(100.0, 105.0, T30),
        }
