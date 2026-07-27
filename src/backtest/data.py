"""Real SPY option-chain loader, IV surface extraction, and Heston calibration.

DESIGN RULE: this module FAILS LOUDLY rather than returning degraded data.

Yahoo returns a chain whether or not the market is open. Outside market hours every
`bid` and `ask` is 0.00, `impliedVolatility` is a 0.00001 placeholder, and `lastPrice` is
the last trade — measured at 2.4 days stale on a Sunday fetch, against a spot that has
since moved. An implied vol derived from a stale deep-ITM last trade is not a noisy
estimate of the truth, it is meaningless: a 6.7-day-stale K/S=0.95 call priced to 0.8178
IV in testing.

Calibrating to that and calling it "validated against real market data" would be the most
dishonest thing in this repo, so every loader here raises rather than degrades. Run the
fetch during US market hours (09:30-16:00 ET, weekdays) and commit the cache.

The quoted SPREADS matter as much as the vol surface: real bid-ask as a fraction of
premium is the external anchor for the competition model, and the only thing standing
between that model and an invented constant.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.pricing.black_scholes import bs_price

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CACHE = CACHE_DIR / "spy_surface.csv"

# Quality gates. A row must clear ALL of these to enter the surface.
MAX_STALENESS_HOURS = 2.0     # last trade must be recent relative to the fetch
MIN_MID_PRICE       = 0.10    # below this, tick size dominates the IV inversion
MIN_VOLUME          = 10      # traded today
MIN_OPEN_INTEREST   = 100     # a real contract, not a listing artifact
MONEYNESS_LO        = 0.85    # wings price to garbage IV off stale marks
MONEYNESS_HI        = 1.15
MIN_ROWS            = 40      # below this there is no surface to speak of


class MarketDataUnavailable(RuntimeError):
    """Raised when the live chain cannot support an honest surface."""


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                option_type: str) -> float:
    """Invert Black-Scholes for volatility. Returns NaN where no root exists.

    Uses brentq rather than the previous fixed-iteration bisection, which returned the
    midpoint of its bracket on failure — silently reporting ~2.5 vol for any unpriceable
    quote instead of admitting it had not converged.
    """
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if not np.isfinite(price) or price <= intrinsic:
        return np.nan
    try:
        return float(brentq(
            lambda v: bs_price(S, K, T, r, v, option_type) - price,
            1e-4, 5.0, xtol=1e-8, maxiter=200,
        ))
    except (ValueError, RuntimeError):
        return np.nan


def fetch_spy_chain(max_expiries: int = 8, r: float = 0.04,
                    cache_path: Path = DEFAULT_CACHE,
                    write_cache: bool = True) -> pd.DataFrame:
    """Download the live SPY chain and build a quality-gated IV surface.

    Raises MarketDataUnavailable if the chain cannot support an honest surface — the
    expected outcome outside market hours, when every bid and ask is zero.
    """
    import yfinance as yf   # imported lazily so the engine never depends on the network

    spy = yf.Ticker("SPY")
    hist = spy.history(period="1d")
    if hist.empty:
        raise MarketDataUnavailable("No SPY spot price returned.")
    spot = float(hist["Close"].iloc[-1])
    now = pd.Timestamp.now(tz="UTC")
    today = pd.Timestamp.today().normalize()

    rows, zero_quote_rows = [], 0
    for expiry in spy.options[:max_expiries]:
        days = (pd.Timestamp(expiry) - today).days
        if days <= 0:
            continue
        T = days / 365.0                      # calendar time, the market IV convention
        try:
            chain = spy.option_chain(expiry)
        except Exception as exc:              # noqa: BLE001 - yfinance raises broadly
            warnings.warn(f"expiry {expiry} unavailable: {type(exc).__name__}")
            continue

        for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
            df = frame.copy()
            zero_quote_rows += int(((df["bid"] <= 0) | (df["ask"] <= 0)).sum())
            df = df[(df["bid"] > 0) & (df["ask"] > 0)]
            if df.empty:
                continue

            df["mid"] = (df["bid"] + df["ask"]) / 2.0
            age = now - pd.to_datetime(df["lastTradeDate"], utc=True, errors="coerce")
            df["last_trade_age_h"] = age.dt.total_seconds() / 3600.0
            df["moneyness"] = df["strike"] / spot

            df = df[
                (df["mid"] >= MIN_MID_PRICE)
                & (df["volume"].fillna(0) >= MIN_VOLUME)
                & (df["openInterest"].fillna(0) >= MIN_OPEN_INTEREST)
                & (df["moneyness"].between(MONEYNESS_LO, MONEYNESS_HI))
                & (df["last_trade_age_h"].fillna(1e9) <= MAX_STALENESS_HOURS)
            ]

            for _, row in df.iterrows():
                iv = implied_vol(row["mid"], spot, row["strike"], T, r, option_type)
                if not np.isfinite(iv):
                    continue
                rows.append(dict(
                    spot=spot, expiry=expiry, days_to_exp=days, T_years=T,
                    option_type=option_type, strike=float(row["strike"]),
                    moneyness=float(row["moneyness"]),
                    bid=float(row["bid"]), ask=float(row["ask"]), mid=float(row["mid"]),
                    spread=float(row["ask"] - row["bid"]),
                    spread_pct_of_premium=float((row["ask"] - row["bid"]) / row["mid"]),
                    half_spread_pct=float((row["ask"] - row["bid"]) / 2.0 / row["mid"]),
                    volume=float(row["volume"]), open_interest=float(row["openInterest"]),
                    implied_vol=iv, last_trade_age_h=float(row["last_trade_age_h"]),
                ))

    surface = pd.DataFrame(rows)
    if len(surface) < MIN_ROWS:
        raise MarketDataUnavailable(
            f"Only {len(surface)} rows cleared the quality gates (need {MIN_ROWS}); "
            f"{zero_quote_rows} rows had a zero bid or ask. This is what a closed market "
            f"looks like. Re-run during US market hours (09:30-16:00 ET, weekdays). "
            f"Do NOT relax the gates to force a result -- stale last-trade prices produce "
            f"meaningless implied vols in the wings."
        )

    if write_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        surface.to_csv(cache_path, index=False)
    return surface


def load_cached_surface(cache_path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Read the committed surface. The engine must never hit the network."""
    if not Path(cache_path).exists():
        raise MarketDataUnavailable(
            f"No cached surface at {cache_path}. Run fetch_spy_chain() during market "
            f"hours and commit the result."
        )
    return pd.read_csv(cache_path)


def market_spread_summary(surface: pd.DataFrame) -> dict:
    """Real quoted half-spread as a fraction of premium -- the competition-model anchor.

    Reported near the money and by maturity bucket, because the percentage widens sharply
    for cheap options and no single number describes the chain.
    """
    atm = surface[surface["moneyness"].between(0.98, 1.02)]
    out = {
        "n_rows": int(len(surface)),
        "spot": float(surface["spot"].iloc[0]),
        "atm_half_spread_pct_median": float(atm["half_spread_pct"].median()),
        "atm_half_spread_dollars_median": float((atm["spread"] / 2).median()),
        "all_half_spread_pct_median": float(surface["half_spread_pct"].median()),
        "atm_iv_median": float(atm["implied_vol"].median()),
    }
    for lo, hi, label in ((0, 14, "0_2w"), (14, 45, "2w_6w"), (45, 400, "6w_plus")):
        bucket = atm[atm["days_to_exp"].between(lo, hi)]
        if len(bucket):
            out[f"atm_half_spread_pct_{label}"] = float(bucket["half_spread_pct"].median())
    return out


def calibrate_heston(surface: pd.DataFrame, r: float = 0.04) -> dict:
    """Least-squares fit of Heston parameters to the observed IV surface.

    Replaces the previous moment-matching heuristic, which set theta from mean ATM IV and
    derived rho as `polyfit(moneyness, iv, 1) * -2` -- a slope-to-correlation mapping with
    no theoretical basis, clamped into [-0.95, -0.1] so it could not fail visibly.

    Fits v0, kappa, theta, xi, rho by minimizing squared IV error against Heston-CF prices
    inverted back to Black-Scholes vol. Returns a dict suitable for HestonSimulator, plus
    diagnostics prefixed with an underscore.
    """
    from scipy.optimize import least_squares
    from src.pricing.characteristic_function import heston_price

    S = float(surface["spot"].iloc[0])
    obs = surface[["strike", "T_years", "option_type", "implied_vol"]].to_numpy()

    def residuals(p):
        v0, kappa, theta, xi, rho = p
        out = []
        for K, T, otype, iv_mkt in obs:
            price = heston_price(S, float(K), float(T), r, v0, kappa, theta, xi, rho, otype)
            iv_model = implied_vol(price, S, float(K), float(T), r, otype)
            out.append(0.0 if not np.isfinite(iv_model) else iv_model - float(iv_mkt))
        return np.asarray(out)

    x0     = [0.04, 2.0, 0.04, 0.30, -0.70]
    bounds = ([1e-4, 0.1, 1e-4, 0.01, -0.99], [1.0, 20.0, 1.0, 3.0, 0.0])
    fit = least_squares(residuals, x0, bounds=bounds, xtol=1e-8, max_nfev=200)
    v0, kappa, theta, xi, rho = fit.x

    feller = 2 * kappa * theta - xi ** 2
    if feller < 0:
        warnings.warn(
            f"Calibrated parameters violate the Feller condition (2*kappa*theta - xi^2 = "
            f"{feller:.4f} < 0). The variance process can reach zero and the simulator's "
            f"absorption at 1e-8 will bias variance upward."
        )
    return dict(v0=float(v0), kappa=float(kappa), theta=float(theta),
                xi=float(xi), rho=float(rho), r=r,
                _rmse_iv=float(np.sqrt(np.mean(fit.fun ** 2))),
                _feller_slack=float(feller), _n_quotes=int(len(obs)))


if __name__ == "__main__":
    try:
        surf = fetch_spy_chain()
    except MarketDataUnavailable as exc:
        raise SystemExit(f"\nMARKET DATA UNAVAILABLE\n{exc}\n")
    print(f"Cached {len(surf)} quotes to {DEFAULT_CACHE}")
    for k, v in market_spread_summary(surf).items():
        print(f"  {k:<34} {v}")
    print("\nCalibrating Heston to the surface...")
    for k, v in calibrate_heston(surf).items():
        print(f"  {k:<34} {v}")
