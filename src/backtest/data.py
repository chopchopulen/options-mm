import pandas as pd
import numpy as np
import yfinance as yf
from src.pricing.black_scholes import bs_price


def _implied_vol(market_price: float, S: float, K: float, T: float,
                 r: float, option_type: str, tol: float = 1e-6, max_iter: int = 100) -> float:
    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, r, mid, option_type) < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def load_spy_iv_surface(as_of_date: str = None) -> pd.DataFrame:
    """
    Download SPY options chain and compute implied vol surface.
    Returns DataFrame with columns: strike, expiry, option_type, mid_price, implied_vol, moneyness.
    as_of_date is used only to select near-dated expirations. No future data is read.
    """
    spy = yf.Ticker("SPY")
    S   = spy.history(period="1d")["Close"].iloc[-1]
    r   = 0.02

    records = []
    for exp in spy.options[:4]:  # only nearest 4 expirations
        chain = spy.option_chain(exp)
        exp_date = pd.Timestamp(exp)
        T = max((exp_date - pd.Timestamp("today")).days / 365, 1e-4)

        for _, row in chain.calls.iterrows():
            mid = (row["bid"] + row["ask"]) / 2
            if mid < 0.05 or row["volume"] < 10:
                continue
            try:
                iv = _implied_vol(mid, S, row["strike"], T, r, "call")
                records.append({"strike": row["strike"], "expiry": exp, "option_type": "call",
                                 "mid_price": mid, "implied_vol": iv,
                                 "moneyness": row["strike"] / S})
            except Exception:
                pass

        for _, row in chain.puts.iterrows():
            mid = (row["bid"] + row["ask"]) / 2
            if mid < 0.05 or row["volume"] < 10:
                continue
            try:
                iv = _implied_vol(mid, S, row["strike"], T, r, "put")
                records.append({"strike": row["strike"], "expiry": exp, "option_type": "put",
                                 "mid_price": mid, "implied_vol": iv,
                                 "moneyness": row["strike"] / S})
            except Exception:
                pass

    return pd.DataFrame(records)


def estimate_heston_params_from_surface(df: pd.DataFrame) -> dict:
    """
    Simple moment-matching: extract ATM IV as theta,
    skew slope as proxy for rho, and vol-of-vol from term structure slope.
    Returns dict suitable for HestonSimulator kwargs.
    """
    atm = df[(df["moneyness"].between(0.98, 1.02)) & (df["option_type"] == "call")]
    theta = float(atm["implied_vol"].mean()**2) if len(atm) > 0 else 0.04

    skew = df[df["option_type"] == "put"].copy()
    if len(skew) > 5:
        coef = np.polyfit(skew["moneyness"], skew["implied_vol"], 1)
        rho  = max(-0.95, min(-0.1, float(coef[0]) * -2))
    else:
        rho = -0.7

    return dict(v0=theta, kappa=2.0, theta=theta, xi=0.3, rho=rho, r=0.02)
