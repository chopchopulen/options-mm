import numpy as np
from scipy.stats import norm


def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return np.inf if S > K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return np.inf if S > K else -np.inf
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
