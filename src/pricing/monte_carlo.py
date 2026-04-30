import numpy as np


def mc_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str, n_paths: int = 100_000,
             rng: np.random.Generator = None, antithetic: bool = True) -> float:
    """
    Price an option using Monte Carlo simulation with optional antithetic variates.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
        n_paths: Number of simulation paths
        rng: numpy random generator (if None, creates new default_rng())
        antithetic: Whether to use antithetic variates for variance reduction

    Returns:
        Option price
    """
    if rng is None:
        rng = np.random.default_rng()

    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(half)
    drift = (r - 0.5 * sigma**2) * T
    vol   = sigma * np.sqrt(T)
    ST_pos = S * np.exp(drift + vol * z)

    if antithetic:
        ST_neg = S * np.exp(drift - vol * z)
        ST = np.concatenate([ST_pos, ST_neg])
    else:
        ST = ST_pos

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return float(np.exp(-r * T) * np.mean(payoffs))
