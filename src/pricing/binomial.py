import numpy as np


def binomial_price(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: str, n_steps: int = 200) -> float:
    """
    Price an option using the Cox-Ross-Rubinstein (CRR) binomial tree.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
        n_steps: Number of steps in the binomial tree

    Returns:
        Option price
    """
    dt = T / n_steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Terminal asset prices
    j   = np.arange(n_steps + 1)
    ST  = S * (u ** (n_steps - j)) * (d ** j)

    # Terminal payoffs
    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    # Backward induction
    for _ in range(n_steps):
        values = discount * (p * values[:-1] + (1 - p) * values[1:])

    return float(values[0])
