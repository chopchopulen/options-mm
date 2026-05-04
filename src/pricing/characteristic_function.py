import numpy as np
from scipy import integrate, optimize


def _heston_cf(u, S, T, r, v0, kappa, theta, xi, rho):
    """
    Heston (1993) characteristic function, little-trap formulation (Albrecher et al. 2007).

    The trap avoids Riemann sheet discontinuities by choosing log_arg = (r_m*h - r_p)/(-2d)
    instead of the original (1 - g*h)/(1-g). Mathematically equivalent, numerically stable
    for all u in [0, inf) without computing exp(+dT).
    """
    iu  = 1j * u
    b   = kappa - rho * xi * iu
    d   = np.sqrt(b**2 + xi**2 * (u**2 + iu))
    r_m = b - d
    r_p = b + d
    h   = np.exp(-d * T)
    log_arg = (r_m * h - r_p) / (-2 * d)
    A = iu * (np.log(S) + r * T) + kappa * theta / xi**2 * (r_m * T - 2 * np.log(log_arg))
    B = r_p / xi**2 * (h - 1) * r_m / (r_m * h - r_p)
    return np.exp(A + B * v0)


def heston_price(S: float, K: float, T: float, r: float,
                 v0: float, kappa: float, theta: float, xi: float, rho: float,
                 option_type: str) -> float:
    """
    Price a European option under the Heston stochastic volatility model
    using Gil-Pelaez Fourier inversion.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        v0: Initial variance (sigma^2 at t=0)
        kappa: Mean-reversion speed of variance
        theta: Long-run variance (sigma^2 long-run mean)
        xi: Vol-of-vol (volatility of variance process)
        rho: Correlation between spot and variance Brownian motions
        option_type: "call" or "put"

    Returns:
        Option price
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    F   = S * np.exp(r * T)
    lnK = np.log(K)

    def integrand_P1(u):
        phi = _heston_cf(u - 1j, S, T, r, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-1j * u * lnK) * phi / (1j * u * F))

    def integrand_P2(u):
        phi = _heston_cf(u, S, T, r, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-1j * u * lnK) * phi / (1j * u))

    P1 = 0.5 + 1 / np.pi * integrate.quad(integrand_P1, 1e-10, 500, limit=500, epsabs=1e-9)[0]
    P2 = 0.5 + 1 / np.pi * integrate.quad(integrand_P2, 1e-10, 500, limit=500, epsabs=1e-9)[0]

    call = np.exp(-r * T) * (F * P1 - K * P2)
    if option_type == "call":
        return float(call)
    # Put via put-call parity (avoids second integration)
    return float(call - (S - K * np.exp(-r * T)))
