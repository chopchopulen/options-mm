import numpy as np
from scipy import integrate


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


def heston_price_grid(S: float, strikes: np.ndarray, T: float, r: float,
                      v0: float, kappa: float, theta: float, xi: float,
                      rho: float) -> np.ndarray:
    """
    Price European calls across a full strike grid under Heston using Carr-Madan FFT.

    Returns call prices. Convert to puts via: put = call - (S - K*exp(-rT)).

    Args:
        S: Spot price
        strikes: 1-D array of strike prices
        T: Time to maturity in years
        r: Risk-free rate
        v0, kappa, theta, xi, rho: Heston parameters (same as heston_price)

    Returns:
        1-D numpy array of call prices, same length as strikes.
    """
    from scipy.interpolate import CubicSpline

    strikes = np.asarray(strikes, dtype=float)
    N     = 4096
    eta   = 0.25
    alpha = 1.5
    lam   = 2 * np.pi / (N * eta)
    b     = N * lam / 2

    j   = np.arange(N)
    u_j = j * eta

    denom = alpha**2 + alpha - u_j**2 + 1j * (2 * alpha + 1) * u_j
    phi_u = _heston_cf(u_j - 1j * (alpha + 1), S, T, r, v0, kappa, theta, xi, rho)
    psi   = np.exp(-r * T) * phi_u / denom

    w      = np.ones(N)
    w[1::2] = 4
    w[2::2] = 2
    w[-1]   = 1
    w      *= eta / 3

    x       = np.exp(-1j * b * u_j) * psi * w
    fft_out = np.real(np.fft.fft(x))

    k_m    = -b + j * lam
    call_m = np.exp(-alpha * k_m) / np.pi * fft_out

    ln_strikes = np.log(strikes)
    cs    = CubicSpline(k_m, call_m)
    calls = cs(ln_strikes)

    # Call price lower bound: max(S - K*exp(-rT), 0) — clip numerical negatives
    lower = np.maximum(S - strikes * np.exp(-r * T), 0.0)
    calls = np.maximum(calls, lower)
    return calls
