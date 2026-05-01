from src.pricing.black_scholes import bs_price


def delta_fd(S: float, K: float, T: float, r: float, sigma: float, option_type: str, h: float = 0.01) -> float:
    up   = bs_price(S + h, K, T, r, sigma, option_type)
    down = bs_price(S - h, K, T, r, sigma, option_type)
    return (up - down) / (2 * h)


def gamma_fd(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", h: float = 0.01) -> float:
    up     = bs_price(S + h, K, T, r, sigma, option_type)
    mid    = bs_price(S,     K, T, r, sigma, option_type)
    down   = bs_price(S - h, K, T, r, sigma, option_type)
    return (up - 2 * mid + down) / h**2


def vega_fd(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", h: float = 1e-4) -> float:
    up   = bs_price(S, K, T, r, sigma + h, option_type)
    down = bs_price(S, K, T, r, sigma - h, option_type)
    return (up - down) / (2 * h)


def theta_fd(S: float, K: float, T: float, r: float, sigma: float, option_type: str, h: float = 1 / 365) -> float:
    if T <= h:
        return 0.0
    up   = bs_price(S, K, T,     r, sigma, option_type)
    down = bs_price(S, K, T - h, r, sigma, option_type)
    return (down - up) / h


def vanna_fd(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call",
             h_S: float = 0.01, h_sigma: float = 1e-4) -> float:
    up   = delta_fd(S, K, T, r, sigma + h_sigma, option_type, h=h_S)
    down = delta_fd(S, K, T, r, sigma - h_sigma, option_type, h=h_S)
    return (up - down) / (2 * h_sigma)


def volga_fd(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call",
             h: float = 1e-4) -> float:
    up   = bs_price(S, K, T, r, sigma + h, option_type)
    mid  = bs_price(S, K, T, r, sigma,     option_type)
    down = bs_price(S, K, T, r, sigma - h, option_type)
    return (up - 2 * mid + down) / h**2
