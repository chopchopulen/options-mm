from typing import List, Dict
from src.greeks.analytical import delta, gamma, vega, theta


def portfolio_greeks(positions: List[Dict], contract_size: int = 100) -> Dict[str, float]:
    total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for pos in positions:
        S, K, T, r, sigma = pos["S"], pos["K"], pos["T"], pos["r"], pos["sigma"]
        ot = pos["option_type"]
        q  = pos["quantity"] * contract_size
        total["delta"] += delta(S, K, T, r, sigma, ot) * q
        total["gamma"] += gamma(S, K, T, r, sigma) * q
        total["vega"]  += vega(S, K, T, r, sigma) * q
        total["theta"] += theta(S, K, T, r, sigma, ot) * q
    return total
