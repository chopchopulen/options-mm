import numpy as np
from typing import Tuple


class Quoter:
    """Risk-loaded symmetric quoter.

    The half-spread is the cost of CARRYING the position over the expected holding
    horizon, not an arbitrary coefficient times a Greek:

        half_spread = base_spread
                    + 0.5 * |gamma| * S^2 * sigma^2 * tau      (gamma cost over tau)
                    + |vega| * (xi / 2) * sqrt(tau)            (vol move over tau)

    The vega term follows from the Heston variance SDE: dv = xi * sqrt(v) dW, so the
    standard deviation of dv over tau is xi * sqrt(v) * sqrt(tau), and since
    sigma = sqrt(v), dsigma = dv / (2 sigma) = (xi / 2) * sqrt(tau).

    tau is the INVENTORY HOLDING HORIZON, not the hedge interval. Delta hedging every
    step removes delta risk only. Gamma and vega exposure persist for the life of the
    option position regardless of how often delta is hedged: the expected gamma cost over
    a holding period is set by realized variance over that period, and vega is untouched
    by delta hedging entirely. So tau is how long the position is held, not how often it
    is re-hedged.

    Every term is per share, matching fair_value. The previous form,
    `gamma_coeff * |gamma| * contract_size`, built a dollar charge out of a dimensionless
    second derivative with no position size, horizon or vol in it, and produced
    half-spreads of 10-51% of premium.
    """

    def __init__(self, base_spread: float, holding_horizon: float,
                 vol_of_vol: float, contract_size: int = 100):
        self.base_spread     = base_spread
        self.holding_horizon = holding_horizon   # tau, in years
        self.vol_of_vol      = vol_of_vol        # xi
        self.contract_size   = contract_size

    def half_spread(self, gamma: float, vega: float, S: float, sigma: float) -> float:
        tau        = self.holding_horizon
        gamma_cost = 0.5 * abs(gamma) * S**2 * sigma**2 * tau
        vega_cost  = abs(vega) * (self.vol_of_vol / 2.0) * np.sqrt(tau)
        return self.base_spread + gamma_cost + vega_cost

    def quote(self, fair_value: float, gamma: float, vega: float,
              S: float, sigma: float) -> Tuple[float, float]:
        hs = self.half_spread(gamma, vega, S, sigma)
        return fair_value - hs, fair_value + hs
