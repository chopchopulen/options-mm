import numpy as np
from scipy.stats import norm
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
                 vol_of_vol: float, flow: dict, contract_size: int = 100):
        self.base_spread     = base_spread
        self.holding_horizon = holding_horizon   # tau, in years
        self.vol_of_vol      = vol_of_vol        # xi
        self.flow            = flow              # structural order-flow constants
        self.contract_size   = contract_size

    def adverse_selection_charge(self, delta: float, S: float, sigma: float) -> float:
        """Glosten-Milgrom break-even charge for informed flow.

        The maker pays the informed trader's edge on informed volume and collects the
        half-spread on ALL volume, so it breaks even at

            h_adverse = E[edge | informed] * (informed share of volume)

        Every input is a structural property of the order-flow model, not a fitted
        constant:

        Informed arrival. An informed trader appears when the underlying has moved more
        than `threshold` over the staleness lag. Over `staleness` steps the log return has
        standard deviation s_m = sigma * sqrt(staleness * dt), so with k = threshold / s_m

            P(informed)          = 2 * (1 - Phi(k))
            E[|m| | |m| > thr]   = s_m * phi(k) / (1 - Phi(k))        (truncated normal)

        Edge in option terms. A move dS is worth |delta| * dS to the option holder, to
        first order — which is exactly the order at which this trader operates, since the
        trigger is a small underlying move.

        Informed share. Informed volume per day per leg is P(informed) * steps_per_day *
        E[informed size]; noise volume is lambda_noise * E[noise size].
        """
        f     = self.flow
        s_m   = sigma * np.sqrt(f["staleness_steps"] * f["dt_step"])
        k     = f["staleness_threshold"] / s_m
        tail  = 1.0 - norm.cdf(k)
        if tail <= 0.0:
            return 0.0
        e_move = s_m * norm.pdf(k) / tail          # E[|m| | |m| > threshold]
        p_inf  = 2.0 * tail

        informed_vol = p_inf * f["steps_per_day"] * f["mean_informed_size"]
        noise_vol    = f["lambda_noise"] * f["mean_noise_size"]
        total_vol    = informed_vol + noise_vol
        if total_vol <= 0.0:
            return 0.0
        informed_share = informed_vol / total_vol

        return abs(delta) * S * e_move * informed_share

    def half_spread(self, gamma: float, vega: float, delta: float,
                    S: float, sigma: float) -> float:
        tau        = self.holding_horizon
        gamma_cost = 0.5 * abs(gamma) * S**2 * sigma**2 * tau
        vega_cost  = abs(vega) * (self.vol_of_vol / 2.0) * np.sqrt(tau)
        adverse    = self.adverse_selection_charge(delta, S, sigma)
        return self.base_spread + gamma_cost + vega_cost + adverse

    def quote(self, fair_value: float, gamma: float, vega: float, delta: float,
              S: float, sigma: float) -> Tuple[float, float]:
        hs = self.half_spread(gamma, vega, delta, S, sigma)
        return fair_value - hs, fair_value + hs
