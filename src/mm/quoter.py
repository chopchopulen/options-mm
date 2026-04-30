from typing import Tuple


class Quoter:
    def __init__(self, base_spread: float, gamma_coeff: float,
                 vega_coeff: float, contract_size: int = 100):
        self.base_spread   = base_spread
        self.gamma_coeff   = gamma_coeff
        self.vega_coeff    = vega_coeff
        self.contract_size = contract_size

    def half_spread(self, gamma: float, vega: float, sigma_uncertainty: float) -> float:
        return (self.base_spread
                + self.gamma_coeff * abs(gamma) * self.contract_size
                + self.vega_coeff  * abs(vega)  * sigma_uncertainty)

    def quote(self, fair_value: float, gamma: float, vega: float,
              sigma_uncertainty: float) -> Tuple[float, float]:
        hs = self.half_spread(gamma, vega, sigma_uncertainty)
        return fair_value - hs, fair_value + hs
