class RiskLimits:
    def __init__(self, max_gamma: float, max_vega: float, max_contracts_per_leg: int):
        self.max_gamma             = max_gamma
        self.max_vega              = max_vega
        self.max_contracts_per_leg = max_contracts_per_leg

    def adjusted_quote_size(self, desired_size: int, portfolio_gamma: float,
                            portfolio_vega: float, current_leg_position: int) -> int:
        size = desired_size

        gamma_headroom = max(0.0, self.max_gamma - abs(portfolio_gamma))
        gamma_fraction = gamma_headroom / self.max_gamma
        size = min(size, max(0, int(desired_size * gamma_fraction)))

        vega_headroom  = max(0.0, self.max_vega - abs(portfolio_vega))
        vega_fraction  = vega_headroom / self.max_vega
        size = min(size, max(0, int(desired_size * vega_fraction)))

        leg_headroom = max(0, self.max_contracts_per_leg - abs(current_leg_position))
        size = min(size, leg_headroom)

        return size
