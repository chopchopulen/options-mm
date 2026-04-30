class RiskLimits:
    SCALING_THRESHOLD = 0.9  # Start scaling down at 90% usage

    def __init__(self, max_gamma: float, max_vega: float, max_contracts_per_leg: int):
        self.max_gamma = max_gamma
        self.max_vega = max_vega
        self.max_contracts_per_leg = max_contracts_per_leg

    def adjusted_quote_size(self, desired_size: int, portfolio_gamma: float,
                            portfolio_vega: float, current_leg_position: int) -> int:
        size = desired_size

        # Gamma scaling: linearly reduce as portfolio gamma approaches max_gamma
        gamma_usage = abs(portfolio_gamma) / self.max_gamma
        if gamma_usage > self.SCALING_THRESHOLD:
            gamma_fraction = (1 - gamma_usage) / (1 - self.SCALING_THRESHOLD)
            size = min(size, max(0, int(desired_size * gamma_fraction)))

        # Vega scaling: linearly reduce as portfolio vega approaches max_vega
        vega_usage = abs(portfolio_vega) / self.max_vega
        if vega_usage > self.SCALING_THRESHOLD:
            vega_fraction = (1 - vega_usage) / (1 - self.SCALING_THRESHOLD)
            size = min(size, max(0, int(desired_size * vega_fraction)))

        # Hard position cap per leg
        leg_headroom = max(0, self.max_contracts_per_leg - abs(current_leg_position))
        size = min(size, leg_headroom)

        return size
