class PnLAttributor:
    def compute(self, spread_fills, portfolio_theta, portfolio_gamma, portfolio_vega,
                portfolio_vanna=0.0, portfolio_volga=0.0,
                S=None, realized_variance=None, implied_variance=None, delta_sigma_implied=None,
                hedge_costs=None, mtm_pnl=0.0, dt=None) -> dict:
        if hedge_costs is None:
            hedge_costs = []
        spread_capture = sum(
            f["spread_captured"] * f["size"] * f["contract_size"]
            for f in spread_fills
        )
        theta_pnl = portfolio_theta * dt
        gamma_pnl = 0.5 * portfolio_gamma * S**2 * (realized_variance - implied_variance) * dt
        vega_pnl = portfolio_vega * delta_sigma_implied
        vanna_pnl = portfolio_vanna   # caller already computed: sum of (vanna_greek * dS * dsigma) per step
        volga_pnl = portfolio_volga   # caller already computed: sum of (0.5 * volga_greek * dsigma**2) per step
        hedge_cost = -sum(hedge_costs)
        explained = spread_capture + theta_pnl + gamma_pnl + vega_pnl + vanna_pnl + volga_pnl + hedge_cost
        residual = mtm_pnl - explained
        total = mtm_pnl
        return {
            "spread_capture": spread_capture,
            "theta_pnl": theta_pnl,
            "gamma_pnl": gamma_pnl,
            "vega_pnl": vega_pnl,
            "vanna_pnl": vanna_pnl,
            "volga_pnl": volga_pnl,
            "hedge_cost": hedge_cost,
            "residual": residual,
            "total": total,
        }
