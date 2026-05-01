class PnLAttributor:
    def compute(self, spread_fills, portfolio_theta, portfolio_gamma, portfolio_vega,
                S, realized_variance, implied_variance, delta_sigma_implied,
                hedge_costs, mtm_pnl, dt) -> dict:
        spread_capture = sum(
            f["spread_captured"] * f["size"] * f["contract_size"]
            for f in spread_fills
        )
        theta_pnl = portfolio_theta * dt
        gamma_pnl = 0.5 * portfolio_gamma * S**2 * (realized_variance - implied_variance) * dt
        vega_pnl = portfolio_vega * delta_sigma_implied
        hedge_cost = -sum(hedge_costs)
        explained = spread_capture + theta_pnl + gamma_pnl + vega_pnl + hedge_cost
        residual = mtm_pnl - explained
        total = mtm_pnl
        return {
            "spread_capture": spread_capture,
            "theta_pnl": theta_pnl,
            "gamma_pnl": gamma_pnl,
            "vega_pnl": vega_pnl,
            "hedge_cost": hedge_cost,
            "residual": residual,
            "total": total,
        }
