import numpy as np
from src.market.underlying import HestonSimulator
from src.market.order_flow import OrderFlowSimulator
from src.pricing.black_scholes import bs_price
from src.greeks.analytical import delta, gamma, vega, theta
from src.greeks.portfolio import portfolio_greeks
from src.mm.quoter import Quoter
from src.mm.inventory import Inventory
from src.mm.hedger import DeltaHedger
from src.risk.limits import RiskLimits
from src.pnl.attribution import PnLAttributor


class BacktestEngine:
    def __init__(self, cfg, seed: int = 42):
        self.cfg  = cfg
        self.seed = seed

    @staticmethod
    def _book_value(options, inventory, S, sigma, r, day, spd, contract_size):
        """Compute mark-to-market value of all option positions.

        Positions are stored in inventory keyed by (K, T_at_trade_time, option_type).
        We aggregate net quantity per (K, option_type) across all stored T keys,
        then price using current remaining time = (T_days/252) - (day/252).
        """
        # Aggregate net quantity per (K, option_type) across all stored expiry keys
        net_qty = {}
        for (K, T_stored, otype), pos in inventory.get_all_positions().items():
            if pos["quantity"] != 0:
                key = (K, otype)
                net_qty[key] = net_qty.get(key, 0) + pos["quantity"]

        total = 0.0
        for o in options:
            key = (o["K"], o["option_type"])
            qty = net_qty.get(key, 0)
            if qty != 0:
                T_o = max(0.0001, (o["T_days"] / 252) - (day / 252))
                price = bs_price(S, o["K"], T_o, r, sigma, o["option_type"])
                total += qty * price * contract_size
        return total

    def _build_option_universe(self, S0: float):
        opts = []
        for (offset, exp_days, otype) in self.cfg.OPTION_UNIVERSE:
            opts.append({
                "K": round(S0 * (1 + offset), 2),
                "T_days": exp_days,
                "option_type": otype,
            })
        return opts

    def run(self):
        bt   = self.cfg.BACKTEST
        n_days   = bt["n_days"]
        spd      = bt["steps_per_day"]
        dt       = 1 / 252 / spd
        r        = bt["risk_free_rate"]
        sigma_window = bt["sigma_uncertainty_window"]
        staleness = bt["quote_staleness_steps"]

        heston = HestonSimulator(**self.cfg.HESTON, seed=self.seed)
        total_steps = n_days * spd
        prices, variances = heston.simulate(n_steps=total_steps, dt=dt)

        flow_sim  = OrderFlowSimulator(**self.cfg.ORDER_FLOW, seed=self.seed + 1)
        quoter    = Quoter(**self.cfg.QUOTER)
        inventory = Inventory(contract_size=self.cfg.QUOTER["contract_size"])
        hedger    = DeltaHedger(**self.cfg.HEDGER)
        risk      = RiskLimits(**self.cfg.RISK)
        attributor = PnLAttributor()

        options = self._build_option_universe(self.cfg.HESTON["S0"])
        contract_size = self.cfg.QUOTER["contract_size"]

        daily_pnl         = []
        daily_attribution = []
        log_ret_history   = [0.0] * sigma_window
        sigma_implied     = bt["default_sigma"]

        for day in range(n_days):
            day_spread_fills = []
            day_hedge_costs  = []
            sigma_sod        = sigma_implied  # start-of-day IV for delta_sigma

            # Start-of-day snapshots for mark-to-market P&L
            S_sod = prices[day * spd]
            sod_book = self._book_value(options, inventory, S_sod, sigma_implied, r, day, spd, contract_size)
            realized_pnl_sod = inventory.realized_pnl
            underlying_sod = inventory.underlying_position * S_sod

            for step in range(spd):
                idx    = day * spd + step
                S_true = prices[idx + 1]
                S_stale = prices[max(0, idx + 1 - staleness)]

                # Compute sigma_implied from existing history (no look-ahead)
                sigma_implied = float(np.std(log_ret_history) * np.sqrt(252 * spd))
                sigma_implied = max(sigma_implied, 0.01)
                sigma_uncertainty = sigma_implied

                # Compute portfolio greeks once per step (O(N) instead of O(N^2))
                all_pos = []
                for o in options:
                    T_o = (o["T_days"] / 252) - (day / 252) - (step / (252 * spd))
                    if T_o <= 0:
                        continue
                    qty = inventory.get_option_position(o["K"], T_o, o["option_type"])
                    all_pos.append({
                        "S": S_stale, "K": o["K"], "T": T_o,
                        "r": r, "sigma": sigma_implied,
                        "option_type": o["option_type"], "quantity": qty,
                    })
                port_g = portfolio_greeks(all_pos, contract_size)

                for opt in options:
                    T_remaining = (opt["T_days"] / 252) - (day / 252) - (step / (252 * spd))
                    if T_remaining <= 0:
                        continue

                    fair    = bs_price(S_stale, opt["K"], T_remaining, r, sigma_implied, opt["option_type"])
                    g       = gamma(S_stale, opt["K"], T_remaining, r, sigma_implied)
                    v_greek = vega(S_stale, opt["K"], T_remaining, r, sigma_implied)

                    leg_pos = abs(inventory.get_option_position(opt["K"], T_remaining, opt["option_type"]))

                    size = risk.adjusted_quote_size(
                        desired_size=bt["desired_quote_size"],
                        portfolio_gamma=port_g["gamma"],
                        portfolio_vega=port_g["vega"],
                        current_leg_position=leg_pos,
                    )
                    if size == 0:
                        continue

                    bid, ask = quoter.quote(fair, g, v_greek, sigma_uncertainty)
                    trades = flow_sim.generate_trades(S_true, S_stale, bid, ask, dt)

                    for trade in trades:
                        fill_size = min(trade["size"], size)
                        if trade["side"] == "buy":
                            inventory.fill_option(opt["K"], T_remaining, opt["option_type"],
                                                  "sell", fill_size, ask)
                            day_spread_fills.append({
                                "spread_captured": ask - fair,
                                "size": fill_size,
                                "contract_size": contract_size,
                            })
                        else:
                            inventory.fill_option(opt["K"], T_remaining, opt["option_type"],
                                                  "buy", fill_size, bid)
                            day_spread_fills.append({
                                "spread_captured": fair - bid,
                                "size": fill_size,
                                "contract_size": contract_size,
                            })

                # Update rolling log-return window after quoting/trading (eliminates look-ahead bias)
                log_ret = np.log(prices[idx + 1] / prices[idx])
                log_ret_history.append(log_ret)
                if len(log_ret_history) > sigma_window:
                    log_ret_history.pop(0)

                # Delta hedge at end of each step
                all_pos_now = []
                for o in options:
                    T_o = (o["T_days"] / 252) - (day / 252) - ((step + 1) / (252 * spd))
                    T_o = max(T_o, 0.0001)
                    qty = inventory.get_option_position(o["K"], T_o, o["option_type"])
                    all_pos_now.append({
                        "S": S_true, "K": o["K"], "T": T_o,
                        "r": r, "sigma": sigma_implied,
                        "option_type": o["option_type"], "quantity": qty,
                    })
                port_now = portfolio_greeks(all_pos_now, contract_size)
                total_delta = port_now["delta"] + inventory.underlying_position
                hedge_trades = hedger.check_and_hedge(total_delta, S_true, inventory)
                for ht in hedge_trades:
                    day_hedge_costs.append(ht["transaction_cost"])

            # End of day
            day_prices   = [prices[day * spd + i] for i in range(spd + 1)]
            log_rets_day = np.diff(np.log(day_prices))
            realized_var = float(np.var(log_rets_day) * 252 * spd)
            implied_var  = sigma_implied ** 2
            delta_sigma  = sigma_implied - sigma_sod

            S_eod = prices[(day + 1) * spd]
            # Aggregate net quantities per (K, option_type) across all stored T keys
            net_qty_eod = {}
            for (K, T_stored, otype), pos in inventory.get_all_positions().items():
                if pos["quantity"] != 0:
                    net_qty_eod[(K, otype)] = net_qty_eod.get((K, otype), 0) + pos["quantity"]
            eod_pos = []
            for o in options:
                T_o = max(0.0001, (o["T_days"] / 252) - ((day + 1) / 252))
                qty = net_qty_eod.get((o["K"], o["option_type"]), 0)
                eod_pos.append({
                    "S": S_eod, "K": o["K"], "T": T_o,
                    "r": r, "sigma": sigma_implied,
                    "option_type": o["option_type"], "quantity": qty,
                })
            port_eod = portfolio_greeks(eod_pos, contract_size)

            # Proper mark-to-market P&L: change in total portfolio value
            eod_book = self._book_value(options, inventory, S_eod, sigma_implied, r, day + 1, spd, contract_size)
            realized_pnl_delta = inventory.realized_pnl - realized_pnl_sod
            underlying_eod = inventory.underlying_position * S_eod
            underlying_pnl = underlying_eod - underlying_sod
            mtm_pnl = (eod_book - sod_book) + realized_pnl_delta + underlying_pnl

            attr = attributor.compute(
                spread_fills=day_spread_fills,
                portfolio_theta=port_eod["theta"],
                portfolio_gamma=port_eod["gamma"],
                portfolio_vega=port_eod["vega"],
                S=S_eod,
                realized_variance=realized_var,
                implied_variance=implied_var,
                delta_sigma_implied=delta_sigma,
                hedge_costs=day_hedge_costs,
                mtm_pnl=mtm_pnl,
                dt=1 / 252,
            )
            daily_pnl.append(attr["total"])
            daily_attribution.append(attr)

        return {
            "daily_pnl":         daily_pnl,
            "daily_attribution": daily_attribution,
            "total_pnl":         sum(daily_pnl),
            "prices":            prices,
        }
