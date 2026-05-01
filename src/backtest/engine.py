import numpy as np
from src.market.underlying import HestonSimulator
from src.market.order_flow import OrderFlowSimulator
from src.pricing.black_scholes import bs_price
from src.greeks.analytical import gamma, vega
from src.greeks.portfolio import portfolio_greeks
from src.mm.quoter import Quoter
from src.mm.inventory import Inventory
from src.mm.hedger import DeltaHedger
from src.risk.limits import RiskLimits

TRADING_DAYS = 252


class BacktestEngine:
    def __init__(self, cfg, seed: int = 42):
        self.cfg  = cfg
        self.seed = seed

    def _build_option_universe(self, S0: float):
        opts = []
        for (offset, exp_days, otype) in self.cfg.OPTION_UNIVERSE:
            opts.append({
                "K": round(S0 * (1 + offset), 2),
                "T_days": exp_days,
                "option_type": otype,
            })
        return opts

    @staticmethod
    def _book_value(options, inventory, S, sigma, r, days_elapsed, contract_size):
        total = 0.0
        for o in options:
            T_o = max(0.0001, (o["T_days"] / TRADING_DAYS) - (days_elapsed / TRADING_DAYS))
            qty = inventory.get_option_position(o["K"], o["T_days"], o["option_type"])
            if qty != 0:
                price = bs_price(S, o["K"], T_o, r, sigma, o["option_type"])
                total += qty * price * contract_size
        return total

    def run(self):
        bt   = self.cfg.BACKTEST
        n_days   = bt["n_days"]
        spd      = bt["steps_per_day"]
        dt       = 1.0 / TRADING_DAYS / spd
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

        options = self._build_option_universe(self.cfg.HESTON["S0"])
        contract_size = self.cfg.QUOTER["contract_size"]

        daily_pnl         = []
        daily_attribution = []
        log_ret_history   = [0.0] * sigma_window
        sigma_implied     = bt["default_sigma"]

        for day in range(n_days):
            day_spread_fills = []
            day_hedge_costs  = []
            sigma_sod        = sigma_implied

            S_sod = prices[day * spd]
            sod_book = self._book_value(options, inventory, S_sod, sigma_implied, r, day, contract_size)
            realized_pnl_sod = inventory.realized_pnl
            underlying_sod = inventory.underlying_position * S_sod

            # Initialize intraday attribution accumulators (theta/gamma: step-by-step; vega/vanna/volga: set at EOD)
            daily_theta_pnl = 0.0
            daily_gamma_pnl = 0.0

            for step in range(spd):
                idx     = day * spd + step
                S_true  = prices[idx + 1]
                S_stale = prices[max(0, idx + 1 - staleness)]

                # Compute sigma_implied from existing history (no look-ahead)
                sigma_implied = float(np.std(log_ret_history) * np.sqrt(TRADING_DAYS * spd))
                sigma_implied = max(sigma_implied, 0.01)
                sigma_uncertainty = sigma_implied

                # Compute portfolio greeks once per step
                all_pos = []
                for o in options:
                    T_o = (o["T_days"] / TRADING_DAYS) - (day / TRADING_DAYS) - (step / (TRADING_DAYS * spd))
                    if T_o <= 0:
                        continue
                    qty = inventory.get_option_position(o["K"], o["T_days"], o["option_type"])
                    all_pos.append({
                        "S": S_stale, "K": o["K"], "T": T_o,
                        "r": r, "sigma": sigma_implied,
                        "option_type": o["option_type"], "quantity": qty,
                    })
                port_g = portfolio_greeks(all_pos, contract_size)

                for opt in options:
                    T_remaining = (opt["T_days"] / TRADING_DAYS) - (day / TRADING_DAYS) - (step / (TRADING_DAYS * spd))
                    if T_remaining <= 0:
                        continue

                    fair    = bs_price(S_stale, opt["K"], T_remaining, r, sigma_implied, opt["option_type"])
                    g       = gamma(S_stale, opt["K"], T_remaining, r, sigma_implied)
                    v_greek = vega(S_stale, opt["K"], T_remaining, r, sigma_implied)

                    leg_pos = abs(inventory.get_option_position(opt["K"], opt["T_days"], opt["option_type"]))

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
                            # Counterparty buys → MM sells at ask
                            inventory.fill_option(opt["K"], opt["T_days"], opt["option_type"],
                                                  "sell", fill_size, ask)
                            day_spread_fills.append({
                                "spread_captured": ask - fair,
                                "size": fill_size,
                                "contract_size": contract_size,
                            })
                        else:
                            # Counterparty sells → MM buys at bid
                            inventory.fill_option(opt["K"], opt["T_days"], opt["option_type"],
                                                  "buy", fill_size, bid)
                            day_spread_fills.append({
                                "spread_captured": fair - bid,
                                "size": fill_size,
                                "contract_size": contract_size,
                            })

                # Update rolling log-return window after quoting/trading (no look-ahead)
                log_ret = np.log(prices[idx + 1] / prices[idx])
                log_ret_history.append(log_ret)
                if len(log_ret_history) > sigma_window:
                    log_ret_history.pop(0)

                # Delta hedge at end of each step
                all_pos_now = []
                for o in options:
                    T_o = max(0.0001, (o["T_days"] / TRADING_DAYS) - (day / TRADING_DAYS) - ((step + 1) / (TRADING_DAYS * spd)))
                    qty = inventory.get_option_position(o["K"], o["T_days"], o["option_type"])
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

                # Accumulate intraday theta and gamma P&L using current step's Greeks
                step_dt = 1.0 / TRADING_DAYS / spd
                step_realized_var = (log_ret ** 2) * TRADING_DAYS * spd  # annualized realized var this step
                step_implied_var = sigma_implied ** 2
                daily_theta_pnl += port_now["theta"] * step_dt
                daily_gamma_pnl += 0.5 * port_now["gamma"] * S_true**2 * (step_realized_var - step_implied_var) * step_dt


            # End of day
            S_eod = prices[(day + 1) * spd]
            eod_book = self._book_value(options, inventory, S_eod, sigma_implied, r, day + 1, contract_size)
            realized_pnl_delta = inventory.realized_pnl - realized_pnl_sod
            underlying_eod = inventory.underlying_position * S_eod
            underlying_pnl = underlying_eod - underlying_sod
            mtm_pnl = (eod_book - sod_book) + realized_pnl_delta + underlying_pnl

            # Vega P&L: use EOD portfolio greeks and net sigma change SOD→EOD
            # MTM book reflects only the net sigma change, not intraday oscillations
            delta_sigma = sigma_implied - sigma_sod
            eod_pos = []
            for o in options:
                T_o = max(0.0001, (o["T_days"] / TRADING_DAYS) - ((day + 1) / TRADING_DAYS))
                qty = inventory.get_option_position(o["K"], o["T_days"], o["option_type"])
                eod_pos.append({
                    "S": S_eod, "K": o["K"], "T": T_o,
                    "r": r, "sigma": sigma_implied,
                    "option_type": o["option_type"], "quantity": qty,
                })
            port_eod = portfolio_greeks(eod_pos, contract_size)
            daily_vega_pnl = port_eod["vega"] * delta_sigma

            # Volga P&L: use EOD portfolio greeks and net (SOD→EOD) sigma change squared
            # MTM book only reflects the net sigma change, not intraday oscillations
            delta_S = S_eod - S_sod
            daily_volga_pnl = 0.5 * port_eod["volga"] * delta_sigma ** 2
            # Vanna P&L: overwrite intraday sum with EOD-based net move
            # (consistent with how vega and volga are computed from net daily moves)
            daily_vanna_pnl = port_eod["vanna"] * delta_S * delta_sigma

            # Build attribution dict using intraday-accumulated Greek P&L components
            spread_capture = sum(
                f["spread_captured"] * f["size"] * f["contract_size"]
                for f in day_spread_fills
            )
            hedge_cost_total = -sum(day_hedge_costs)
            residual = mtm_pnl - (spread_capture + daily_theta_pnl + daily_gamma_pnl + daily_vega_pnl + daily_vanna_pnl + daily_volga_pnl + hedge_cost_total)

            attr = {
                "spread_capture": spread_capture,
                "theta_pnl":      daily_theta_pnl,
                "gamma_pnl":      daily_gamma_pnl,
                "vega_pnl":       daily_vega_pnl,
                "vanna_pnl":      daily_vanna_pnl,
                "volga_pnl":      daily_volga_pnl,
                "hedge_cost":     hedge_cost_total,
                "residual":       residual,
                "total":          mtm_pnl,
            }
            daily_pnl.append(attr["total"])
            daily_attribution.append(attr)

        return {
            "daily_pnl":         daily_pnl,
            "daily_attribution": daily_attribution,
            "total_pnl":         sum(daily_pnl),
            "prices":            prices,
        }
