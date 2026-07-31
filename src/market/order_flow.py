import numpy as np
from typing import List, Dict


class OrderFlowSimulator:
    def __init__(self, lambda_noise: float, max_noise_size: int,
                 min_informed_size: int, max_informed_size: int,
                 staleness_threshold: float, seed: int = None):
        self.lambda_noise        = lambda_noise
        self.max_noise_size      = max_noise_size
        self.min_informed_size   = min_informed_size
        self.max_informed_size   = max_informed_size
        self.staleness_threshold = staleness_threshold
        self.rng = np.random.default_rng(seed)

    def generate_trades(self, S_true: float, S_stale: float,
                        bid: float, ask: float, dt_days: float,
                        option_type: str = "call",
                        option_edge: float = None,
                        reservation_scale: float = None,
                        vol_signal: float = None,
                        vol_edge: float = None,
                        vol_threshold: float = None) -> List[Dict]:
        trades = []
        # Noise traders: Poisson arrivals. lambda_noise is per DAY (README: "λ=8/day × dt"),
        # so dt_days must be in days. Passing an annualized dt here understated arrivals by
        # 252x — 8.0 * (1/252/78) yields ~5.7 noise trades per MONTH instead of per day.
        n_noise = self.rng.poisson(self.lambda_noise * dt_days)
        half_spread = (ask - bid) / 2.0
        for _ in range(n_noise):
            side  = "buy" if self.rng.random() < 0.5 else "sell"
            size  = int(self.rng.integers(1, self.max_noise_size + 1))
            price = ask if side == "buy" else bid

            # ---- Counterparty reservation price -------------------------------------
            # ASSUMPTION, NOT A DERIVATION. This is the one place in this model where a
            # behavioural rule is ASSERTED rather than derived from the dynamics.
            #
            # An arriving noise trader is assumed to hold a reservation price
            # fair_now +/- eps with eps ~ Exponential(reservation_scale), and to trade
            # only if the quote falls inside it. Under an exponential the fill
            # probability is exp(-cost / scale), where cost is what the trader gives up
            # against the contemporaneous fair. The exponential shape is a choice; a
            # logistic or a truncated normal would give the same qualitative elasticity
            # with different tail behaviour, and nothing in this model selects between
            # them.
            #
            # Only the SCALE is anchored: it is the option's own price uncertainty over
            # the holding horizon, |vega| * (xi/2) * sqrt(tau) — the same quantity the
            # quoter charges for in its vega carry term. The argument is that a
            # counterparty's willingness to pay away from fair is dispersed by roughly
            # the amount the price itself moves over that horizon.
            #
            # Without this, fill probability is independent of quote width and P&L is
            # linear and unbounded in a quantity the market maker chooses.
            if reservation_scale is not None and reservation_scale > 1e-12:
                edge = 0.0 if option_edge is None else option_edge
                cost = (half_spread - edge) if side == "buy" else (half_spread + edge)
                if cost > 0.0:
                    # Cap the exponent: reservation_scale goes to zero for near-expiry or
                    # deep-OTM legs (vega -> 0), and cost/scale then overflows. The limit
                    # is p_fill -> 0, so clamping to a large ratio is the correct value,
                    # not an approximation.
                    ratio = min(cost / reservation_scale, 700.0)
                    if self.rng.random() >= np.exp(-ratio):
                        continue    # quote outside this trader's reservation price

            trades.append({"side": side, "size": size, "price": price, "trader_type": "noise"})

        # Informed traders: arrive only when quotes are stale.
        # The trigger is staleness of the UNDERLYING, but the profitable side depends on
        # the option type: a call gains value when S rises, a put loses it. Deciding the
        # side from the sign of the underlying move alone is correct for calls and exactly
        # backwards for puts.
        #
        # A trader that is informed will not knowingly cross a spread wider than its own
        # edge, so arrival is gated on `abs(edge) > half_spread`. Without that gate 7.2%
        # of informed fills traded at a certain loss, and — more importantly — fill
        # probability was completely independent of quote width, which made P&L linear
        # and unbounded in a quantity the market maker chooses.
        #
        # `option_edge` is the edge in DOLLARS (contemporaneous fair minus quote-time
        # fair, signed for the option). When it is not supplied the gate cannot be
        # applied, and the side falls back to the sign of the underlying move.
        mispricing = (S_true - S_stale) / S_stale
        if abs(mispricing) > self.staleness_threshold:
            size = int(self.rng.integers(self.min_informed_size, self.max_informed_size + 1))
            if option_edge is None:
                edge = mispricing if option_type == "call" else -mispricing
                worth_crossing = True
            else:
                edge = option_edge
                # Not worth crossing -- but only THIS population stands down. An early
                # return here would also suppress the vol-informed trader below, whose
                # edge is an entirely different quantity.
                worth_crossing = abs(edge) > half_spread
            if not worth_crossing:
                pass
            elif edge > 0:
                # Option worth more than the MM's stale fair → MM ask is cheap → informed buys
                trades.append({"side": "buy",  "size": size, "price": ask, "trader_type": "informed"})
            else:
                # Option worth less than the MM's stale fair → MM bid is rich → informed sells
                trades.append({"side": "sell", "size": size, "price": bid, "trader_type": "informed"})

        # VOL-informed traders. A second, economically distinct informed population.
        #
        # The population above exploits staleness in the maker's view of SPOT. This one
        # exploits staleness in its view of VOLATILITY: the maker quotes off a 10-sample
        # rolling realized-vol estimate, which lags the true instantaneous variance, and a
        # counterparty who knows where vol actually is can buy or sell the whole surface
        # against it. In real options markets this is the flow that hurts a maker most --
        # it arrives ahead of vol events, and it is why a maker manages vega rather than
        # just delta. It could not be modelled at all before the vol surface existed.
        #
        # `vol_signal` is the level error in IV points and `vol_edge` is that error priced
        # through the leg's vega, in dollars per share. Arrival needs BOTH a signal larger
        # than the estimator could produce by chance, and an edge that clears the spread
        # the trader must cross -- the same discipline applied to the spot population.
        if (vol_signal is not None and vol_edge is not None
                and vol_threshold is not None
                and abs(vol_signal) > vol_threshold
                and abs(vol_edge) > half_spread):
            size = int(self.rng.integers(self.min_informed_size, self.max_informed_size + 1))
            if vol_signal > 0:
                # True vol above the maker's mark → options are cheap → buy vega.
                trades.append({"side": "buy", "size": size, "price": ask,
                               "trader_type": "informed_vol"})
            else:
                trades.append({"side": "sell", "size": size, "price": bid,
                               "trader_type": "informed_vol"})

        return trades
