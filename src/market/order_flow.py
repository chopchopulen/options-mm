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
                        option_edge: float = None) -> List[Dict]:
        trades = []
        # Noise traders: Poisson arrivals. lambda_noise is per DAY (README: "λ=8/day × dt"),
        # so dt_days must be in days. Passing an annualized dt here understated arrivals by
        # 252x — 8.0 * (1/252/78) yields ~5.7 noise trades per MONTH instead of per day.
        n_noise = self.rng.poisson(self.lambda_noise * dt_days)
        for _ in range(n_noise):
            side  = "buy" if self.rng.random() < 0.5 else "sell"
            size  = int(self.rng.integers(1, self.max_noise_size + 1))
            price = ask if side == "buy" else bid
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
            else:
                edge = option_edge
                if abs(edge) <= (ask - bid) / 2.0:
                    return trades          # not worth crossing; informed stays out
            if edge > 0:
                # Option worth more than the MM's stale fair → MM ask is cheap → informed buys
                trades.append({"side": "buy",  "size": size, "price": ask, "trader_type": "informed"})
            else:
                # Option worth less than the MM's stale fair → MM bid is rich → informed sells
                trades.append({"side": "sell", "size": size, "price": bid, "trader_type": "informed"})

        return trades
