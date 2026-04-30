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
                        bid: float, ask: float, dt: float) -> List[Dict]:
        trades = []
        # Noise traders: Poisson arrivals
        n_noise = self.rng.poisson(self.lambda_noise * dt)
        for _ in range(n_noise):
            side  = "buy" if self.rng.random() < 0.5 else "sell"
            size  = int(self.rng.integers(1, self.max_noise_size + 1))
            price = ask if side == "buy" else bid
            trades.append({"side": side, "size": size, "price": price, "trader_type": "noise"})

        # Informed traders: arrive only when quotes are stale
        mispricing = (S_true - S_stale) / S_stale
        if abs(mispricing) > self.staleness_threshold:
            size = int(self.rng.integers(self.min_informed_size, self.max_informed_size + 1))
            if mispricing > 0:
                # True price higher → MM ask is cheap → informed buys
                trades.append({"side": "buy",  "size": size, "price": ask, "trader_type": "informed"})
            else:
                # True price lower → MM bid is rich → informed sells
                trades.append({"side": "sell", "size": size, "price": bid, "trader_type": "informed"})

        return trades
