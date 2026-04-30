from typing import List, Dict
from src.mm.inventory import Inventory


class DeltaHedger:
    def __init__(self, delta_threshold: float, transaction_cost: float):
        self.delta_threshold = delta_threshold  # in shares
        self.transaction_cost = transaction_cost  # fraction of trade value

    def check_and_hedge(self, portfolio_delta: float, S: float,
                        inventory: Inventory) -> List[Dict]:
        if abs(portfolio_delta) <= self.delta_threshold:
            return []

        hedge_shares = -portfolio_delta  # flatten to zero
        side = "buy" if hedge_shares > 0 else "sell"
        size = abs(hedge_shares)
        cost = size * S * self.transaction_cost
        price = S * (1 + self.transaction_cost) if side == "buy" else S * (1 - self.transaction_cost)

        inventory.fill_underlying(side=side, size=size, price=price)

        return [{"side": side, "size": size, "price": price, "transaction_cost": cost}]
