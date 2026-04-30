from typing import Dict, Tuple
from collections import defaultdict


class Inventory:
    def __init__(self, contract_size: int = 100):
        self.contract_size       = contract_size
        self.underlying_position = 0.0
        self.realized_pnl        = 0.0
        self._options: Dict[Tuple, Dict] = defaultdict(lambda: {"quantity": 0, "avg_cost": 0.0})

    def fill_option(self, strike: float, expiry: float, option_type: str,
                    side: str, size: int, price: float) -> None:
        key = (strike, expiry, option_type)
        pos = self._options[key]
        signed    = size if side == "buy" else -size
        old_qty   = pos["quantity"]
        old_cost  = pos["avg_cost"]
        new_qty   = old_qty + signed

        if new_qty == 0:
            # Position closes completely
            self.realized_pnl += (price - old_cost) * old_qty * self.contract_size
            pos["quantity"] = 0
            pos["avg_cost"] = 0.0
        elif (old_qty >= 0 and signed > 0) or (old_qty <= 0 and signed < 0):
            # Adding to existing position (same direction)
            total_cost    = old_cost * abs(old_qty) + price * abs(signed)
            pos["avg_cost"] = total_cost / abs(new_qty)
            pos["quantity"] = new_qty
        else:
            # Closing part of position (opposite direction)
            close_qty = min(abs(old_qty), abs(signed))
            self.realized_pnl += (price - old_cost) * close_qty * self.contract_size * (1 if old_qty > 0 else -1)
            pos["quantity"] = new_qty
            if new_qty != 0:
                pos["avg_cost"] = price

    def fill_underlying(self, side: str, size: float, price: float) -> None:
        signed = size if side == "buy" else -size
        self.realized_pnl       += -signed * price
        self.underlying_position += signed

    def get_option_position(self, strike: float, expiry: float, option_type: str) -> int:
        return self._options[(strike, expiry, option_type)]["quantity"]

    def get_all_positions(self) -> Dict:
        return dict(self._options)
