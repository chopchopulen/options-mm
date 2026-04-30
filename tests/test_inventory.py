from src.mm.inventory import Inventory

def test_fill_updates_position():
    inv = Inventory(contract_size=100)
    inv.fill_option(strike=100.0, expiry=1.0, option_type="call",
                    side="sell", size=5, price=3.0)
    pos = inv.get_option_position(strike=100.0, expiry=1.0, option_type="call")
    assert pos == -5  # sold 5 → short 5

def test_fill_buy_adds():
    inv = Inventory(contract_size=100)
    inv.fill_option(strike=100.0, expiry=1.0, option_type="call",
                    side="buy", size=3, price=3.0)
    pos = inv.get_option_position(strike=100.0, expiry=1.0, option_type="call")
    assert pos == 3

def test_hedge_fill_tracks_underlying():
    inv = Inventory(contract_size=100)
    inv.fill_underlying(side="buy", size=50, price=100.0)
    assert inv.underlying_position == 50

def test_realized_pnl_on_close():
    inv = Inventory(contract_size=100)
    inv.fill_option(strike=100.0, expiry=1.0, option_type="call",
                    side="sell", size=1, price=5.0)
    inv.fill_option(strike=100.0, expiry=1.0, option_type="call",
                    side="buy", size=1, price=3.0)
    # Sold at 5, bought back at 3 → profit of 2 per share × 100 = 200
    assert abs(inv.realized_pnl - 200.0) < 1e-6
