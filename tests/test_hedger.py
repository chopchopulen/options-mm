from src.mm.hedger import DeltaHedger
from src.mm.inventory import Inventory


def test_no_hedge_below_threshold():
    inv = Inventory()
    hedger = DeltaHedger(delta_threshold=50.0, transaction_cost=0.01)
    trades = hedger.check_and_hedge(portfolio_delta=30.0, S=100.0, inventory=inv)
    assert trades == []


def test_hedge_above_threshold():
    inv = Inventory()
    hedger = DeltaHedger(delta_threshold=50.0, transaction_cost=0.01)
    # portfolio_delta = 80 shares → need to sell 80 shares to flatten
    trades = hedger.check_and_hedge(portfolio_delta=80.0, S=100.0, inventory=inv)
    assert len(trades) == 1
    assert trades[0]["side"] == "sell"
    assert abs(trades[0]["size"] - 80.0) < 1e-6


def test_hedge_negative_delta():
    inv = Inventory()
    hedger = DeltaHedger(delta_threshold=50.0, transaction_cost=0.01)
    trades = hedger.check_and_hedge(portfolio_delta=-75.0, S=100.0, inventory=inv)
    assert len(trades) == 1
    assert trades[0]["side"] == "buy"


def test_hedge_updates_inventory():
    inv = Inventory()
    hedger = DeltaHedger(delta_threshold=50.0, transaction_cost=0.01)
    hedger.check_and_hedge(portfolio_delta=80.0, S=100.0, inventory=inv)
    assert abs(inv.underlying_position - (-80.0)) < 1e-6
