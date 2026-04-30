from src.mm.quoter import Quoter

def test_bid_below_ask():
    q = Quoter(base_spread=0.05, gamma_coeff=0.5, vega_coeff=0.1, contract_size=100)
    bid, ask = q.quote(fair_value=5.0, gamma=0.02, vega=10.0, sigma_uncertainty=0.01)
    assert bid < ask

def test_symmetric_around_fair():
    q = Quoter(base_spread=0.05, gamma_coeff=0.5, vega_coeff=0.1, contract_size=100)
    bid, ask = q.quote(fair_value=5.0, gamma=0.02, vega=10.0, sigma_uncertainty=0.01)
    assert abs((bid + ask) / 2 - 5.0) < 1e-10

def test_wider_with_higher_gamma():
    q = Quoter(base_spread=0.05, gamma_coeff=0.5, vega_coeff=0.0, contract_size=100)
    bid_lo, ask_lo = q.quote(5.0, gamma=0.01, vega=0.0, sigma_uncertainty=0.0)
    bid_hi, ask_hi = q.quote(5.0, gamma=0.10, vega=0.0, sigma_uncertainty=0.0)
    assert (ask_hi - bid_hi) > (ask_lo - bid_lo)

def test_wider_with_higher_vol_uncertainty():
    q = Quoter(base_spread=0.05, gamma_coeff=0.0, vega_coeff=0.1, contract_size=100)
    bid_lo, ask_lo = q.quote(5.0, gamma=0.0, vega=10.0, sigma_uncertainty=0.01)
    bid_hi, ask_hi = q.quote(5.0, gamma=0.0, vega=10.0, sigma_uncertainty=0.10)
    assert (ask_hi - bid_hi) > (ask_lo - bid_lo)
