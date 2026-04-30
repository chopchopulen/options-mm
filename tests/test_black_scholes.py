import pytest
import numpy as np
from src.pricing.black_scholes import bs_price, bs_d1, bs_d2

def test_call_put_parity():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    call = bs_price(S, K, T, r, sigma, "call")
    put  = bs_price(S, K, T, r, sigma, "put")
    # Put-call parity: C - P = S - K*e^(-rT)
    assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-6

def test_atm_call_known_value():
    # ATM call, S=K=100, T=1, r=0, sigma=0.2 → ~7.9656
    call = bs_price(100.0, 100.0, 1.0, 0.0, 0.2, "call")
    assert abs(call - 7.9656) < 1e-3

def test_deep_itm_call_approaches_intrinsic():
    call = bs_price(200.0, 100.0, 0.01, 0.0, 0.2, "call")
    assert abs(call - 100.0) < 1.0

def test_expired_call():
    call = bs_price(110.0, 100.0, 0.0, 0.05, 0.2, "call")
    assert abs(call - 10.0) < 1e-6

def test_invalid_option_type():
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.05, 0.2, "future")

from src.pricing.binomial import binomial_price

def test_binomial_converges_to_bs():
    from src.pricing.black_scholes import bs_price
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    bs = bs_price(S, K, T, r, sigma, "call")
    bt = binomial_price(S, K, T, r, sigma, "call", n_steps=500)
    assert abs(bt - bs) < 0.05  # within 5 cents at 500 steps

def test_binomial_put_call_parity():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    call = binomial_price(S, K, T, r, sigma, "call", n_steps=200)
    put  = binomial_price(S, K, T, r, sigma, "put",  n_steps=200)
    import numpy as np
    assert abs((call - put) - (S - K * np.exp(-r * T))) < 0.1
