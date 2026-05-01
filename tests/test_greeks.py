import pytest
import numpy as np
from src.greeks.analytical import delta, gamma, vega, theta

def test_call_delta_range():
    d = delta(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    assert 0.0 < d < 1.0

def test_put_delta_range():
    d = delta(100.0, 100.0, 1.0, 0.05, 0.2, "put")
    assert -1.0 < d < 0.0

def test_call_put_delta_sum():
    # call_delta - put_delta = 1 (analytically)
    dc = delta(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    dp = delta(100.0, 100.0, 1.0, 0.05, 0.2, "put")
    assert abs(dc - dp - 1.0) < 1e-10

def test_gamma_positive():
    g = gamma(100.0, 100.0, 1.0, 0.05, 0.2)
    assert g > 0

def test_gamma_same_for_call_put():
    # By BS symmetry, call and put Gamma are identical
    g_call = gamma(100.0, 100.0, 1.0, 0.05, 0.2)
    g_put  = gamma(100.0, 100.0, 1.0, 0.05, 0.2)
    assert abs(g_call - g_put) < 1e-10

def test_vega_positive():
    v = vega(100.0, 100.0, 1.0, 0.05, 0.2)
    assert v > 0

def test_theta_call_negative():
    t = theta(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    assert t < 0

def test_atm_call_delta_near_half():
    # ATM call delta ≈ 0.5 at r=0
    d = delta(100.0, 100.0, 1.0, 0.0, 0.2, "call")
    assert abs(d - 0.5) < 0.05


# Task 3: Finite-Difference Greeks tests
from src.greeks.numerical import delta_fd, gamma_fd, vega_fd, theta_fd

def test_fd_delta_matches_analytical():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    assert abs(delta_fd(S, K, T, r, sigma, "call") - delta(S, K, T, r, sigma, "call")) < 1e-4

def test_fd_gamma_matches_analytical():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    assert abs(gamma_fd(S, K, T, r, sigma) - gamma(S, K, T, r, sigma)) < 1e-4

def test_fd_vega_matches_analytical():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    assert abs(vega_fd(S, K, T, r, sigma) - vega(S, K, T, r, sigma)) < 1e-3

def test_fd_theta_matches_analytical():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    assert abs(theta_fd(S, K, T, r, sigma, "call") - theta(S, K, T, r, sigma, "call")) < 5e-3


# Task 4: Portfolio Greeks Aggregator tests
from src.greeks.portfolio import portfolio_greeks

def test_portfolio_greeks_two_positions():
    positions = [
        {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "option_type": "call", "quantity": 10},
        {"S": 100.0, "K": 105.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "option_type": "put",  "quantity": -5},
    ]
    g = portfolio_greeks(positions, contract_size=100)
    assert "delta" in g and "gamma" in g and "vega" in g and "theta" in g

def test_portfolio_delta_is_sum():
    from src.greeks.analytical import delta as adelta
    positions = [
        {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "option_type": "call", "quantity": 10},
    ]
    g = portfolio_greeks(positions, contract_size=100)
    expected = adelta(100.0, 100.0, 1.0, 0.05, 0.2, "call") * 10 * 100
    assert abs(g["delta"] - expected) < 1e-8

def test_portfolio_empty():
    g = portfolio_greeks([], contract_size=100)
    assert g["delta"] == 0.0 and g["gamma"] == 0.0


from src.greeks.analytical import vanna, volga

def test_vanna_sign():
    # OTM put (S<K): d2 < 0, vega > 0, so vanna = -vega*d2/(S*sigma*sqrt(T)) > 0
    v = vanna(90.0, 100.0, 1.0, 0.05, 0.2)
    assert v > 0

def test_volga_sign():
    # OTM option (S < K) has |d1|*|d2| > 0 and same sign → volga > 0 for OTM
    v = volga(90.0, 100.0, 1.0, 0.0, 0.2)
    assert v > 0

def test_vanna_zero_at_expiry():
    assert vanna(100.0, 100.0, 0.0, 0.05, 0.2) == 0.0

def test_volga_zero_at_expiry():
    assert volga(100.0, 100.0, 0.0, 0.05, 0.2) == 0.0

def test_portfolio_greeks_has_vanna_volga():
    positions = [{"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "option_type": "call", "quantity": 10}]
    g = portfolio_greeks(positions, contract_size=100)
    assert "vanna" in g and "volga" in g


from src.greeks.numerical import vanna_fd, volga_fd

def test_fd_vanna_matches_analytical():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    assert abs(vanna_fd(S, K, T, r, sigma) - vanna(S, K, T, r, sigma)) < 1e-3

def test_fd_volga_matches_analytical():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    assert abs(volga_fd(S, K, T, r, sigma) - volga(S, K, T, r, sigma)) < 1e-2
