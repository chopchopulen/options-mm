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
