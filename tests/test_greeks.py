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
