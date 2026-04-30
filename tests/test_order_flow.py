import numpy as np
import pytest
from src.market.underlying import HestonSimulator


class TestHeston:
    def test_output_shape(self):
        sim = HestonSimulator(S0=100.0, v0=0.04, kappa=2.0, theta=0.04,
                              xi=0.3, rho=-0.7, r=0.0, seed=42)
        prices, vols = sim.simulate(n_steps=252, dt=1/252)
        assert len(prices) == 253  # n_steps + 1 (includes t=0)
        assert len(vols) == 253

    def test_variance_stays_positive(self):
        sim = HestonSimulator(S0=100.0, v0=0.04, kappa=2.0, theta=0.04,
                              xi=0.3, rho=-0.7, r=0.0, seed=42)
        _, vols = sim.simulate(n_steps=252, dt=1/252)
        assert np.all(np.array(vols) > 0)

    def test_price_positive(self):
        sim = HestonSimulator(S0=100.0, v0=0.04, kappa=2.0, theta=0.04,
                              xi=0.3, rho=-0.7, r=0.0, seed=42)
        prices, _ = sim.simulate(n_steps=252, dt=1/252)
        assert np.all(np.array(prices) > 0)

    def test_different_seeds_differ(self):
        s1 = HestonSimulator(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.0, seed=1)
        s2 = HestonSimulator(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.0, seed=2)
        p1, _ = s1.simulate(n_steps=100, dt=1/252)
        p2, _ = s2.simulate(n_steps=100, dt=1/252)
        assert not np.allclose(p1, p2)
