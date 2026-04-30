import numpy as np
import pytest
from src.market.underlying import HestonSimulator
from src.market.order_flow import OrderFlowSimulator


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


class TestOrderFlow:
    def setup_method(self):
        self.sim = OrderFlowSimulator(
            lambda_noise=10.0,
            max_noise_size=5,
            min_informed_size=3,
            max_informed_size=15,
            staleness_threshold=0.002,
            seed=42,
        )

    def test_no_informed_when_prices_equal(self):
        trades = self.sim.generate_trades(
            S_true=100.0, S_stale=100.0, bid=99.0, ask=101.0, dt=1/252
        )
        # No informed traders when prices match
        for t in trades:
            assert t["trader_type"] == "noise"

    def test_informed_hit_correct_side(self):
        # S_true > S_stale: call is underpriced by MM, informed buys (hits ask)
        trades = self.sim.generate_trades(
            S_true=101.0, S_stale=100.0, bid=99.5, ask=100.5, dt=1/252
        )
        informed = [t for t in trades if t["trader_type"] == "informed"]
        if informed:
            # All informed should be buying (hitting ask) when S_true > S_stale
            for t in informed:
                assert t["side"] == "buy"

    def test_trade_has_required_fields(self):
        trades = self.sim.generate_trades(
            S_true=100.0, S_stale=100.0, bid=99.0, ask=101.0, dt=1.0
        )
        if trades:
            for t in trades:
                assert "side" in t and "size" in t and "price" in t and "trader_type" in t

    def test_informed_larger_than_noise(self):
        # Run many steps and check informed avg size > noise avg size
        noise_sizes, informed_sizes = [], []
        for _ in range(1000):
            trades = self.sim.generate_trades(
                S_true=101.0, S_stale=100.0, bid=99.0, ask=101.0, dt=1/252
            )
            for t in trades:
                if t["trader_type"] == "noise":
                    noise_sizes.append(t["size"])
                else:
                    informed_sizes.append(t["size"])
        if noise_sizes and informed_sizes:
            import numpy as np
            assert np.mean(informed_sizes) > np.mean(noise_sizes)
