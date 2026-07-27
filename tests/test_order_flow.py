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
            S_true=100.0, S_stale=100.0, bid=99.0, ask=101.0, dt_days=1/78
        )
        # No informed traders when prices match
        for t in trades:
            assert t["trader_type"] == "noise"

    def test_informed_hit_correct_side_call(self):
        # S_true > S_stale: the call is worth MORE than the MM's stale fair, so the
        # informed trader lifts the ask.
        trades = self.sim.generate_trades(
            S_true=101.0, S_stale=100.0, bid=99.5, ask=100.5, dt_days=1/78,
            option_type="call",
        )
        informed = [t for t in trades if t["trader_type"] == "informed"]
        assert len(informed) == 1, "staleness above threshold must produce one informed trade"
        assert informed[0]["side"] == "buy"

    def test_informed_hit_correct_side_put(self):
        # Same underlying move, but a PUT is worth LESS when S rises, so the informed
        # trader must hit the bid. Deciding the side from the underlying move alone
        # (the pre-fix behaviour) would say "buy" here, which is backwards.
        trades = self.sim.generate_trades(
            S_true=101.0, S_stale=100.0, bid=99.5, ask=100.5, dt_days=1/78,
            option_type="put",
        )
        informed = [t for t in trades if t["trader_type"] == "informed"]
        assert len(informed) == 1
        assert informed[0]["side"] == "sell"

    def test_informed_side_reverses_with_underlying(self):
        # S_true < S_stale reverses both: calls sold, puts bought.
        for otype, expected in (("call", "sell"), ("put", "buy")):
            trades = self.sim.generate_trades(
                S_true=99.0, S_stale=100.0, bid=99.5, ask=100.5, dt_days=1/78,
                option_type=otype,
            )
            informed = [t for t in trades if t["trader_type"] == "informed"]
            assert len(informed) == 1
            assert informed[0]["side"] == expected, f"{otype} on a down move"

    def test_trade_has_required_fields(self):
        trades = self.sim.generate_trades(
            S_true=101.0, S_stale=100.0, bid=99.0, ask=101.0, dt_days=1.0
        )
        assert trades, "an above-threshold move must generate at least the informed trade"
        for t in trades:
            assert "side" in t and "size" in t and "price" in t and "trader_type" in t

    def test_informed_larger_than_noise(self):
        # Run many steps and check informed avg size > noise avg size
        noise_sizes, informed_sizes = [], []
        for _ in range(1000):
            trades = self.sim.generate_trades(
                S_true=101.0, S_stale=100.0, bid=99.0, ask=101.0, dt_days=1/78
            )
            for t in trades:
                if t["trader_type"] == "noise":
                    noise_sizes.append(t["size"])
                else:
                    informed_sizes.append(t["size"])
        assert noise_sizes and informed_sizes, "both populations must actually arrive"
        import numpy as np
        assert np.mean(informed_sizes) > np.mean(noise_sizes)

    def test_noise_arrival_rate_is_per_day(self):
        # lambda_noise is documented as a per-DAY rate, so over one full day's worth of
        # steps the realized count must be near lambda_noise, not 252x smaller.
        steps, lam = 78, self.sim.lambda_noise
        counts = [
            len([t for t in self.sim.generate_trades(
                S_true=100.0, S_stale=100.0, bid=99.0, ask=101.0, dt_days=1/steps
            ) if t["trader_type"] == "noise"])
            for _ in range(steps * 200)
        ]
        realized_per_day = sum(counts) / 200
        assert 0.8 * lam < realized_per_day < 1.2 * lam, (
            f"expected ~{lam} noise arrivals/day, realized {realized_per_day:.2f}"
        )

    def test_informed_stays_out_when_edge_below_half_spread(self):
        # Edge of $0.05 against a $0.50 half-spread: crossing is a certain loss, so the
        # informed trader must not arrive even though the staleness trigger has fired.
        trades = self.sim.generate_trades(
            S_true=101.0, S_stale=100.0, bid=99.5, ask=100.5, dt_days=1/78,
            option_type="call", option_edge=0.05,
        )
        assert not [t for t in trades if t["trader_type"] == "informed"]

    def test_informed_trades_when_edge_exceeds_half_spread(self):
        trades = self.sim.generate_trades(
            S_true=101.0, S_stale=100.0, bid=99.5, ask=100.5, dt_days=1/78,
            option_type="call", option_edge=2.00,
        )
        informed = [t for t in trades if t["trader_type"] == "informed"]
        assert len(informed) == 1 and informed[0]["side"] == "buy"

    def test_dollar_edge_sets_the_side_directly(self):
        # A negative dollar edge means the option is worth LESS than the stale fair,
        # so the informed trader hits the bid regardless of the underlying's direction.
        trades = self.sim.generate_trades(
            S_true=101.0, S_stale=100.0, bid=99.5, ask=100.5, dt_days=1/78,
            option_type="call", option_edge=-2.00,
        )
        informed = [t for t in trades if t["trader_type"] == "informed"]
        assert len(informed) == 1 and informed[0]["side"] == "sell"

    def test_noise_fill_probability_falls_with_quote_width(self):
        """The reservation price must make noise flow ELASTIC to quote width.

        Without it, fill counts were literally invariant across a 16x width sweep and
        P&L was linear and unbounded in a quantity the market maker chooses.
        """
        def noise_count(half_spread):
            n = 0
            for _ in range(4000):
                ts = self.sim.generate_trades(
                    S_true=100.0, S_stale=100.0,
                    bid=10.0 - half_spread, ask=10.0 + half_spread,
                    dt_days=1.0, option_type="call", option_edge=0.0,
                    reservation_scale=0.25,
                )
                n += len([t for t in ts if t["trader_type"] == "noise"])
            return n

        tight, wide = noise_count(0.05), noise_count(0.80)
        assert wide < tight * 0.5, f"tight {tight} vs wide {wide}: flow is not elastic"

    def test_no_reservation_scale_leaves_flow_inelastic(self):
        # Backwards-compatible path: omitting the scale disables the gate entirely.
        n_tight = n_wide = 0
        for _ in range(2000):
            n_tight += len([t for t in self.sim.generate_trades(
                S_true=100.0, S_stale=100.0, bid=9.95, ask=10.05, dt_days=1.0) 
                if t["trader_type"] == "noise"])
            n_wide += len([t for t in self.sim.generate_trades(
                S_true=100.0, S_stale=100.0, bid=9.20, ask=10.80, dt_days=1.0)
                if t["trader_type"] == "noise"])
        assert abs(n_tight - n_wide) / max(n_tight, 1) < 0.10
