import numpy as np
from typing import Tuple, List


class HestonSimulator:
    def __init__(self, S0: float, v0: float, kappa: float, theta: float,
                 xi: float, rho: float, r: float, seed: int = None):
        self.S0    = S0
        self.v0    = v0
        self.kappa = kappa
        self.theta = theta
        self.xi    = xi
        self.rho   = rho
        self.r     = r
        self.rng   = np.random.default_rng(seed)

    def simulate(self, n_steps: int, dt: float) -> Tuple[List[float], List[float]]:
        prices = [self.S0]
        vols   = [self.v0]
        S, v   = self.S0, self.v0
        corr   = np.array([[1.0, self.rho], [self.rho, 1.0]])
        L      = np.linalg.cholesky(corr)
        for _ in range(n_steps):
            z       = self.rng.standard_normal(2)
            dW      = L @ z * np.sqrt(dt)
            v_plus  = max(v, 0.0)
            dv      = self.kappa * (self.theta - v_plus) * dt + self.xi * np.sqrt(v_plus) * dW[1]
            v       = max(v + dv, 1e-8)
            dS      = self.r * S * dt + np.sqrt(v_plus) * S * dW[0]
            S       = S + dS
            prices.append(S)
            vols.append(v)
        return prices, vols
