"""Log-Logistic severity. The Burr XII family with :math:`\\alpha=1`,
exposed as its own class so callers don't have to fix a parameter."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, gammaln

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("LogLogistic")
class LogLogistic(SeverityDistribution):
    r"""Log-Logistic with shape :math:`\gamma>0` and scale :math:`\theta>0`.

    .. math::
        f(x) &= \frac{\gamma\,(x/\theta)^{\gamma}}{x\,\bigl(1+(x/\theta)^{\gamma}\bigr)^{2}},\\
        F(x) &= \frac{(x/\theta)^{\gamma}}{1+(x/\theta)^{\gamma}},\\
        E[X] &= \theta\,\Gamma(1+1/\gamma)\,\Gamma(1-1/\gamma),\quad(\gamma>1).

    Klugman §A.2.1.4 (Burr XII with α=1).
    """

    n_params = 2

    def __init__(
        self, theta: float | None = None, gamma: float | None = None
    ) -> None:
        if theta is None and gamma is None:
            super().__init__(params=None)
            return
        if theta is None or gamma is None:
            raise ValueError("LogLogistic needs both theta and gamma")
        theta = float(theta)
        gamma = float(gamma)
        if theta <= 0 or gamma <= 0:
            raise ValueError(f"theta, gamma must be > 0; got {theta}, {gamma}")
        super().__init__(params={"theta": theta, "gamma": gamma})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("theta", "log"), ("gamma", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        return {"theta": max(float(np.median(arr)), 1e-6), "gamma": 2.0}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        th, g = self.theta, self.gamma
        xv = x[m]
        log_z = np.log(xv) - np.log(th)
        with np.errstate(over="ignore"):
            log_pdf = (
                np.log(g) + g * log_z - np.log(xv) - 2.0 * np.log1p(np.exp(g * log_z))
            )
        out[m] = np.exp(np.maximum(log_pdf, -700.0))
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        with np.errstate(over="ignore"):
            zg = (x[m] / self.theta) ** self.gamma
        out[m] = zg / (1.0 + zg)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.theta * (q / (1.0 - q)) ** (1.0 / self.gamma)

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        u = rng.uniform(size=size)
        return self.ppf(u)

    def mean(self) -> float:
        if self.gamma <= 1.0:
            return float("inf")
        # θ Γ(1+1/γ) Γ(1-1/γ)
        return float(
            self.theta
            * np.exp(gammaln(1.0 + 1.0 / self.gamma) + gammaln(1.0 - 1.0 / self.gamma))
        )

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        th, g = self.theta, self.gamma
        zg = (d / th) ** g
        u = zg / (1.0 + zg)
        if g > 1.0:
            ratio = np.exp(
                gammaln(1.0 + 1.0 / g) + gammaln(1.0 - 1.0 / g)
            )
            first = th * ratio * betainc(1.0 + 1.0 / g, 1.0 - 1.0 / g, u)
        else:
            from actudist._numerics import numeric_lev

            return numeric_lev(lambda x: float(self.survival_function(x)), float(d))
        second = d * (1.0 - zg / (1.0 + zg))
        return float(first + second)
