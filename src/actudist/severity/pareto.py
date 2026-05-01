"""Pareto (Type II / Lomax) severity distribution.

Klugman, Panjer & Willmot, *Loss Models* (5th ed.), Appendix A.2.3.4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("Pareto")
class Pareto(SeverityDistribution):
    r"""Pareto Type II (Lomax) with shape :math:`\alpha>0`, scale :math:`\theta>0`.

    .. math::
        f(x) &= \frac{\alpha\,\theta^{\alpha}}{(x+\theta)^{\alpha+1}},\\
        F(x) &= 1 - \Bigl(\frac{\theta}{x+\theta}\Bigr)^{\alpha},\\
        E[X] &= \frac{\theta}{\alpha-1}\quad(\alpha>1),\\
        E[X\wedge d] &= \frac{\theta}{\alpha-1}\Bigl(1 - (\theta/(d+\theta))^{\alpha-1}\Bigr)
                       \quad(\alpha\neq 1).

    Klugman 5e, Appendix A.2.3.4.
    """

    n_params = 2

    def __init__(
        self, alpha: float | None = None, theta: float | None = None
    ) -> None:
        if alpha is None and theta is None:
            super().__init__(params=None)
            return
        if alpha is None or theta is None:
            raise ValueError("Pareto needs both alpha and theta")
        alpha = float(alpha)
        theta = float(theta)
        if alpha <= 0 or theta <= 0:
            raise ValueError(f"alpha, theta must be > 0; got {alpha}, {theta}")
        super().__init__(params={"alpha": alpha, "theta": theta})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("alpha", "log"), ("theta", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        m = max(float(arr.mean()), 1e-6)
        v = max(float(arr.var()), m * m + 1e-6)
        # method of moments for Pareto: alpha = 2*v/(v - m^2), theta = m*(alpha-1)
        alpha = 2.0 * v / (v - m * m + 1e-12)
        alpha = float(np.clip(alpha, 1.5, 50.0))
        theta = m * (alpha - 1.0)
        theta = max(theta, 1e-3)
        return {"alpha": alpha, "theta": theta}

    # -- core functions ---------------------------------------------------

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        a, t = self.alpha, self.theta
        xx = x[m]
        out[m] = a * t**a / (xx + t) ** (a + 1.0)
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        a, t = self.alpha, self.theta
        out[m] = 1.0 - (t / (x[m] + t)) ** a
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.theta * ((1.0 - q) ** (-1.0 / self.alpha) - 1.0)

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
        if self.alpha <= 1.0:
            return float("inf")
        return float(self.theta / (self.alpha - 1.0))

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        a, t = self.alpha, self.theta
        if abs(a - 1.0) < 1e-12:
            return float(t * np.log1p(d / t))
        return float((t / (a - 1.0)) * (1.0 - (t / (d + t)) ** (a - 1.0)))
