"""Gamma severity. Shape :math:`\\alpha`, scale :math:`\\theta`. LEV via
the lower regularized incomplete gamma."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammainc, gammaincinv, gammaln

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("Gamma")
class Gamma(SeverityDistribution):
    r"""Gamma with shape :math:`\alpha>0` and scale :math:`\theta>0`.

    .. math::
        f(x) &= \frac{x^{\alpha-1} e^{-x/\theta}}{\theta^{\alpha}\Gamma(\alpha)},\\
        F(x) &= \Gamma(\alpha;\,x/\theta),\\
        E[X] &= \alpha\theta,\\
        E[X\wedge d] &= \alpha\theta\,\Gamma(\alpha+1;\,d/\theta)
                       + d\,\bigl(1 - \Gamma(\alpha;\,d/\theta)\bigr),

    where :math:`\Gamma(a;x)` is the regularized lower incomplete gamma.
    """

    n_params = 2
    alpha: float
    theta: float

    def __init__(self, alpha: float | None = None, theta: float | None = None) -> None:
        if alpha is None and theta is None:
            super().__init__(params=None)
            return
        if alpha is None or theta is None:
            raise ValueError("Gamma needs both alpha and theta")
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
        v = max(float(arr.var()), 1e-6)
        # MoM: mean = αθ, var = αθ² => θ = v/m, α = m/θ
        theta = v / m
        alpha = m / theta
        return {"alpha": max(alpha, 1e-3), "theta": max(theta, 1e-6)}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        a, th = self.alpha, self.theta
        xv = x[m]
        log_pdf = (a - 1.0) * np.log(xv) - xv / th - a * np.log(th) - gammaln(a)
        out[m] = np.exp(np.maximum(log_pdf, -700.0))
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        out[m] = gammainc(self.alpha, x[m] / self.theta)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.theta * gammaincinv(self.alpha, q)

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        return rng.gamma(shape=self.alpha, scale=self.theta, size=size)

    def mean(self) -> float:
        return float(self.alpha * self.theta)

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        a, th = self.alpha, self.theta
        z = d / th
        first = a * th * gammainc(a + 1.0, z)
        second = d * (1.0 - gammainc(a, z))
        return float(first + second)
