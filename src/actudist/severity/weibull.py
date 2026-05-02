"""Weibull severity, scale-shape :math:`(\\theta, \\tau)`. Closed-form
LEV via the lower regularized incomplete gamma."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gamma as gammafn, gammainc

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("Weibull")
class Weibull(SeverityDistribution):
    r"""Weibull with scale :math:`\theta>0` and shape :math:`\tau>0`.

    .. math::
        f(x) &= \frac{\tau}{x}(x/\theta)^{\tau}\,e^{-(x/\theta)^{\tau}},\\
        F(x) &= 1 - e^{-(x/\theta)^{\tau}},\\
        E[X] &= \theta\,\Gamma(1+1/\tau),\\
        E[X\wedge d] &= \theta\,\Gamma(1+1/\tau)\,\Gamma(1+1/\tau;\,(d/\theta)^{\tau})
                       + d\,e^{-(d/\theta)^{\tau}},

    where :math:`\Gamma(a;x)` is the regularized lower incomplete gamma.
    Klugman 5e ch. 5; parameterization in Appendix A.
    """

    n_params = 2
    theta: float
    tau: float

    def __init__(self, theta: float | None = None, tau: float | None = None) -> None:
        if theta is None and tau is None:
            super().__init__(params=None)
            return
        if theta is None or tau is None:
            raise ValueError("Weibull needs both theta and tau")
        theta = float(theta)
        tau = float(tau)
        if theta <= 0 or tau <= 0:
            raise ValueError(f"theta, tau must be > 0; got {theta}, {tau}")
        super().__init__(params={"theta": theta, "tau": tau})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("theta", "log"), ("tau", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        m = max(float(arr.mean()), 1e-6)
        return {"theta": m, "tau": 1.0}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        th, t = self.theta, self.tau
        xv = x[m]
        log_z = np.log(xv) - np.log(th)
        with np.errstate(over="ignore"):
            log_pdf = np.log(t) - np.log(xv) + t * log_z - np.exp(t * log_z)
        out[m] = np.exp(np.maximum(log_pdf, -700.0))
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        out[m] = -np.expm1(-((x[m] / self.theta) ** self.tau))
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.theta * (-np.log1p(-q)) ** (1.0 / self.tau)

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
        return float(self.theta * gammafn(1.0 + 1.0 / self.tau))

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        th, t = self.theta, self.tau
        z = (d / th) ** t
        first = th * gammafn(1.0 + 1.0 / t) * gammainc(1.0 + 1.0 / t, z)
        second = d * np.exp(-z)
        return float(first + second)
