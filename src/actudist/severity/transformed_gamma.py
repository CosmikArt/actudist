"""Transformed Gamma. Three parameters :math:`(\\alpha, \\theta, \\tau)`:
the Gamma family with the variate raised to :math:`1/\\tau`."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammainc, gammaincinv, gammaln

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("TransformedGamma")
class TransformedGamma(SeverityDistribution):
    r"""Transformed Gamma with shapes :math:`\alpha,\tau>0` and scale :math:`\theta>0`.

    Three-parameter parent of Gamma (:math:`\tau=1`), Weibull (:math:`\alpha=1`),
    and Exponential (:math:`\alpha=\tau=1`).

    .. math::
        f(x) &= \frac{\tau\,u^{\alpha}\,e^{-u}}{x\,\Gamma(\alpha)},
            \quad u=(x/\theta)^{\tau},\\
        F(x) &= \Gamma(\alpha;\,(x/\theta)^{\tau}),\\
        E[X] &= \theta\,\Gamma(\alpha+1/\tau)/\Gamma(\alpha),\\
        E[X\wedge d] &= \theta\,\frac{\Gamma(\alpha+1/\tau)}{\Gamma(\alpha)}
                       \,\Gamma(\alpha+1/\tau;\,(d/\theta)^{\tau})
                       + d\,(1-\Gamma(\alpha;\,(d/\theta)^{\tau})).
    """

    n_params = 3
    alpha: float
    theta: float
    tau: float

    def __init__(
        self,
        alpha: float | None = None,
        theta: float | None = None,
        tau: float | None = None,
    ) -> None:
        if alpha is None and theta is None and tau is None:
            super().__init__(params=None)
            return
        if alpha is None or theta is None or tau is None:
            raise ValueError("TransformedGamma needs alpha, theta, tau")
        alpha = float(alpha)
        theta = float(theta)
        tau = float(tau)
        if alpha <= 0 or theta <= 0 or tau <= 0:
            raise ValueError(
                f"alpha, theta, tau must be > 0; got {alpha}, {theta}, {tau}"
            )
        super().__init__(params={"alpha": alpha, "theta": theta, "tau": tau})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("alpha", "log"), ("theta", "log"), ("tau", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        m = max(float(arr.mean()), 1e-6)
        return {"alpha": 2.0, "theta": m, "tau": 1.0}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        a, th, t = self.alpha, self.theta, self.tau
        xv = x[m]
        log_z = np.log(xv) - np.log(th)
        with np.errstate(over="ignore"):
            log_pdf = (
                np.log(t) + a * t * log_z - np.exp(t * log_z) - np.log(xv) - gammaln(a)
            )
        out[m] = np.exp(np.maximum(log_pdf, -700.0))
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        a, th, t = self.alpha, self.theta, self.tau
        with np.errstate(over="ignore"):
            u = (x[m] / th) ** t
        out[m] = gammainc(a, u)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.theta * gammaincinv(self.alpha, q) ** (1.0 / self.tau)

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        y = rng.gamma(shape=self.alpha, scale=1.0, size=size)
        return self.theta * y ** (1.0 / self.tau)

    def mean(self) -> float:
        a, t = self.alpha, self.tau
        return float(self.theta * np.exp(gammaln(a + 1.0 / t) - gammaln(a)))

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        a, th, t = self.alpha, self.theta, self.tau
        u = (d / th) ** t
        ratio = np.exp(gammaln(a + 1.0 / t) - gammaln(a))
        first = th * ratio * gammainc(a + 1.0 / t, u)
        second = d * (1.0 - gammainc(a, u))
        return float(first + second)
