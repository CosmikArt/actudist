"""Inverse Transformed Gamma severity distribution.

Klugman, Panjer & Willmot, *Loss Models* (5th ed.), Appendix A.2.2.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammainc, gammaincinv, gammaln

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("InverseTransformedGamma")
class InverseTransformedGamma(SeverityDistribution):
    r"""Inverse Transformed Gamma; shapes :math:`\alpha,\tau>0`, scale :math:`\theta>0`.

    Three-parameter parent of Inverse Gamma (:math:`\tau=1`), Inverse Weibull
    (:math:`\alpha=1`), and Inverse Exponential (:math:`\alpha=\tau=1`).

    .. math::
        f(x) &= \frac{\tau\,u^{\alpha}\,e^{-u}}{x\,\Gamma(\alpha)},
            \quad u=(\theta/x)^{\tau},\\
        F(x) &= 1 - \Gamma(\alpha;\,(\theta/x)^{\tau}),\\
        E[X] &= \theta\,\Gamma(\alpha-1/\tau)/\Gamma(\alpha),\quad(\alpha\tau>1),\\
        E[X\wedge d] &= \theta\,\frac{\Gamma(\alpha-1/\tau)}{\Gamma(\alpha)}
                       \bigl[1 - \Gamma(\alpha-1/\tau;\,(\theta/d)^{\tau})\bigr]
                       + d\,\Gamma(\alpha;\,(\theta/d)^{\tau}).

    Klugman 5e, Appendix A.2.2.2.
    """

    n_params = 3

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
            raise ValueError("InverseTransformedGamma needs alpha, theta, tau")
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
        m = max(float(np.median(arr)), 1e-6)
        return {"alpha": 2.0, "theta": m, "tau": 1.0}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        a, th, t = self.alpha, self.theta, self.tau
        xv = x[m]
        log_z = np.log(th) - np.log(xv)  # log(θ/x)
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
            u = (th / x[m]) ** t
        out[m] = 1.0 - gammainc(a, u)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        # F = 1 - gammainc(α, u) = q ⇒ gammainc(α, u) = 1 - q ⇒ u = gammaincinv(α, 1-q)
        # x = θ / u^(1/τ)
        u = gammaincinv(self.alpha, 1.0 - q)
        return self.theta / u ** (1.0 / self.tau)

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        y = rng.gamma(shape=self.alpha, scale=1.0, size=size)
        return self.theta / y ** (1.0 / self.tau)

    def mean(self) -> float:
        a, t = self.alpha, self.tau
        if a * t <= 1.0:
            return float("inf")
        return float(self.theta * np.exp(gammaln(a - 1.0 / t) - gammaln(a)))

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        a, th, t = self.alpha, self.theta, self.tau
        u = (th / d) ** t
        if a * t > 1.0:
            ratio = np.exp(gammaln(a - 1.0 / t) - gammaln(a))
            first = th * ratio * (1.0 - gammainc(a - 1.0 / t, u))
            second = d * gammainc(a, u)
            return float(first + second)
        from actudist._numerics import numeric_lev

        return numeric_lev(lambda x: float(self.survival_function(x)), float(d))
