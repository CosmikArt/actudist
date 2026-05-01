"""Transformed Beta (4-parameter) severity distribution.

Klugman, Panjer & Willmot, *Loss Models* (5th ed.), Appendix A.2.1.1.

Parent of the Burr (alpha-1) family: Burr (tau=1), Inverse Burr (alpha=1),
Pareto (gamma=tau=1), Loglogistic (alpha=tau=1), Paralogistic (tau=1, gamma=alpha),
Inverse Paralogistic (alpha=1, gamma=tau), Generalized Pareto (gamma=1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, betaincinv, gammaln

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("TransformedBeta")
class TransformedBeta(SeverityDistribution):
    r"""Transformed Beta with shapes :math:`\alpha,\gamma,\tau>0` and scale :math:`\theta>0`.

    .. math::
        f(x) &= \frac{\Gamma(\alpha+\tau)}{\Gamma(\alpha)\Gamma(\tau)}
                \,\frac{\gamma\,(x/\theta)^{\gamma\tau}}
                       {x\,(1+(x/\theta)^{\gamma})^{\alpha+\tau}},\\
        F(x) &= \beta(\tau,\alpha;\,u),\qquad u=(x/\theta)^{\gamma}/(1+(x/\theta)^{\gamma}),\\
        E[X] &= \theta\,\frac{\Gamma(\tau+1/\gamma)\Gamma(\alpha-1/\gamma)}
                              {\Gamma(\tau)\Gamma(\alpha)},\quad(\alpha\gamma>1),\\
        E[X\wedge d] &= \theta\,\frac{\Gamma(\tau+1/\gamma)\Gamma(\alpha-1/\gamma)}
                                     {\Gamma(\tau)\Gamma(\alpha)}
                       \,\beta(\tau+1/\gamma,\,\alpha-1/\gamma;\,u)
                     + d\,(1-F(d)).

    Klugman 5e, Appendix A.2.1.1.
    """

    n_params = 4

    def __init__(
        self,
        alpha: float | None = None,
        theta: float | None = None,
        gamma: float | None = None,
        tau: float | None = None,
    ) -> None:
        if alpha is None and theta is None and gamma is None and tau is None:
            super().__init__(params=None)
            return
        if alpha is None or theta is None or gamma is None or tau is None:
            raise ValueError("TransformedBeta needs alpha, theta, gamma, tau")
        for nm, v in [("alpha", alpha), ("theta", theta), ("gamma", gamma), ("tau", tau)]:
            if v <= 0:
                raise ValueError(f"{nm} must be > 0; got {v}")
        super().__init__(
            params={
                "alpha": float(alpha),
                "theta": float(theta),
                "gamma": float(gamma),
                "tau": float(tau),
            }
        )

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [
            ("alpha", "log"),
            ("theta", "log"),
            ("gamma", "log"),
            ("tau", "log"),
        ]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        return {
            "alpha": 2.0,
            "theta": max(float(np.median(arr)), 1e-6),
            "gamma": 2.0,
            "tau": 1.0,
        }

    def _log_norm_const(self) -> float:
        return float(gammaln(self.alpha + self.tau) - gammaln(self.alpha) - gammaln(self.tau))

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        a, th, g, t = self.alpha, self.theta, self.gamma, self.tau
        xv = x[m]
        log_z = np.log(xv) - np.log(th)  # log(x/θ)
        with np.errstate(over="ignore"):
            log_pdf = (
                self._log_norm_const()
                + np.log(g)
                + g * t * log_z
                - np.log(xv)
                - (a + t) * np.log1p(np.exp(g * log_z))
            )
        out[m] = np.exp(np.maximum(log_pdf, -700.0))
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        a, th, g, t = self.alpha, self.theta, self.gamma, self.tau
        with np.errstate(over="ignore"):
            zg = (x[m] / th) ** g
        u = zg / (1.0 + zg)
        out[m] = betainc(t, a, u)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        u = betaincinv(self.tau, self.alpha, q)
        return self.theta * (u / (1.0 - u)) ** (1.0 / self.gamma)

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        u = rng.beta(a=self.tau, b=self.alpha, size=size)
        return self.theta * (u / (1.0 - u)) ** (1.0 / self.gamma)

    def _gamma_ratio(self) -> float:
        a, g, t = self.alpha, self.gamma, self.tau
        return float(
            np.exp(
                gammaln(t + 1.0 / g)
                + gammaln(a - 1.0 / g)
                - gammaln(t)
                - gammaln(a)
            )
        )

    def mean(self) -> float:
        if self.alpha * self.gamma <= 1.0:
            return float("inf")
        return float(self.theta * self._gamma_ratio())

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        a, th, g, t = self.alpha, self.theta, self.gamma, self.tau
        zg = (d / th) ** g
        u = zg / (1.0 + zg)
        if a * g > 1.0:
            first = th * self._gamma_ratio() * betainc(t + 1.0 / g, a - 1.0 / g, u)
            second = d * (1.0 - betainc(t, a, u))
            return float(first + second)
        from actudist._numerics import numeric_lev

        return numeric_lev(lambda x: float(self.survival_function(x)), float(d))
