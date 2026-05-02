"""Burr Type XII severity. Closed-form LEV via the regularized incomplete
beta function. See Klugman §A.2.1.2 for the parameterization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, gammaln

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("BurrXII")
class BurrXII(SeverityDistribution):
    r"""Burr Type XII with shape :math:`\alpha,\gamma>0` and scale :math:`\theta>0`.

    .. math::
        f(x) &= \frac{\alpha\gamma\,(x/\theta)^{\gamma}}
                     {x\,\bigl(1+(x/\theta)^{\gamma}\bigr)^{\alpha+1}},\\
        F(x) &= 1 - \Bigl(\frac{1}{1+(x/\theta)^{\gamma}}\Bigr)^{\alpha},\\
        E[X] &= \theta\,\frac{\Gamma(1+1/\gamma)\,\Gamma(\alpha-1/\gamma)}{\Gamma(\alpha)},
                \quad(\alpha\gamma>1).

    The LEV uses the regularized incomplete beta:

    .. math::
        E[X\wedge d] = \theta\,\frac{\Gamma(1+1/\gamma)\,\Gamma(\alpha-1/\gamma)}{\Gamma(\alpha)}
                       \,\beta(1+1/\gamma,\,\alpha-1/\gamma;\,u)
                     + d\,\bigl(1+(d/\theta)^{\gamma}\bigr)^{-\alpha},

    where :math:`u=(d/\theta)^{\gamma}/(1+(d/\theta)^{\gamma})`.
    Klugman 5e, Appendix A.2.1.2.
    """

    n_params = 3

    def __init__(
        self,
        alpha: float | None = None,
        theta: float | None = None,
        gamma: float | None = None,
    ) -> None:
        if alpha is None and theta is None and gamma is None:
            super().__init__(params=None)
            return
        if alpha is None or theta is None or gamma is None:
            raise ValueError("BurrXII needs alpha, theta, gamma")
        alpha = float(alpha)
        theta = float(theta)
        gamma = float(gamma)
        if alpha <= 0 or theta <= 0 or gamma <= 0:
            raise ValueError(
                f"alpha, theta, gamma must be > 0; got {alpha}, {theta}, {gamma}"
            )
        super().__init__(params={"alpha": alpha, "theta": theta, "gamma": gamma})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("alpha", "log"), ("theta", "log"), ("gamma", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        m = max(float(arr.mean()), 1e-6)
        return {"alpha": 2.0, "theta": m, "gamma": 2.0}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        a, th, g = self.alpha, self.theta, self.gamma
        xv = x[m]
        log_z = np.log(xv) - np.log(th)  # log(x/θ)
        with np.errstate(over="ignore"):
            log_pdf = (
                np.log(a)
                + np.log(g)
                + g * log_z
                - np.log(xv)
                - (a + 1.0) * np.log1p(np.exp(g * log_z))
            )
        out[m] = np.exp(np.maximum(log_pdf, -700.0))
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        a, th, g = self.alpha, self.theta, self.gamma
        with np.errstate(over="ignore"):
            zg = (x[m] / th) ** g
        out[m] = 1.0 - (1.0 + zg) ** (-a)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.theta * ((1.0 - q) ** (-1.0 / self.alpha) - 1.0) ** (
            1.0 / self.gamma
        )

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

    def _gamma_ratio_log(self) -> float:
        # log[ Γ(1+1/γ) Γ(α-1/γ) / Γ(α) ]
        return float(
            gammaln(1.0 + 1.0 / self.gamma)
            + gammaln(self.alpha - 1.0 / self.gamma)
            - gammaln(self.alpha)
        )

    def mean(self) -> float:
        if self.alpha * self.gamma <= 1.0:
            return float("inf")
        return float(self.theta * np.exp(self._gamma_ratio_log()))

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        a, th, g = self.alpha, self.theta, self.gamma
        zg = (d / th) ** g
        u = zg / (1.0 + zg)
        if a * g > 1.0:
            ratio = np.exp(self._gamma_ratio_log())
            first = th * ratio * betainc(1.0 + 1.0 / g, a - 1.0 / g, u)
        else:
            # mean is infinite; fall back to numeric LEV via S(x) integral
            from actudist._numerics import numeric_lev

            return numeric_lev(lambda x: float(self.survival_function(x)), float(d))
        second = d * (1.0 + zg) ** (-a)
        return float(first + second)
