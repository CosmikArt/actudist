"""Paralogistic severity distribution.

Klugman, Panjer & Willmot, *Loss Models* (5th ed.), Appendix A.2.1.5.

Special case of Burr XII with :math:`\\gamma=\\alpha`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, gammaln

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("Paralogistic")
class Paralogistic(SeverityDistribution):
    r"""Paralogistic with shape :math:`\alpha>0` and scale :math:`\theta>0`.

    .. math::
        F(x) = 1 - \Bigl(\frac{1}{1+(x/\theta)^{\alpha}}\Bigr)^{\alpha},\qquad
        f(x) = \frac{\alpha^{2}(x/\theta)^{\alpha}}{x\bigl(1+(x/\theta)^{\alpha}\bigr)^{\alpha+1}}.

    Mean is finite for :math:`\alpha>1`. Klugman 5e, Appendix A.2.1.5.
    """

    n_params = 2

    def __init__(
        self, alpha: float | None = None, theta: float | None = None
    ) -> None:
        if alpha is None and theta is None:
            super().__init__(params=None)
            return
        if alpha is None or theta is None:
            raise ValueError("Paralogistic needs both alpha and theta")
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
        return {"alpha": 2.0, "theta": max(float(np.median(arr)), 1e-6)}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        a, th = self.alpha, self.theta
        xv = x[m]
        log_z = np.log(xv) - np.log(th)
        with np.errstate(over="ignore"):
            log_pdf = (
                2.0 * np.log(a)
                + a * log_z
                - np.log(xv)
                - (a + 1.0) * np.log1p(np.exp(a * log_z))
            )
        out[m] = np.exp(np.maximum(log_pdf, -700.0))
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        a, th = self.alpha, self.theta
        with np.errstate(over="ignore"):
            za = (x[m] / th) ** a
        out[m] = 1.0 - (1.0 + za) ** (-a)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        a = self.alpha
        return self.theta * ((1.0 - q) ** (-1.0 / a) - 1.0) ** (1.0 / a)

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        return self.ppf(rng.uniform(size=size))

    def mean(self) -> float:
        a = self.alpha
        if a <= 1.0:
            return float("inf")
        return float(
            self.theta * np.exp(gammaln(1.0 + 1.0 / a) + gammaln(a - 1.0 / a) - gammaln(a))
        )

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        a, th = self.alpha, self.theta
        za = (d / th) ** a
        u = za / (1.0 + za)
        if a > 1.0:
            ratio = np.exp(
                gammaln(1.0 + 1.0 / a) + gammaln(a - 1.0 / a) - gammaln(a)
            )
            first = th * ratio * betainc(1.0 + 1.0 / a, a - 1.0 / a, u)
        else:
            from actudist._numerics import numeric_lev

            return numeric_lev(lambda x: float(self.survival_function(x)), float(d))
        second = d * (1.0 + za) ** (-a)
        return float(first + second)
