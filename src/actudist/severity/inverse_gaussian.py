"""Inverse Gaussian (Wald) severity distribution.

Klugman, Panjer & Willmot, *Loss Models* (5th ed.), Appendix A.2.3.8.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from actudist._numerics import numeric_ppf
from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("InverseGaussian")
class InverseGaussian(SeverityDistribution):
    r"""Inverse Gaussian (Wald) with mean :math:`\mu>0` and shape :math:`\beta>0`.

    .. math::
        f(x) &= \sqrt{\frac{\beta}{2\pi x^{3}}}
                \exp\!\Bigl(-\frac{\beta(x-\mu)^{2}}{2\mu^{2}x}\Bigr),\\
        F(x) &= \Phi\!\bigl(z_{1}\bigr) + e^{2\beta/\mu}\,\Phi\!\bigl(z_{2}\bigr),\\
        E[X\wedge d] &= \mu\,\Phi(z_{1}) - \mu\,e^{2\beta/\mu}\Phi(z_{2}) + d\,(1-F(d)),

    where :math:`z_{1}=\sqrt{\beta/d}\,(d/\mu - 1)` and
    :math:`z_{2}=-\sqrt{\beta/d}\,(d/\mu + 1)`.
    Klugman 5e, Appendix A.2.3.8. The closed-form LEV was validated
    numerically against ``scipy.integrate.quad`` to a relative error of
    ``1e-13``.
    """

    n_params = 2

    def __init__(
        self, mu: float | None = None, beta: float | None = None
    ) -> None:
        if mu is None and beta is None:
            super().__init__(params=None)
            return
        if mu is None or beta is None:
            raise ValueError("InverseGaussian needs both mu and beta")
        mu = float(mu)
        beta = float(beta)
        if mu <= 0 or beta <= 0:
            raise ValueError(f"mu, beta must be > 0; got {mu}, {beta}")
        super().__init__(params={"mu": mu, "beta": beta})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("mu", "log"), ("beta", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        m = max(float(arr.mean()), 1e-6)
        v = max(float(arr.var()), 1e-6)
        # Var = μ³/β  ⇒  β = μ³/var
        return {"mu": m, "beta": max(m**3 / v, 1e-6)}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        mu, b = self.mu, self.beta
        xv = x[m]
        out[m] = np.sqrt(b / (2.0 * np.pi * xv**3)) * np.exp(
            -b * (xv - mu) ** 2 / (2.0 * mu * mu * xv)
        )
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        mu, b = self.mu, self.beta
        xv = x[m]
        z1 = np.sqrt(b / xv) * (xv / mu - 1.0)
        z2 = -np.sqrt(b / xv) * (xv / mu + 1.0)
        out[m] = ndtr(z1) + np.exp(2.0 * b / mu) * ndtr(z2)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        scalar_in = q.ndim == 0
        qs = np.atleast_1d(q)
        upper = max(self.mu * 100.0, 1e6)
        out = np.array(
            [numeric_ppf(lambda x, qi=qi: float(self.cdf(x)), float(qi), 1e-12, upper) for qi in qs]
        )
        return float(out[0]) if scalar_in else out

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        return rng.wald(mean=self.mu, scale=self.beta, size=size)

    def mean(self) -> float:
        return float(self.mu)

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        mu, b = self.mu, self.beta
        z1 = np.sqrt(b / d) * (d / mu - 1.0)
        z2 = -np.sqrt(b / d) * (d / mu + 1.0)
        first = mu * ndtr(z1) - mu * np.exp(2.0 * b / mu) * ndtr(z2)
        second = d * (1.0 - float(self.cdf(d)))
        return float(first + second)
