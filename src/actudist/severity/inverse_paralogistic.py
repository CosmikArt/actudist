"""Inverse Paralogistic severity. Special case of the Inverse Burr family
with :math:`\\gamma=\\tau`."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, gammaln

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("InverseParalogistic")
class InverseParalogistic(SeverityDistribution):
    r"""Inverse Paralogistic with shape :math:`\tau>0` and scale :math:`\theta>0`.

    .. math::
        F(x) = \Bigl(\frac{(x/\theta)^{\tau}}{1+(x/\theta)^{\tau}}\Bigr)^{\tau},\qquad
        f(x) = \frac{\tau^{2}(x/\theta)^{\tau^{2}}}
                    {x\,\bigl(1+(x/\theta)^{\tau}\bigr)^{\tau+1}}.

    Mean is finite for :math:`\tau>1`. Inverse Burr with γ=τ; Klugman §A.2.2.3.
    """

    n_params = 2
    tau: float
    theta: float

    def __init__(self, tau: float | None = None, theta: float | None = None) -> None:
        if tau is None and theta is None:
            super().__init__(params=None)
            return
        if tau is None or theta is None:
            raise ValueError("InverseParalogistic needs both tau and theta")
        tau = float(tau)
        theta = float(theta)
        if tau <= 0 or theta <= 0:
            raise ValueError(f"tau, theta must be > 0; got {tau}, {theta}")
        super().__init__(params={"tau": tau, "theta": theta})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("tau", "log"), ("theta", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        return {"tau": 2.0, "theta": max(float(np.median(arr)), 1e-6)}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        if not np.any(m):
            return out
        t, th = self.tau, self.theta
        xv = x[m]
        log_z = np.log(xv) - np.log(th)  # log(x/θ)
        with np.errstate(over="ignore"):
            log_pdf = (
                2.0 * np.log(t)
                + (t * t) * log_z
                - np.log(xv)
                - (t + 1.0) * np.log1p(np.exp(t * log_z))
            )
        out[m] = np.exp(np.maximum(log_pdf, -700.0))
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        t, th = self.tau, self.theta
        with np.errstate(over="ignore"):
            zt = (x[m] / th) ** t
        out[m] = (zt / (1.0 + zt)) ** t
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        t = self.tau
        # F = u^t where u = z^t/(1+z^t); solving: u = q^(1/t) → z^t = u/(1-u)
        u = q ** (1.0 / t)
        return self.theta * (u / (1.0 - u)) ** (1.0 / t)

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
        t = self.tau
        if t <= 1.0:
            return float("inf")
        # E[X] = θ Γ(τ+1/τ) Γ(1-1/τ) / Γ(τ)
        return float(
            self.theta
            * np.exp(gammaln(t + 1.0 / t) + gammaln(1.0 - 1.0 / t) - gammaln(t))
        )

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        t, th = self.tau, self.theta
        zt = (d / th) ** t
        u = zt / (1.0 + zt)
        if t > 1.0:
            ratio = np.exp(gammaln(t + 1.0 / t) + gammaln(1.0 - 1.0 / t) - gammaln(t))
            first = th * ratio * betainc(t + 1.0 / t, 1.0 - 1.0 / t, u)
        else:
            from actudist._numerics import numeric_lev

            return numeric_lev(lambda x: float(self.survival_function(x)), float(d))
        second = d * (1.0 - u**t)
        return float(first + second)
