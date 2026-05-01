"""Lognormal severity distribution.

Klugman, Panjer & Willmot, *Loss Models* (5th ed.), Appendix A.2.3.5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("Lognormal")
class Lognormal(SeverityDistribution):
    r"""Lognormal with location :math:`\mu\in\mathbb R` and scale :math:`\sigma>0`.

    .. math::
        f(x) &= \frac{1}{x\sigma\sqrt{2\pi}}
                \exp\!\Bigl(-\tfrac{(\ln x - \mu)^2}{2\sigma^2}\Bigr),\\
        F(x) &= \Phi\!\bigl((\ln x - \mu)/\sigma\bigr),\\
        E[X] &= e^{\mu+\sigma^2/2},\\
        E[X\wedge d] &= e^{\mu+\sigma^2/2}\,\Phi\!\bigl((\ln d - \mu - \sigma^2)/\sigma\bigr)
                       + d\,\bigl(1 - \Phi((\ln d - \mu)/\sigma)\bigr).

    Klugman 5e, Appendix A.2.3.5.
    """

    n_params = 2

    def __init__(
        self, mu: float | None = None, sigma: float | None = None
    ) -> None:
        if mu is None and sigma is None:
            super().__init__(params=None)
            return
        if mu is None or sigma is None:
            raise ValueError("Lognormal needs both mu and sigma")
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0; got {sigma!r}")
        super().__init__(params={"mu": float(mu), "sigma": sigma})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("mu", "identity"), ("sigma", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        arr = arr[arr > 0]
        if arr.size == 0:
            return {"mu": 0.0, "sigma": 1.0}
        log = np.log(arr)
        return {"mu": float(log.mean()), "sigma": max(float(log.std(ddof=0)), 1e-3)}

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        z = (np.log(x[m]) - self.mu) / self.sigma
        out[m] = np.exp(-0.5 * z * z) / (x[m] * self.sigma * np.sqrt(2.0 * np.pi))
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x > 0
        out[m] = ndtr((np.log(x[m]) - self.mu) / self.sigma)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.exp(self.mu + self.sigma * ndtri(q))

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        return rng.lognormal(mean=self.mu, sigma=self.sigma, size=size)

    def mean(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma * self.sigma))

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        mu, sig = self.mu, self.sigma
        ld = np.log(d)
        first = float(np.exp(mu + 0.5 * sig * sig) * ndtr((ld - mu - sig * sig) / sig))
        second = float(d * (1.0 - ndtr((ld - mu) / sig)))
        return first + second
