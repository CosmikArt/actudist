"""Exponential severity. Single scale parameter :math:`\\theta = E[X]`."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from actudist.base import SeverityDistribution
from actudist.fitting import register_severity


@register_severity("Exponential")
class Exponential(SeverityDistribution):
    r"""Exponential distribution parameterized by mean :math:`\theta>0`.

    .. math::
        f(x) = \frac{1}{\theta} e^{-x/\theta},\qquad
        F(x) = 1 - e^{-x/\theta},\qquad
        E[X\wedge d] = \theta\bigl(1 - e^{-d/\theta}\bigr).

    Klugman, Loss Models 5e §A.2.3.1.
    """

    n_params = 1

    def __init__(self, theta: float | None = None) -> None:
        if theta is None:
            super().__init__(params=None)
        else:
            theta = float(theta)
            if theta <= 0:
                raise ValueError(f"theta must be > 0; got {theta!r}")
            super().__init__(params={"theta": theta})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("theta", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        return {"theta": max(float(arr.mean()), 1e-6)}

    # -- core functions ---------------------------------------------------

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        out[m] = np.exp(-x[m] / self.theta) / self.theta
        return out

    def cdf(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        m = x >= 0
        out[m] = -np.expm1(-x[m] / self.theta)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return -self.theta * np.log1p(-q)

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        return rng.exponential(scale=self.theta, size=size)

    def mean(self) -> float:
        return float(self.theta)

    def limited_expected_value(self, d: float) -> float:
        if d <= 0:
            return 0.0
        return float(self.theta * (-np.expm1(-d / self.theta)))
