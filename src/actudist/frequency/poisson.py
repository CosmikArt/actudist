"""Poisson frequency, single rate parameter :math:`\\lambda`."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

from actudist.base import FrequencyDistribution
from actudist.fitting import register_frequency


@register_frequency("Poisson")
class Poisson(FrequencyDistribution):
    r"""Poisson with rate :math:`\lambda>0`.

    .. math::
        p_k = \frac{e^{-\lambda}\lambda^{k}}{k!},\quad k=0,1,\dots

    Klugman 5e, Section 6.6.1.
    """

    n_params = 1

    def __init__(self, lam: float | None = None) -> None:
        if lam is None:
            super().__init__(params=None)
            return
        lam = float(lam)
        if lam <= 0:
            raise ValueError(f"lam must be > 0; got {lam}")
        super().__init__(params={"lam": lam})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("lam", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        return {"lam": max(float(arr.mean()), 1e-6)}

    def pmf(self, k: ArrayLike) -> np.ndarray:
        k = np.asarray(k)
        out = np.zeros(k.shape, dtype=float)
        valid = (k >= 0) & (k == np.floor(k))
        if not np.any(valid):
            return out
        kv = k[valid].astype(float)
        log_pmf = -self.lam + kv * np.log(self.lam) - gammaln(kv + 1.0)
        out[valid] = np.exp(np.maximum(log_pmf, -700.0))
        return out

    def cdf(self, k: ArrayLike) -> np.ndarray:
        k = np.asarray(k)
        from scipy.special import pdtr

        out = np.zeros(k.shape, dtype=float)
        m = k >= 0
        out[m] = pdtr(np.floor(k[m]).astype(float), self.lam)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.atleast_1d(np.asarray(q, dtype=float))
        from scipy.stats import poisson as _sp_poisson

        return _sp_poisson.ppf(q, mu=self.lam)

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        return rng.poisson(lam=self.lam, size=size)

    def mean(self) -> float:
        return float(self.lam)
