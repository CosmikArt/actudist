"""Geometric frequency. NegativeBinomial with :math:`r=1` (Klugman §6.6.3)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from actudist.base import FrequencyDistribution
from actudist.fitting import register_frequency


@register_frequency("Geometric")
class Geometric(FrequencyDistribution):
    r"""Geometric (NegBin with r=1) parameterized by :math:`\beta>0`.

    .. math::
        p_k = \frac{\beta^{k}}{(1+\beta)^{k+1}},\quad k=0,1,\dots

    Mean :math:`=\beta`. Klugman 5e, Section 6.6.3.
    """

    n_params = 1
    beta: float

    def __init__(self, beta: float | None = None) -> None:
        if beta is None:
            super().__init__(params=None)
            return
        beta = float(beta)
        if beta <= 0:
            raise ValueError(f"beta must be > 0; got {beta}")
        super().__init__(params={"beta": beta})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("beta", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        return {"beta": max(float(arr.mean()), 1e-6)}

    def pmf(self, k: ArrayLike) -> np.ndarray:
        k = np.asarray(k)
        out = np.zeros(k.shape, dtype=float)
        valid = (k >= 0) & (k == np.floor(k))
        if not np.any(valid):
            return out
        kv = k[valid].astype(float)
        b = self.beta
        log_pmf = kv * np.log(b) - (kv + 1.0) * np.log1p(b)
        out[valid] = np.exp(np.maximum(log_pmf, -700.0))
        return out

    def cdf(self, k: ArrayLike) -> np.ndarray:
        k = np.asarray(k)
        out = np.zeros(k.shape, dtype=float)
        m = k >= 0
        # F(k) = 1 - (β/(1+β))^(floor(k)+1)
        kv = np.floor(k[m]).astype(float)
        out[m] = 1.0 - (self.beta / (1.0 + self.beta)) ** (kv + 1.0)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        from scipy.stats import geom as _sp_geom

        # scipy's geom is on {1,2,...}; shift by -1 to get Klugman support {0,1,...}.
        # It uses success probability p = 1/(1+β).
        q = np.asarray(q, dtype=float)
        return _sp_geom.ppf(q, p=1.0 / (1.0 + self.beta)) - 1.0

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        # numpy's geometric is on {1,2,...} with success probability p
        return rng.geometric(p=1.0 / (1.0 + self.beta), size=size) - 1

    def mean(self) -> float:
        return float(self.beta)
