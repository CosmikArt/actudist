"""Binomial frequency. ``m`` (trial count) is structural and held fixed;
only the success probability :math:`q` is fit."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

from actudist.base import FrequencyDistribution
from actudist.fitting import register_frequency


@register_frequency("Binomial")
class Binomial(FrequencyDistribution):
    r"""Binomial with trial count :math:`m\in\mathbb N` and success probability
    :math:`q\in(0,1)`.

    .. math::
        p_k = \binom{m}{k}q^{k}(1-q)^{m-k},\quad k=0,\dots,m.

    Klugman 5e, Section 6.6.4.
    """

    n_params = 1  # only q is fit; m is structural

    def __init__(self, m: int | None = None, q: float | None = None) -> None:
        if m is None and q is None:
            super().__init__(params=None)
            return
        if m is None or q is None:
            raise ValueError("Binomial needs both m and q")
        m_int = int(m)
        if m_int <= 0:
            raise ValueError(f"m must be a positive integer; got {m}")
        q = float(q)
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1); got {q}")
        super().__init__(params={"m": m_int, "q": q})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("q", "logit")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        m = max(int(arr.max()), 1)
        # MoM: q = mean / m
        q = float(np.clip(arr.mean() / m, 1e-6, 1.0 - 1e-6))
        return {"m": m, "q": q}

    def pmf(self, k: ArrayLike) -> np.ndarray:
        k = np.asarray(k)
        out = np.zeros(k.shape, dtype=float)
        valid = (k >= 0) & (k <= self.m) & (k == np.floor(k))
        if not np.any(valid):
            return out
        kv = k[valid].astype(float)
        log_pmf = (
            gammaln(self.m + 1.0)
            - gammaln(kv + 1.0)
            - gammaln(self.m - kv + 1.0)
            + kv * np.log(self.q)
            + (self.m - kv) * np.log1p(-self.q)
        )
        out[valid] = np.exp(np.maximum(log_pmf, -700.0))
        return out

    def cdf(self, k: ArrayLike) -> np.ndarray:
        from scipy.stats import binom as _sp_binom

        k = np.asarray(k)
        return np.asarray(_sp_binom.cdf(np.floor(k), n=self.m, p=self.q), dtype=float)

    def ppf(self, q: ArrayLike) -> np.ndarray:
        from scipy.stats import binom as _sp_binom

        return _sp_binom.ppf(np.asarray(q, dtype=float), n=self.m, p=self.q)

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        return rng.binomial(n=self.m, p=self.q, size=size)

    def mean(self) -> float:
        return float(self.m * self.q)

    def mle_fit(self, data: ArrayLike) -> dict[str, float]:
        """MLE with ``m`` fixed at ``max(data)``; for Binomial that is the
        smallest ``m`` consistent with the observations and the only choice
        the data identifies without external information."""
        arr = np.asarray(data, dtype=int)
        m = max(int(arr.max()), 1)
        q = float(np.clip(arr.mean() / m, 1e-6, 1.0 - 1e-6))
        self.params = {"m": m, "q": q}
        self._fitted = True
        self.m = m
        self.q = q
        return self.params
