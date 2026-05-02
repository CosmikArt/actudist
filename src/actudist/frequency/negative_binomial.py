"""Negative Binomial frequency in Klugman's :math:`(r, \\beta)`
parameterization, with ``r`` a positive real."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

from actudist.base import FrequencyDistribution
from actudist.fitting import register_frequency


@register_frequency("NegativeBinomial")
class NegativeBinomial(FrequencyDistribution):
    r"""Negative Binomial in Klugman's (r, β) parameterization.

    .. math::
        p_k = \binom{r+k-1}{k}\Bigl(\frac{1}{1+\beta}\Bigr)^{r}
              \Bigl(\frac{\beta}{1+\beta}\Bigr)^{k},\quad k=0,1,\dots

    with :math:`r,\beta>0`. Mean :math:`= r\beta`, variance :math:`= r\beta(1+\beta)`.
    Klugman 5e, Section 6.6.2.
    """

    n_params = 2
    r: float
    beta: float

    def __init__(self, r: float | None = None, beta: float | None = None) -> None:
        if r is None and beta is None:
            super().__init__(params=None)
            return
        if r is None or beta is None:
            raise ValueError("NegativeBinomial needs both r and beta")
        r = float(r)
        beta = float(beta)
        if r <= 0 or beta <= 0:
            raise ValueError(f"r, beta must be > 0; got {r}, {beta}")
        super().__init__(params={"r": r, "beta": beta})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("r", "log"), ("beta", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        m = max(float(arr.mean()), 1e-6)
        v = max(float(arr.var()), m + 1e-6)
        # MoM: m = rβ, v = rβ(1+β) ⇒ β = v/m - 1, r = m/β
        beta = max(v / m - 1.0, 1e-3)
        r = max(m / beta, 1e-3)
        return {"r": r, "beta": beta}

    def pmf(self, k: ArrayLike) -> np.ndarray:
        k = np.asarray(k)
        out = np.zeros(k.shape, dtype=float)
        valid = (k >= 0) & (k == np.floor(k))
        if not np.any(valid):
            return out
        kv = k[valid].astype(float)
        r, b = self.r, self.beta
        log_pmf = (
            gammaln(r + kv)
            - gammaln(kv + 1.0)
            - gammaln(r)
            - r * np.log1p(b)
            + kv * (np.log(b) - np.log1p(b))
        )
        out[valid] = np.exp(np.maximum(log_pmf, -700.0))
        return out

    def cdf(self, k: ArrayLike) -> np.ndarray:
        from scipy.stats import nbinom as _sp_nbinom

        # scipy's nbinom uses (n=r, p=1/(1+β))
        k = np.asarray(k)
        return np.asarray(
            _sp_nbinom.cdf(np.floor(k), n=self.r, p=1.0 / (1.0 + self.beta)),
            dtype=float,
        )

    def ppf(self, q: ArrayLike) -> np.ndarray:
        from scipy.stats import nbinom as _sp_nbinom

        return _sp_nbinom.ppf(
            np.asarray(q, dtype=float), n=self.r, p=1.0 / (1.0 + self.beta)
        )

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        # Sample via Gamma-Poisson mixture: X | λ ~ Poisson(λ), λ ~ Gamma(r, β)
        lam = rng.gamma(shape=self.r, scale=self.beta, size=size)
        return rng.poisson(lam=lam, size=size)

    def mean(self) -> float:
        return float(self.r * self.beta)
