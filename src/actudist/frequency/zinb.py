"""Zero-Inflated Negative Binomial. Mixture of a point mass at 0 and a
NegativeBinomial body in :math:`(r, \\beta)` form. See Klugman ch. 6 on
zero-modified distributions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

from actudist.base import FrequencyDistribution
from actudist.fitting import register_frequency


@register_frequency("ZINB")
class ZeroInflatedNegativeBinomial(FrequencyDistribution):
    r"""Zero-Inflated Negative Binomial.

    Parameters: extra-zero mass :math:`\pi\in[0,1)`, dispersion
    :math:`r>0`, and scale :math:`\beta>0`.

    .. math::
        p_0 &= \pi + (1-\pi)\,(1+\beta)^{-r},\\
        p_k &= (1-\pi)\,\binom{r+k-1}{k}\Bigl(\tfrac{1}{1+\beta}\Bigr)^{r}
                \Bigl(\tfrac{\beta}{1+\beta}\Bigr)^{k},\quad k\geq 1.

    Mean :math:`= (1-\pi)\,r\beta`.
    """

    n_params = 3

    def __init__(
        self,
        pi: float | None = None,
        r: float | None = None,
        beta: float | None = None,
    ) -> None:
        if pi is None and r is None and beta is None:
            super().__init__(params=None)
            return
        if pi is None or r is None or beta is None:
            raise ValueError("ZINB needs pi, r, beta")
        pi = float(pi)
        r = float(r)
        beta = float(beta)
        if not 0.0 <= pi < 1.0:
            raise ValueError(f"pi must be in [0, 1); got {pi}")
        if r <= 0 or beta <= 0:
            raise ValueError(f"r, beta must be > 0; got {r}, {beta}")
        super().__init__(params={"pi": pi, "r": r, "beta": beta})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("pi", "logit"), ("r", "log"), ("beta", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        n = arr.size
        n0 = int(np.sum(arr == 0))
        m = max(float(arr.mean()), 1e-3)
        v = max(float(arr.var()), m + 1e-3)
        # MoM for NB body, then back out π from observed zeros
        beta = max(v / m - 1.0, 1e-3)
        r = max(m / beta, 1e-3)
        p0_nb = float((1.0 + beta) ** (-r))
        p0_emp = n0 / max(n, 1)
        pi0 = float(np.clip((p0_emp - p0_nb) / (1.0 - p0_nb + 1e-12), 1e-3, 0.5))
        return {"pi": pi0, "r": r, "beta": beta}

    def _nb_log_pmf(self, kv: np.ndarray) -> np.ndarray:
        r, b = self.r, self.beta
        return (
            gammaln(r + kv)
            - gammaln(kv + 1.0)
            - gammaln(r)
            - r * np.log1p(b)
            + kv * (np.log(b) - np.log1p(b))
        )

    def pmf(self, k: ArrayLike) -> np.ndarray:
        k = np.asarray(k)
        out = np.zeros(k.shape, dtype=float)
        valid = (k >= 0) & (k == np.floor(k))
        if not np.any(valid):
            return out
        kv = k[valid].astype(float)
        body = (1.0 - self.pi) * np.exp(np.maximum(self._nb_log_pmf(kv), -700.0))
        body[kv == 0] += self.pi
        out[valid] = body
        return out

    def cdf(self, k: ArrayLike) -> np.ndarray:
        from scipy.stats import nbinom as _sp_nbinom

        k = np.asarray(k)
        out = np.zeros(k.shape, dtype=float)
        m = k >= 0
        floor_k = np.floor(k[m])
        nb_cdf = _sp_nbinom.cdf(floor_k, n=self.r, p=1.0 / (1.0 + self.beta))
        out[m] = self.pi + (1.0 - self.pi) * nb_cdf
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.empty_like(q, dtype=float)
        for i, qi in enumerate(q):
            k = 0
            while self.cdf(np.array([k]))[0] < qi:
                k += 1
                if k > 10**6:
                    break
            out[i] = float(k)
        return out

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        is_zero = rng.uniform(size=size) < self.pi
        lam = rng.gamma(shape=self.r, scale=self.beta, size=size)
        out = rng.poisson(lam=lam, size=size)
        out[is_zero] = 0
        return out

    def mean(self) -> float:
        return float((1.0 - self.pi) * self.r * self.beta)
