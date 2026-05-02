"""Zero-Inflated Poisson. Excess-zero mass :math:`\\pi` mixed with a
Poisson body of rate :math:`\\lambda`."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

from actudist.base import FrequencyDistribution
from actudist.fitting import register_frequency


@register_frequency("ZIP")
class ZeroInflatedPoisson(FrequencyDistribution):
    r"""Zero-Inflated Poisson with extra-zero mass :math:`\pi\in[0,1)` and
    Poisson rate :math:`\lambda>0`.

    .. math::
        p_0 &= \pi + (1-\pi)\,e^{-\lambda},\\
        p_k &= (1-\pi)\,\frac{e^{-\lambda}\lambda^{k}}{k!},\quad k\geq 1.

    Mean :math:`= (1-\pi)\lambda`.
    """

    n_params = 2
    pi: float
    lam: float

    def __init__(self, pi: float | None = None, lam: float | None = None) -> None:
        if pi is None and lam is None:
            super().__init__(params=None)
            return
        if pi is None or lam is None:
            raise ValueError("ZIP needs both pi and lam")
        pi = float(pi)
        lam = float(lam)
        if not 0.0 <= pi < 1.0:
            raise ValueError(f"pi must be in [0, 1); got {pi}")
        if lam <= 0:
            raise ValueError(f"lam must be > 0; got {lam}")
        super().__init__(params={"pi": pi, "lam": lam})

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        return [("pi", "logit"), ("lam", "log")]

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        arr = np.asarray(data, dtype=float)
        n = arr.size
        n0 = int(np.sum(arr == 0))
        m = max(float(arr.mean()), 1e-3)
        # crude: if zero-fraction exceeds Poisson(λ=mean) zero-prob, set π accordingly
        p0_emp = n0 / max(n, 1)
        p0_pois = float(np.exp(-m))
        pi0 = float(np.clip((p0_emp - p0_pois) / (1.0 - p0_pois + 1e-12), 1e-3, 0.5))
        # adjust λ given π: mean = (1-π)λ
        lam0 = m / max(1.0 - pi0, 1e-6)
        return {"pi": pi0, "lam": lam0}

    def pmf(self, k: ArrayLike) -> np.ndarray:
        k = np.asarray(k)
        out = np.zeros(k.shape, dtype=float)
        valid = (k >= 0) & (k == np.floor(k))
        if not np.any(valid):
            return out
        kv = k[valid].astype(float)
        pi, lam = self.pi, self.lam
        log_pois = -lam + kv * np.log(lam) - gammaln(kv + 1.0)
        body = (1.0 - pi) * np.exp(np.maximum(log_pois, -700.0))
        body[kv == 0] += pi
        out[valid] = body
        return out

    def cdf(self, k: ArrayLike) -> np.ndarray:
        from scipy.special import pdtr

        k = np.asarray(k)
        out = np.zeros(k.shape, dtype=float)
        m = k >= 0
        floor_k = np.floor(k[m]).astype(float)
        out[m] = self.pi + (1.0 - self.pi) * pdtr(floor_k, self.lam)
        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        # discrete inverse via cdf
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
        out = rng.poisson(lam=self.lam, size=size)
        out[is_zero] = 0
        return out

    def mean(self) -> float:
        return float((1.0 - self.pi) * self.lam)
