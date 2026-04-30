"""Maximum-likelihood estimation infrastructure.

Provides a single log-likelihood implementation that supports right-censoring
and left/right truncation for continuous severity distributions, a simpler
discrete variant for frequency distributions, and the L-BFGS-B drivers that
optimize over an unconstrained reparameterization (see
:mod:`actudist._numerics`).

Concrete distributions never call ``scipy.optimize`` directly; they declare
``_transforms()`` and ``_initial_guess()`` and the helpers in this module
take care of the rest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from actudist._numerics import from_unconstrained, safe_log, to_unconstrained


_HUGE = 1e15  # penalty for parameter sets that raise during eval


# ---------------------------------------------------------------------------
# Continuous: log-likelihood with censoring / truncation
# ---------------------------------------------------------------------------


def loglik_continuous(
    dist: Any,
    data: np.ndarray,
    *,
    censored: ArrayLike | None = None,
    trunc_lower: float | None = None,
    trunc_upper: float | None = None,
) -> float:
    r"""Log-likelihood for a fitted ``SeverityDistribution`` on data.

    Notation: each ``data[i]`` is either an exact observation (default) or a
    right-censoring point (when ``censored[i]`` is True). All observations
    are conditional on the truncation interval :math:`(\ell, u]` when
    ``trunc_lower`` / ``trunc_upper`` are supplied.

    .. math::
        \ell\!\ell = \sum_{i \text{ observed}} \log f(x_i)
                   + \sum_{j \text{ censored}} \log S(c_j)
                   - n\,\log\!\bigl(S(\ell) - S(u)\bigr)
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0:
        return 0.0

    if censored is None:
        observed_mask = np.ones(n, dtype=bool)
    else:
        observed_mask = ~np.asarray(censored, dtype=bool)
        if observed_mask.shape != data.shape:
            raise ValueError("censored mask must match data shape")

    obs = data[observed_mask]
    cens = data[~observed_mask]

    ll = 0.0
    if obs.size:
        ll += float(np.sum(safe_log(np.asarray(dist.pdf(obs)))))
    if cens.size:
        ll += float(np.sum(safe_log(np.asarray(dist.survival_function(cens)))))

    if trunc_lower is not None or trunc_upper is not None:
        s_lower = 1.0 if trunc_lower is None else float(dist.survival_function(trunc_lower))
        s_upper = 0.0 if trunc_upper is None else float(dist.survival_function(trunc_upper))
        prob = s_lower - s_upper
        if prob <= 0:
            return -np.inf
        ll -= n * float(np.log(prob))

    return ll


# ---------------------------------------------------------------------------
# Continuous: L-BFGS-B driver in log/logit-reparameterized space
# ---------------------------------------------------------------------------


def fit_continuous_mle(
    dist_class: type,
    data: np.ndarray,
    *,
    censored: ArrayLike | None = None,
    trunc_lower: float | None = None,
    trunc_upper: float | None = None,
    initial_params: dict[str, float] | None = None,
    bounds_unconstrained: tuple[float, float] = (-20.0, 20.0),
) -> dict[str, float]:
    """Fit ``dist_class`` to ``data`` by MLE in unconstrained space.

    The optimizer is L-BFGS-B with explicit unconstrained bounds; on failure
    we fall back to Nelder-Mead, which is derivative-free and tolerates
    nastier likelihood surfaces.
    """
    transforms = dist_class._transforms()
    if initial_params is None:
        initial_params = dist_class._initial_guess(data)
    u0 = to_unconstrained(initial_params, transforms)

    def neg_ll(u: np.ndarray) -> float:
        try:
            params = from_unconstrained(u, transforms)
            inst = dist_class(**params)
            ll = loglik_continuous(
                inst,
                data,
                censored=censored,
                trunc_lower=trunc_lower,
                trunc_upper=trunc_upper,
            )
            if not np.isfinite(ll):
                return _HUGE
            return -ll
        except (ValueError, FloatingPointError, OverflowError):
            return _HUGE

    bounds = [bounds_unconstrained] * len(u0)
    res = minimize(neg_ll, u0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        res = minimize(neg_ll, u0, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})

    return from_unconstrained(np.asarray(res.x), transforms)


# ---------------------------------------------------------------------------
# Discrete (frequency): log-likelihood and MLE driver
# ---------------------------------------------------------------------------


def loglik_discrete(dist: Any, data: np.ndarray) -> float:
    """Log-likelihood for a frequency distribution on integer counts."""
    data = np.asarray(data)
    if data.size == 0:
        return 0.0
    return float(np.sum(safe_log(np.asarray(dist.pmf(data)))))


def fit_discrete_mle(
    dist_class: type,
    data: np.ndarray,
    *,
    initial_params: dict[str, float] | None = None,
    bounds_unconstrained: tuple[float, float] = (-20.0, 20.0),
) -> dict[str, float]:
    """L-BFGS-B MLE for a frequency distribution; same pattern as the
    continuous driver but without censoring/truncation hooks (frequency
    censoring is reserved for v0.2)."""
    transforms = dist_class._transforms()
    if initial_params is None:
        initial_params = dist_class._initial_guess(data)
    u0 = to_unconstrained(initial_params, transforms)

    def neg_ll(u: np.ndarray) -> float:
        try:
            params = from_unconstrained(u, transforms)
            inst = dist_class(**params)
            ll = loglik_discrete(inst, data)
            if not np.isfinite(ll):
                return _HUGE
            return -ll
        except (ValueError, FloatingPointError, OverflowError):
            return _HUGE

    bounds = [bounds_unconstrained] * len(u0)
    res = minimize(neg_ll, u0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        res = minimize(neg_ll, u0, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})

    return from_unconstrained(np.asarray(res.x), transforms)
