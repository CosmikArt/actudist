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


def profile_likelihood_ci(
    fitted_dist: Any,
    data: np.ndarray,
    param: str,
    *,
    alpha: float = 0.05,
    fit_kwargs: dict[str, Any] | None = None,
    max_expansions: int = 30,
) -> tuple[float, float]:
    r"""Profile-likelihood confidence interval for a single parameter of a
    fitted distribution.

    The :math:`(1-\alpha)` CI is the set of values :math:`\theta` for which

    .. math::
        2\,\bigl(\ell(\hat\theta_{\text{full}}) - \ell_{\text{p}}(\theta)\bigr)
        \leq \chi^{2}_{1,1-\alpha},

    where :math:`\ell_{\text{p}}(\theta)` is the maximized log-likelihood
    with ``param`` fixed at :math:`\theta`. We bracket each boundary by
    walking geometrically away from the MLE until the profile likelihood
    drops below the threshold, then refine via Brent's root-finder.
    """
    from scipy.optimize import brentq
    from scipy.stats import chi2

    if fit_kwargs is None:
        fit_kwargs = {}

    transforms = type(fitted_dist)._transforms()
    transform_map = dict(transforms)
    if param not in transform_map:
        raise KeyError(f"{param!r} is not a fittable parameter")

    cls = type(fitted_dist)
    is_continuous = hasattr(cls, "pdf")
    full_ll = float(fitted_dist.loglik(data, **fit_kwargs))
    threshold = full_ll - 0.5 * float(chi2.ppf(1.0 - alpha, df=1))
    point = float(getattr(fitted_dist, param))
    other_transforms = [t for t in transforms if t[0] != param]

    def _profile_ll(theta: float) -> float:
        """Maximized log-likelihood with ``param`` fixed at ``theta``."""
        from actudist._numerics import to_unconstrained

        if not other_transforms:
            inst = cls(**{param: float(theta)})
            return float(inst.loglik(data, **fit_kwargs))
        init = {n: float(getattr(fitted_dist, n)) for n, _ in other_transforms}
        u0 = to_unconstrained(init, other_transforms)

        def _neg(u: np.ndarray) -> float:
            try:
                free = from_unconstrained(u, other_transforms)
                params = {**free, param: float(theta)}
                inst = cls(**params)
                if is_continuous:
                    ll = loglik_continuous(inst, data, **fit_kwargs)
                else:
                    ll = loglik_discrete(inst, data)
                return -ll if np.isfinite(ll) else _HUGE
            except Exception:
                return _HUGE

        res = minimize(
            _neg, u0, method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
        )
        return -float(res.fun)

    def _g(theta: float) -> float:
        return _profile_ll(theta) - threshold

    # Initial step: arithmetic for identity-transform params, geometric for others
    is_log = transform_map[param] == "log"
    is_logit = transform_map[param] == "logit"

    def _walk(direction: int) -> float | None:
        # 'direction' ∈ {-1, +1}; return crossing or None if not found
        bracket_inner = point
        # initial step size
        step = max(abs(point) * 0.05, 0.05)
        bracket_outer = point + direction * step
        if is_log and bracket_outer <= 0:
            bracket_outer = point * 0.5 if direction < 0 else point * 1.5
        if is_logit and direction > 0 and bracket_outer >= 1.0:
            bracket_outer = (point + 1.0) / 2.0
        if is_logit and direction < 0 and bracket_outer <= 0.0:
            bracket_outer = point / 2.0

        for _ in range(max_expansions):
            try:
                g_outer = _g(bracket_outer)
            except Exception:
                return None
            if g_outer < 0.0:
                # found a sign change between inner and outer; refine
                try:
                    return float(brentq(_g, bracket_inner, bracket_outer, xtol=1e-6))
                except Exception:
                    return None
            # extend further: geometric for log-transformed params
            bracket_inner = bracket_outer
            if is_log:
                bracket_outer = (
                    bracket_inner * 2.0 if direction > 0 else bracket_inner * 0.5
                )
                if bracket_outer < 1e-12:
                    return None
            elif is_logit:
                if direction > 0:
                    bracket_outer = (bracket_inner + 1.0) / 2.0
                    if bracket_outer >= 1.0 - 1e-9:
                        return None
                else:
                    bracket_outer = bracket_inner / 2.0
                    if bracket_outer <= 1e-9:
                        return None
            else:
                step *= 2.0
                bracket_outer = bracket_inner + direction * step
        return None

    lo_cross = _walk(-1)
    hi_cross = _walk(+1)
    return (
        -np.inf if lo_cross is None else lo_cross,
        np.inf if hi_cross is None else hi_cross,
    )


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
