"""Numerical helpers shared by distributions: stable log/exp, root finding,
numerical LEV via quadrature, and parameter-space transforms."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


_LOG_FLOOR = -700.0  # below this, np.exp underflows; equivalent to ~0 prob


def safe_log(x: np.ndarray | float) -> np.ndarray | float:
    """``log(x)`` with a finite floor instead of ``-inf`` for ``x <= 0``."""
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, _LOG_FLOOR, dtype=float)
    pos = arr > 0
    out[pos] = np.log(arr[pos])
    return out if out.ndim else float(out)


def numeric_ppf(
    cdf: Callable[[float], float],
    q: float,
    lower: float = 0.0,
    upper: float = 1e12,
    xtol: float = 1e-10,
) -> float:
    """Invert ``cdf`` numerically via Brent's method.

    Used as a fallback for distributions whose quantile function lacks a
    closed form. Caller guarantees ``cdf(lower) <= q <= cdf(upper)``.
    """
    if not 0.0 < q < 1.0:
        if q == 0.0:
            return lower
        if q == 1.0:
            return upper
        raise ValueError(f"q must be in (0, 1); got {q!r}")
    return brentq(lambda x: cdf(x) - q, lower, upper, xtol=xtol)


def numeric_lev(
    survival_function: Callable[[float], float],
    d: float,
    *,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> float:
    r"""Numerical limited expected value: :math:`\int_0^d S(x)\,dx`.

    Used either as a fallback when no closed form is known, or as a test
    oracle to validate closed-form LEV implementations.
    """
    if d <= 0:
        return 0.0
    val, _ = quad(survival_function, 0.0, d, epsabs=epsabs, epsrel=epsrel)
    return float(val)


# ---------------------------------------------------------------------------
# Parameter-space transforms for unconstrained MLE
# ---------------------------------------------------------------------------


def to_unconstrained(
    params: dict[str, float],
    transforms: list[tuple[str, str]],
) -> np.ndarray:
    """Map a parameter dict to an unconstrained vector for optimizer input."""
    out = []
    for name, t in transforms:
        v = params[name]
        if t == "log":
            if v <= 0:
                raise ValueError(f"{name}={v!r} not positive; cannot apply log")
            out.append(np.log(v))
        elif t == "logit":
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name}={v!r} not in (0, 1); cannot apply logit")
            out.append(np.log(v / (1.0 - v)))
        elif t == "identity":
            out.append(v)
        else:
            raise ValueError(f"unknown transform {t!r}")
    return np.asarray(out, dtype=float)


def from_unconstrained(
    u: np.ndarray,
    transforms: list[tuple[str, str]],
) -> dict[str, float]:
    """Inverse of :func:`to_unconstrained`."""
    out: dict[str, float] = {}
    for (name, t), val in zip(transforms, u, strict=True):
        if t == "log":
            out[name] = float(np.exp(val))
        elif t == "logit":
            out[name] = float(1.0 / (1.0 + np.exp(-val)))
        elif t == "identity":
            out[name] = float(val)
        else:
            raise ValueError(f"unknown transform {t!r}")
    return out
