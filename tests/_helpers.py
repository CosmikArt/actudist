"""Shared numerical assertions for distribution test files.

Underscore-prefixed so pytest does not collect this as a test module.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.integrate import quad

from actudist._numerics import numeric_lev


def assert_pdf_integrates_to_one(dist, upper: float = np.inf, tol: float = 1e-4) -> None:
    val, _ = quad(lambda x: float(dist.pdf(x)), 0.0, upper, limit=400)
    assert abs(val - 1.0) < tol, f"pdf integrates to {val}, not 1"


def assert_cdf_monotone(dist, xs: Iterable[float]) -> None:
    cdfs = np.asarray(dist.cdf(np.asarray(list(xs), dtype=float)))
    diffs = np.diff(cdfs)
    assert np.all(diffs >= -1e-10), f"cdf not monotone, min diff={diffs.min()}"


def assert_ppf_inverts_cdf(dist, qs: Iterable[float], tol: float = 1e-6) -> None:
    for q in qs:
        x = float(dist.ppf(q))
        back = float(dist.cdf(x))
        assert abs(back - q) < tol, f"ppf({q})={x}, cdf back={back}"


def assert_lev_matches_quad(dist, ds: Iterable[float], tol: float = 1e-6) -> None:
    for d in ds:
        closed = float(dist.limited_expected_value(d))
        num = numeric_lev(lambda x: float(dist.survival_function(x)), float(d))
        denom = max(abs(num), 1e-9)
        assert abs(closed - num) / denom < tol or abs(closed - num) < tol, (
            f"LEV({d}): closed={closed}, numeric={num}, rel={(closed - num) / denom}"
        )


def assert_mean_matches_survival_integral(
    dist, upper: float = np.inf, tol: float = 1e-4
) -> None:
    """E[X] = ∫₀^∞ S(x) dx for non-negative X."""
    val, _ = quad(lambda x: float(dist.survival_function(x)), 0.0, upper, limit=400)
    expected = float(dist.mean())
    denom = max(abs(expected), 1.0)
    assert abs(val - expected) / denom < tol, (
        f"E[X]={expected} vs ∫S dx={val}"
    )
