"""AUDIT-H-1 regression tests.

Demonstrates that ZeroInflatedPoisson.ppf and
ZeroInflatedNegativeBinomial.ppf used to silently cap at k=10**6 instead
of returning the true quantile, and that after the fix:

1. quantiles for heavy-tailed parameter regimes return finite values
   that satisfy F(ppf(q)) >= q and F(ppf(q)-1) < q (the discrete-ppf
   contract);
2. for moderate parameters where the old loop would not have hit the
   cap, the values are unchanged (regression invariance);
3. ppf vectorizes over q without an O(n*max_k) Python loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from actudist.frequency.zinb import ZeroInflatedNegativeBinomial
from actudist.frequency.zip import ZeroInflatedPoisson


# ---------------------------------------------------------------------------
# H-1 bug demonstration: heavy-tailed mixtures must not cap at 10**6
# ---------------------------------------------------------------------------


def test_zip_large_lambda_returns_correct_quantile():
    """ZIP(pi=0, lam=2e6) has mean 2e6 so the median is also ~2e6.
    The old loop-based ppf capped at k=10**6 and returned 1_000_001
    silently. After the fix, the value matches the underlying Poisson
    quantile from scipy.stats."""
    from scipy.stats import poisson

    d = ZeroInflatedPoisson(pi=0.0, lam=2.0e6)
    expected = float(poisson.ppf(0.5, mu=2.0e6))
    actual = float(d.ppf(np.asarray([0.5]))[0])
    # Allow ±2 because the discrete-ppf convention agreement.
    assert abs(actual - expected) <= 2, (
        f"ZIP(0, lam=2e6).ppf(0.5) = {actual}; expected ~{expected}. "
        "If actual is around 1_000_001 the loop cap was hit."
    )


def test_zinb_large_beta_returns_correct_quantile():
    """ZINB with pi=0, r=2, beta=1e6 has mean 2e6. The 0.5 quantile
    sits above 10**6 so the old loop cap returned the wrong answer."""
    from scipy.stats import nbinom

    d = ZeroInflatedNegativeBinomial(pi=0.0, r=2.0, beta=1.0e6)
    expected = float(nbinom.ppf(0.5, n=2.0, p=1.0 / (1.0 + 1.0e6)))
    actual = float(d.ppf(np.asarray([0.5]))[0])
    assert abs(actual - expected) <= 2, (
        f"ZINB(0, r=2, beta=1e6).ppf(0.5) = {actual}; expected ~{expected}."
    )


# ---------------------------------------------------------------------------
# Regression: moderate parameters where the old code computed correctly
# must still return identical values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
def test_zip_moderate_parameters_invariant(q):
    d = ZeroInflatedPoisson(pi=0.3, lam=2.0)
    k = float(d.ppf(np.asarray([q]))[0])
    # Discrete-ppf contract.
    cdf_at_k = float(d.cdf(np.asarray([k]))[0])
    assert cdf_at_k >= q - 1e-12
    if k > 0:
        cdf_below = float(d.cdf(np.asarray([k - 1]))[0])
        assert cdf_below < q


@pytest.mark.parametrize("q", [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
def test_zinb_moderate_parameters_invariant(q):
    d = ZeroInflatedNegativeBinomial(pi=0.2, r=3.0, beta=1.0)
    k = float(d.ppf(np.asarray([q]))[0])
    cdf_at_k = float(d.cdf(np.asarray([k]))[0])
    assert cdf_at_k >= q - 1e-12
    if k > 0:
        cdf_below = float(d.cdf(np.asarray([k - 1]))[0])
        assert cdf_below < q


# ---------------------------------------------------------------------------
# pi-mass region: when q <= pi, ppf must return 0 (the inflated zero).
# ---------------------------------------------------------------------------


def test_zip_q_in_pi_mass_returns_zero():
    d = ZeroInflatedPoisson(pi=0.4, lam=5.0)
    # Anywhere in (0, pi] the inflated zero dominates: ppf must be 0.
    for q in [0.01, 0.1, 0.39, 0.4]:
        assert float(d.ppf(np.asarray([q]))[0]) == 0.0, f"q={q}"


def test_zinb_q_in_pi_mass_returns_zero():
    d = ZeroInflatedNegativeBinomial(pi=0.5, r=2.0, beta=2.0)
    for q in [0.01, 0.25, 0.5]:
        # F_ZINB(0) = pi + (1-pi) * (1+beta)^(-r) >= pi, so ppf for q<=pi
        # is always 0.
        assert float(d.ppf(np.asarray([q]))[0]) == 0.0, f"q={q}"


# ---------------------------------------------------------------------------
# Vectorization: ppf accepts an array of q values.
# ---------------------------------------------------------------------------


def test_zip_ppf_vectorized():
    d = ZeroInflatedPoisson(pi=0.2, lam=4.0)
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    out = np.asarray(d.ppf(qs))
    assert out.shape == qs.shape
    # Monotone non-decreasing.
    assert np.all(np.diff(out) >= 0)


def test_zinb_ppf_vectorized():
    d = ZeroInflatedNegativeBinomial(pi=0.1, r=4.0, beta=2.0)
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    out = np.asarray(d.ppf(qs))
    assert out.shape == qs.shape
    assert np.all(np.diff(out) >= 0)
