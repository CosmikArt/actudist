"""AUDIT-M-4 regression tests.

The chi-squared goodness-of-fit test requires expected counts >= 5 per
bin (Klugman, Loss Models 5e, 16.4.1). The original implementation
divided n / n_bins blindly; with small samples this gave invalid p-values
based on bins where the chi-squared approximation does not hold.

The fix merges adjacent equiprobable bins until each expected count is
at least 5, emits a UserWarning when bins are merged, and reports the
effective bin count in the result dict.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from actudist import GoodnessOfFit
from actudist.severity.exponential import Exponential


def test_chi_squared_merges_when_expected_below_five():
    """With n=20 and n_bins=10 the raw expected is 2 per bin, well below
    Klugman's threshold. The fixed test must merge bins, warn, and
    report the post-merge bin count."""
    rng = np.random.default_rng(0)
    data = Exponential(theta=1.0).rvs(size=20, random_state=rng)
    fit = Exponential()
    fit.mle_fit(data)
    gof = GoodnessOfFit(fit, data)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gof.chi_squared_test(n_bins=10)

    # A warning must mention bin merging.
    assert any(
        "merged" in str(w.message).lower() or "klugman" in str(w.message).lower()
        for w in caught
    ), f"Expected a bin-merge warning; got: {[str(w.message) for w in caught]}"

    # The reported effective bin count must be smaller than the requested 10
    # (we asked for 2-per-bin, fix should drop us to <=4 bins so each has >=5).
    eff_bins = result.get("effective_bins", result["n_bins"])
    assert eff_bins < 10, f"effective_bins={eff_bins}; expected merging"
    # Each post-merge expected count must be >= 5.
    assert 20 / eff_bins >= 5.0


def test_chi_squared_no_merge_when_sample_large():
    """With n=500 and n_bins=10, each bin has expected = 50 which is
    fine; no merge or warning expected."""
    rng = np.random.default_rng(1)
    data = Exponential(theta=1.0).rvs(size=500, random_state=rng)
    fit = Exponential()
    fit.mle_fit(data)
    gof = GoodnessOfFit(fit, data)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gof.chi_squared_test(n_bins=10)

    # No bin-merge warning.
    bin_warnings = [
        w
        for w in caught
        if "merge" in str(w.message).lower() or "klugman" in str(w.message).lower()
    ]
    assert not bin_warnings, f"Unexpected merge warnings: {bin_warnings}"
    eff_bins = result.get("effective_bins", result["n_bins"])
    assert eff_bins == 10


def test_chi_squared_result_has_required_keys():
    rng = np.random.default_rng(2)
    data = Exponential(theta=1.0).rvs(size=200, random_state=rng)
    fit = Exponential()
    fit.mle_fit(data)
    gof = GoodnessOfFit(fit, data)
    result = gof.chi_squared_test(n_bins=10)
    for k in ("statistic", "p_value", "df", "n_bins"):
        assert k in result, f"missing key {k!r}"


@pytest.mark.parametrize("n", [200, 500, 1000])
def test_chi_squared_unchanged_for_well_powered_samples(n):
    """Regression invariance: when there is no need to merge, the
    statistic and p-value must be identical to the formula
    sum((O - E)^2 / E) on equiprobable bins."""
    rng = np.random.default_rng(3)
    data = Exponential(theta=1.0).rvs(size=n, random_state=rng)
    fit = Exponential()
    fit.mle_fit(data)
    gof = GoodnessOfFit(fit, data)
    result = gof.chi_squared_test(n_bins=10)

    # Hand-compute against the formula.
    qs = np.linspace(0.0, 1.0, 11)
    edges = np.asarray(fit.ppf(qs[1:-1]), dtype=float)
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    observed, _ = np.histogram(data, bins=edges)
    expected = n / 10
    stat_expected = float(np.sum((observed - expected) ** 2 / expected))
    assert result["statistic"] == pytest.approx(stat_expected, rel=1e-12)
