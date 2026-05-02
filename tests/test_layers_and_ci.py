"""Tests for layer statistics and profile-likelihood confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from actudist.layers import (
    excess_pure_premium,
    increased_limits_factor,
    layer_expected_value,
)
from actudist.severity.exponential import Exponential
from actudist.severity.lognormal import Lognormal
from actudist.severity.pareto import Pareto


class TestLayerHelpers:
    def test_excess_pure_premium_matches_method(self) -> None:
        d = Exponential(theta=2.0)
        for d_val in [0.5, 1.0, 5.0]:
            assert excess_pure_premium(d, d_val) == pytest.approx(
                d.excess_pure_premium(d_val)
            )

    def test_excess_pure_premium_known_exponential(self) -> None:
        # E[X] - LEV(d) = θ - θ(1 - exp(-d/θ)) = θ exp(-d/θ)
        d = Exponential(theta=3.0)
        for x in [1.0, 3.0, 10.0]:
            assert excess_pure_premium(d, x) == pytest.approx(
                3.0 * np.exp(-x / 3.0), rel=1e-9
            )

    def test_increased_limits_factor_monotone(self) -> None:
        d = Lognormal(mu=1.0, sigma=0.6)
        ilf_1 = increased_limits_factor(d, 1.0, base_d=0.5)
        ilf_2 = increased_limits_factor(d, 5.0, base_d=0.5)
        ilf_3 = increased_limits_factor(d, 50.0, base_d=0.5)
        # ILF should be non-decreasing and ≥ 1 for limits above the base
        assert ilf_1 >= 1.0
        assert ilf_2 >= ilf_1
        assert ilf_3 >= ilf_2

    def test_layer_expected_value_equals_diff_of_lev(self) -> None:
        d = Pareto(alpha=3.0, theta=2.0)
        layer = layer_expected_value(d, 1.0, 5.0)
        assert layer == pytest.approx(
            d.limited_expected_value(5.0) - d.limited_expected_value(1.0)
        )

    def test_layer_rejects_inverted_bounds(self) -> None:
        d = Pareto(alpha=3.0, theta=2.0)
        with pytest.raises(ValueError):
            layer_expected_value(d, 5.0, 1.0)


class TestProfileLikelihoodCI:
    def test_ci_contains_truth_for_exponential(self) -> None:
        rng = np.random.default_rng(0)
        true_theta = 2.0
        data = Exponential(theta=true_theta).rvs(size=2_000, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        lo, hi = fit.profile_likelihood_ci(data, "theta", alpha=0.05)
        assert lo < true_theta < hi
        # CI should not be absurdly wide for n=2000
        assert hi - lo < 0.5

    def test_ci_for_two_param_distribution(self) -> None:
        rng = np.random.default_rng(0)
        true = Lognormal(mu=1.5, sigma=0.6)
        data = true.rvs(size=2_000, random_state=rng)
        fit = Lognormal()
        fit.mle_fit(data)
        lo, hi = fit.profile_likelihood_ci(data, "mu", alpha=0.05)
        assert lo < 1.5 < hi
        lo_s, hi_s = fit.profile_likelihood_ci(data, "sigma", alpha=0.05)
        assert lo_s < 0.6 < hi_s
