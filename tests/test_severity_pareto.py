"""Numerical tests for Pareto (Type II) severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.pareto import Pareto

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestParetoBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["Pareto"] is Pareto

    def test_pdf_integrates_to_one(self) -> None:
        assert_pdf_integrates_to_one(Pareto(alpha=3.0, theta=2.0))

    def test_cdf_monotone(self) -> None:
        d = Pareto(alpha=2.5, theta=1.0)
        assert_cdf_monotone(d, np.linspace(0.0, 100.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = Pareto(alpha=3.5, theta=2.0)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_mean_closed_form(self) -> None:
        d = Pareto(alpha=3.0, theta=2.0)
        assert d.mean() == pytest.approx(1.0)

    def test_mean_infinite_when_alpha_le_1(self) -> None:
        assert Pareto(alpha=0.8, theta=2.0).mean() == float("inf")

    def test_lev_closed_form_matches_quad(self) -> None:
        d = Pareto(alpha=3.0, theta=2.0)
        assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 50.0])

    def test_lev_alpha_one_special_case(self) -> None:
        # closed form switches to θ ln(1 + d/θ)
        d = Pareto(alpha=1.0, theta=4.0)
        # E[X∧d] = 4 ln(1 + d/4)
        assert d.limited_expected_value(4.0) == pytest.approx(4.0 * np.log(2.0))
        assert_lev_matches_quad(d, [1.0, 4.0, 20.0])


class TestParetoMle:
    def test_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        # Pareto Type II via inverse-cdf sampling
        true = Pareto(alpha=3.0, theta=2.0)
        data = true.rvs(size=8_000, random_state=rng)
        fit = Pareto()
        params = fit.mle_fit(data)
        assert params["alpha"] == pytest.approx(3.0, rel=0.10)
        assert params["theta"] == pytest.approx(2.0, rel=0.15)
