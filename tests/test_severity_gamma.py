"""Numerical tests for Gamma severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.gamma import Gamma

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_mean_matches_survival_integral,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestGammaBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["Gamma"] is Gamma

    def test_pdf_integrates_to_one(self) -> None:
        for a, th in [(0.7, 1.0), (1.0, 2.0), (3.5, 0.8)]:
            assert_pdf_integrates_to_one(Gamma(alpha=a, theta=th))

    def test_cdf_monotone(self) -> None:
        d = Gamma(alpha=2.0, theta=1.0)
        assert_cdf_monotone(d, np.linspace(0.0, 30.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = Gamma(alpha=2.5, theta=1.5)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_mean_matches_survival_integral(self) -> None:
        d = Gamma(alpha=2.5, theta=1.5)
        assert_mean_matches_survival_integral(d)

    def test_lev_closed_form_matches_quad(self) -> None:
        for a, th in [(0.7, 2.0), (2.0, 1.5), (5.0, 0.8)]:
            d = Gamma(alpha=a, theta=th)
            assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 30.0])


class TestGammaMle:
    def test_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = Gamma(alpha=2.5, theta=1.2)
        data = true.rvs(size=4_000, random_state=rng)
        fit = Gamma()
        params = fit.mle_fit(data)
        assert params["alpha"] == pytest.approx(2.5, rel=0.10)
        assert params["theta"] == pytest.approx(1.2, rel=0.10)
