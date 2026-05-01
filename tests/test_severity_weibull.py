"""Numerical tests for Weibull severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.weibull import Weibull

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_mean_matches_survival_integral,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestWeibullBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["Weibull"] is Weibull

    def test_pdf_integrates_to_one(self) -> None:
        for theta, tau in [(1.0, 0.7), (2.0, 1.0), (3.0, 1.8)]:
            assert_pdf_integrates_to_one(Weibull(theta=theta, tau=tau))

    def test_cdf_monotone(self) -> None:
        d = Weibull(theta=2.0, tau=1.5)
        assert_cdf_monotone(d, np.linspace(0.0, 20.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = Weibull(theta=3.0, tau=0.8)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_mean_matches_survival_integral(self) -> None:
        d = Weibull(theta=2.5, tau=1.6)
        assert_mean_matches_survival_integral(d)

    def test_lev_closed_form_matches_quad(self) -> None:
        for theta, tau in [(2.0, 0.7), (2.0, 1.0), (2.0, 2.0)]:
            d = Weibull(theta=theta, tau=tau)
            assert_lev_matches_quad(d, [0.5, 1.0, 4.0, 20.0])


class TestWeibullMle:
    def test_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = Weibull(theta=2.5, tau=1.5)
        data = true.rvs(size=4_000, random_state=rng)
        fit = Weibull()
        params = fit.mle_fit(data)
        assert params["theta"] == pytest.approx(2.5, rel=0.10)
        assert params["tau"] == pytest.approx(1.5, rel=0.10)
