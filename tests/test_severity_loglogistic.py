"""Numerical tests for Log-Logistic severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.loglogistic import LogLogistic

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestLogLogisticBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["LogLogistic"] is LogLogistic

    def test_pdf_integrates_to_one(self) -> None:
        for th, g in [(1.0, 2.0), (2.5, 3.0), (1.0, 0.7)]:
            assert_pdf_integrates_to_one(LogLogistic(theta=th, gamma=g))

    def test_cdf_monotone(self) -> None:
        d = LogLogistic(theta=1.0, gamma=2.0)
        assert_cdf_monotone(d, np.linspace(0.0, 50.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = LogLogistic(theta=2.0, gamma=2.5)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_mean_infinite_when_gamma_le_1(self) -> None:
        assert LogLogistic(theta=1.0, gamma=0.7).mean() == float("inf")

    def test_lev_closed_form_matches_quad(self) -> None:
        for th, g in [(1.0, 2.0), (2.5, 3.0)]:
            d = LogLogistic(theta=th, gamma=g)
            assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 50.0])

    def test_lev_falls_back_to_numeric_when_mean_infinite(self) -> None:
        d = LogLogistic(theta=1.0, gamma=0.7)
        l1 = d.limited_expected_value(1.0)
        l2 = d.limited_expected_value(5.0)
        assert l2 > l1 > 0


class TestLogLogisticMle:
    def test_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = LogLogistic(theta=2.0, gamma=2.5)
        data = true.rvs(size=4_000, random_state=rng)
        fit = LogLogistic()
        params = fit.mle_fit(data)
        assert params["theta"] == pytest.approx(2.0, rel=0.10)
        assert params["gamma"] == pytest.approx(2.5, rel=0.10)
