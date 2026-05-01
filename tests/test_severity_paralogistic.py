"""Numerical tests for Paralogistic severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.paralogistic import Paralogistic

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestParalogisticBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["Paralogistic"] is Paralogistic

    def test_pdf_integrates_to_one(self) -> None:
        for a, th in [(2.0, 1.0), (3.0, 2.5), (1.5, 1.0)]:
            assert_pdf_integrates_to_one(Paralogistic(alpha=a, theta=th))

    def test_cdf_monotone(self) -> None:
        d = Paralogistic(alpha=2.0, theta=1.0)
        assert_cdf_monotone(d, np.linspace(0.0, 50.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = Paralogistic(alpha=2.5, theta=1.5)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_lev_closed_form_matches_quad(self) -> None:
        for a, th in [(2.0, 1.0), (3.0, 2.5)]:
            d = Paralogistic(alpha=a, theta=th)
            assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 50.0])

    def test_mean_infinite_when_alpha_le_1(self) -> None:
        assert Paralogistic(alpha=0.7, theta=1.0).mean() == float("inf")


class TestParalogisticMle:
    def test_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = Paralogistic(alpha=2.5, theta=2.0)
        data = true.rvs(size=4_000, random_state=rng)
        fit = Paralogistic()
        params = fit.mle_fit(data)
        assert params["alpha"] == pytest.approx(2.5, rel=0.15)
        assert params["theta"] == pytest.approx(2.0, rel=0.20)
