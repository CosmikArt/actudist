"""Numerical tests for Exponential severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.exponential import Exponential

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_mean_matches_survival_integral,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestExponentialBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["Exponential"] is Exponential

    def test_pdf_integrates_to_one(self) -> None:
        assert_pdf_integrates_to_one(Exponential(theta=2.5))

    def test_cdf_monotone(self) -> None:
        d = Exponential(theta=1.0)
        assert_cdf_monotone(d, np.linspace(0.0, 10.0, 50))

    def test_ppf_inverts_cdf(self) -> None:
        d = Exponential(theta=3.7)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_mean_closed_form(self) -> None:
        d = Exponential(theta=4.2)
        assert d.mean() == pytest.approx(4.2)
        assert_mean_matches_survival_integral(d)

    def test_lev_closed_form_matches_quad(self) -> None:
        d = Exponential(theta=2.0)
        assert_lev_matches_quad(d, [0.5, 1.0, 2.0, 5.0, 20.0])

    def test_rejects_nonpositive_theta(self) -> None:
        with pytest.raises(ValueError):
            Exponential(theta=0.0)
        with pytest.raises(ValueError):
            Exponential(theta=-1.0)


class TestExponentialMle:
    def test_recovers_theta(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.exponential(scale=2.5, size=5_000)
        fit = Exponential()
        params = fit.mle_fit(data)
        assert params["theta"] == pytest.approx(2.5, rel=0.05)

    def test_handles_right_censoring(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.exponential(scale=3.0, size=4_000)
        cap = 4.0
        censored = data >= cap
        clipped = np.minimum(data, cap)
        fit = Exponential()
        params = fit.mle_fit(clipped, censored=censored)
        assert params["theta"] == pytest.approx(3.0, rel=0.1)
