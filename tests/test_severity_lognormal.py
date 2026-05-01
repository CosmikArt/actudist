"""Numerical tests for Lognormal severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.lognormal import Lognormal

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestLognormalBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["Lognormal"] is Lognormal

    def test_pdf_integrates_to_one(self) -> None:
        assert_pdf_integrates_to_one(Lognormal(mu=0.0, sigma=1.0))

    def test_cdf_monotone(self) -> None:
        d = Lognormal(mu=1.0, sigma=0.8)
        assert_cdf_monotone(d, np.linspace(1e-3, 50.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = Lognormal(mu=2.0, sigma=0.5)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_mean_closed_form(self) -> None:
        mu, sigma = 1.0, 0.8
        d = Lognormal(mu=mu, sigma=sigma)
        assert d.mean() == pytest.approx(np.exp(mu + 0.5 * sigma**2))

    def test_lev_closed_form_matches_quad(self) -> None:
        d = Lognormal(mu=1.0, sigma=0.7)
        assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 20.0, 100.0])

    def test_rejects_nonpositive_sigma(self) -> None:
        with pytest.raises(ValueError):
            Lognormal(mu=0.0, sigma=0.0)


class TestLognormalMle:
    def test_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = Lognormal(mu=1.5, sigma=0.6)
        data = true.rvs(size=4_000, random_state=rng)
        fit = Lognormal()
        params = fit.mle_fit(data)
        assert params["mu"] == pytest.approx(1.5, abs=0.05)
        assert params["sigma"] == pytest.approx(0.6, rel=0.10)
