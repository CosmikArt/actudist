"""Numerical tests for Inverse Gaussian severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.inverse_gaussian import InverseGaussian

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestInverseGaussianBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["InverseGaussian"] is InverseGaussian

    def test_pdf_integrates_to_one(self) -> None:
        for mu, b in [(1.0, 1.0), (2.0, 5.0), (0.5, 2.0)]:
            assert_pdf_integrates_to_one(InverseGaussian(mu=mu, beta=b))

    def test_cdf_monotone(self) -> None:
        d = InverseGaussian(mu=1.0, beta=2.0)
        assert_cdf_monotone(d, np.linspace(1e-3, 20.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = InverseGaussian(mu=2.0, beta=3.0)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95], tol=1e-5)

    def test_mean_closed_form(self) -> None:
        d = InverseGaussian(mu=1.7, beta=2.3)
        assert d.mean() == pytest.approx(1.7)

    def test_lev_closed_form_matches_quad(self) -> None:
        for mu, b in [(1.0, 0.5), (1.0, 5.0), (3.0, 2.0)]:
            d = InverseGaussian(mu=mu, beta=b)
            assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 50.0], tol=1e-8)


class TestInverseGaussianMle:
    def test_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = InverseGaussian(mu=2.0, beta=3.0)
        data = true.rvs(size=4_000, random_state=rng)
        fit = InverseGaussian()
        params = fit.mle_fit(data)
        assert params["mu"] == pytest.approx(2.0, rel=0.05)
        assert params["beta"] == pytest.approx(3.0, rel=0.10)
