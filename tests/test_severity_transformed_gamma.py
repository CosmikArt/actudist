"""Numerical tests for Transformed Gamma & Inverse Transformed Gamma."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.transformed_gamma import TransformedGamma
from actudist.severity.inverse_transformed_gamma import InverseTransformedGamma

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestTransformedGamma:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["TransformedGamma"] is TransformedGamma

    def test_pdf_integrates_to_one(self) -> None:
        for a, th, t in [(2.0, 1.0, 2.0), (3.0, 2.0, 1.5), (1.5, 1.0, 0.7)]:
            assert_pdf_integrates_to_one(TransformedGamma(alpha=a, theta=th, tau=t))

    def test_cdf_monotone(self) -> None:
        d = TransformedGamma(alpha=2.0, theta=1.0, tau=2.0)
        assert_cdf_monotone(d, np.linspace(0.0, 30.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = TransformedGamma(alpha=2.5, theta=1.5, tau=1.5)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_lev_closed_form_matches_quad(self) -> None:
        for a, th, t in [(2.0, 1.0, 2.0), (3.0, 2.0, 1.5), (0.7, 2.0, 0.8)]:
            d = TransformedGamma(alpha=a, theta=th, tau=t)
            assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 30.0])

    def test_reduces_to_gamma_when_tau_one(self) -> None:
        from actudist.severity.gamma import Gamma

        a, th = 2.0, 1.5
        tg = TransformedGamma(alpha=a, theta=th, tau=1.0)
        g = Gamma(alpha=a, theta=th)
        assert tg.pdf(2.5) == pytest.approx(g.pdf(2.5))
        assert tg.cdf(2.5) == pytest.approx(g.cdf(2.5))

    def test_mle_recovers(self) -> None:
        rng = np.random.default_rng(0)
        true = TransformedGamma(alpha=2.5, theta=1.5, tau=1.5)
        data = true.rvs(size=6_000, random_state=rng)
        fit = TransformedGamma()
        params = fit.mle_fit(data)
        # 3 params with weak identifiability — wider tolerances expected
        assert params["alpha"] == pytest.approx(2.5, rel=0.40)
        assert params["theta"] == pytest.approx(1.5, rel=0.40)
        assert params["tau"] == pytest.approx(1.5, rel=0.30)


class TestInverseTransformedGamma:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["InverseTransformedGamma"] is InverseTransformedGamma

    def test_pdf_integrates_to_one(self) -> None:
        for a, th, t in [(2.0, 1.0, 2.0), (3.0, 2.0, 1.5), (3.0, 1.0, 0.8)]:
            assert_pdf_integrates_to_one(
                InverseTransformedGamma(alpha=a, theta=th, tau=t)
            )

    def test_cdf_monotone(self) -> None:
        d = InverseTransformedGamma(alpha=2.0, theta=1.0, tau=2.0)
        assert_cdf_monotone(d, np.linspace(1e-3, 30.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = InverseTransformedGamma(alpha=2.5, theta=1.5, tau=1.5)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_lev_closed_form_matches_quad_when_mean_finite(self) -> None:
        for a, th, t in [(2.0, 1.0, 2.0), (3.0, 2.0, 1.5)]:
            d = InverseTransformedGamma(alpha=a, theta=th, tau=t)
            assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 30.0])

    def test_lev_falls_back_when_mean_infinite(self) -> None:
        # ατ ≤ 1 ⇒ mean infinite
        d = InverseTransformedGamma(alpha=1.0, theta=1.0, tau=0.7)
        assert d.mean() == float("inf")
        assert d.limited_expected_value(5.0) > 0
