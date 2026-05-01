"""Numerical tests for Burr Type XII severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.burrxii import BurrXII

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestBurrXIIBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["BurrXII"] is BurrXII

    def test_pdf_integrates_to_one(self) -> None:
        for a, th, g in [(2.0, 1.0, 2.0), (3.0, 2.0, 1.5), (1.5, 1.0, 3.0)]:
            assert_pdf_integrates_to_one(BurrXII(alpha=a, theta=th, gamma=g))

    def test_cdf_monotone(self) -> None:
        d = BurrXII(alpha=2.0, theta=1.0, gamma=2.0)
        assert_cdf_monotone(d, np.linspace(0.0, 50.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = BurrXII(alpha=2.5, theta=1.5, gamma=2.0)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_mean_infinite_when_alpha_gamma_le_1(self) -> None:
        # α γ ≤ 1 ⇒ mean is infinite
        assert BurrXII(alpha=0.5, theta=1.0, gamma=1.5).mean() == float("inf")

    def test_lev_closed_form_matches_quad(self) -> None:
        for a, th, g in [(2.0, 1.0, 2.0), (3.0, 2.0, 1.5), (1.5, 1.0, 3.0)]:
            d = BurrXII(alpha=a, theta=th, gamma=g)
            assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 50.0])

    def test_lev_falls_back_to_numeric_when_mean_infinite(self) -> None:
        d = BurrXII(alpha=0.5, theta=1.0, gamma=1.5)
        # the closed-form path is undefined; fallback should still produce
        # a finite, monotone non-decreasing LEV
        l1 = d.limited_expected_value(1.0)
        l2 = d.limited_expected_value(5.0)
        assert l2 > l1 > 0


class TestBurrXIIMle:
    def test_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = BurrXII(alpha=2.5, theta=2.0, gamma=2.0)
        data = true.rvs(size=8_000, random_state=rng)
        fit = BurrXII()
        params = fit.mle_fit(data)
        # 3 params + heavy tail → wider tolerances are reasonable
        assert params["alpha"] == pytest.approx(2.5, rel=0.30)
        assert params["theta"] == pytest.approx(2.0, rel=0.30)
        assert params["gamma"] == pytest.approx(2.0, rel=0.20)
