"""Numerical tests for Transformed Beta severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.transformed_beta import TransformedBeta

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestTransformedBeta:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["TransformedBeta"] is TransformedBeta

    def test_pdf_integrates_to_one(self) -> None:
        for params in [
            (2.0, 1.0, 2.0, 1.0),
            (3.0, 2.0, 1.5, 2.0),
            (2.5, 1.0, 2.0, 0.8),
        ]:
            a, th, g, t = params
            assert_pdf_integrates_to_one(
                TransformedBeta(alpha=a, theta=th, gamma=g, tau=t)
            )

    def test_cdf_monotone(self) -> None:
        d = TransformedBeta(alpha=2.0, theta=1.0, gamma=2.0, tau=1.5)
        assert_cdf_monotone(d, np.linspace(0.0, 50.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = TransformedBeta(alpha=2.5, theta=1.5, gamma=1.5, tau=1.5)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_lev_closed_form_matches_quad(self) -> None:
        for a, th, g, t in [
            (2.5, 1.0, 2.0, 1.0),
            (3.0, 2.0, 1.5, 2.0),
            (2.0, 1.0, 2.0, 0.8),
        ]:
            d = TransformedBeta(alpha=a, theta=th, gamma=g, tau=t)
            assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 50.0])

    def test_reduces_to_burrxii_when_tau_one(self) -> None:
        from actudist.severity.burrxii import BurrXII

        a, th, g = 2.5, 1.5, 2.0
        tb = TransformedBeta(alpha=a, theta=th, gamma=g, tau=1.0)
        b = BurrXII(alpha=a, theta=th, gamma=g)
        for x in [0.5, 1.0, 5.0]:
            assert tb.pdf(x) == pytest.approx(b.pdf(x), rel=1e-10)
            assert tb.cdf(x) == pytest.approx(b.cdf(x), rel=1e-10)
