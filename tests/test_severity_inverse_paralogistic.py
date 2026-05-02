"""Numerical tests for Inverse Paralogistic severity."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import SEVERITY_REGISTRY
from actudist.severity.inverse_paralogistic import InverseParalogistic

from tests._helpers import (
    assert_cdf_monotone,
    assert_lev_matches_quad,
    assert_pdf_integrates_to_one,
    assert_ppf_inverts_cdf,
)


class TestInverseParalogisticBasic:
    def test_registered(self) -> None:
        assert SEVERITY_REGISTRY["InverseParalogistic"] is InverseParalogistic

    def test_pdf_integrates_to_one(self) -> None:
        for t, th in [(2.0, 1.0), (3.0, 2.0), (1.5, 1.0)]:
            assert_pdf_integrates_to_one(InverseParalogistic(tau=t, theta=th))

    def test_cdf_monotone(self) -> None:
        d = InverseParalogistic(tau=2.0, theta=1.0)
        assert_cdf_monotone(d, np.linspace(0.0, 50.0, 60))

    def test_ppf_inverts_cdf(self) -> None:
        d = InverseParalogistic(tau=2.5, theta=1.5)
        assert_ppf_inverts_cdf(d, [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_lev_closed_form_matches_quad(self) -> None:
        for t, th in [(2.0, 1.0), (3.0, 2.0)]:
            d = InverseParalogistic(tau=t, theta=th)
            assert_lev_matches_quad(d, [0.5, 1.0, 5.0, 50.0])


class TestInverseParalogisticMle:
    def test_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = InverseParalogistic(tau=2.5, theta=2.0)
        data = true.rvs(size=4_000, random_state=rng)
        fit = InverseParalogistic()
        params = fit.mle_fit(data)
        assert params["tau"] == pytest.approx(2.5, rel=0.20)
        assert params["theta"] == pytest.approx(2.0, rel=0.25)
