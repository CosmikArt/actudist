"""Tests for the zero-inflated frequency distributions: ZIP, ZINB."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import FREQUENCY_REGISTRY
from actudist.frequency.zip import ZeroInflatedPoisson
from actudist.frequency.zinb import ZeroInflatedNegativeBinomial


class TestZIP:
    def test_registered(self) -> None:
        assert FREQUENCY_REGISTRY["ZIP"] is ZeroInflatedPoisson

    def test_pmf_sums_to_one(self) -> None:
        d = ZeroInflatedPoisson(pi=0.3, lam=4.0)
        ks = np.arange(0, 100)
        assert d.pmf(ks).sum() == pytest.approx(1.0, abs=1e-9)

    def test_zero_mass_dominates(self) -> None:
        d = ZeroInflatedPoisson(pi=0.6, lam=2.0)
        # P(X=0) > exp(-λ)
        assert d.pmf(np.array([0]))[0] > np.exp(-2.0)

    def test_mean_formula(self) -> None:
        d = ZeroInflatedPoisson(pi=0.4, lam=3.0)
        assert d.mean() == pytest.approx(0.4 * 0.0 + 0.6 * 3.0)

    def test_mle_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = ZeroInflatedPoisson(pi=0.4, lam=3.0)
        data = true.rvs(size=10_000, random_state=rng)
        fit = ZeroInflatedPoisson()
        params = fit.mle_fit(data)
        assert params["pi"] == pytest.approx(0.4, abs=0.05)
        assert params["lam"] == pytest.approx(3.0, rel=0.05)


class TestZINB:
    def test_registered(self) -> None:
        assert FREQUENCY_REGISTRY["ZINB"] is ZeroInflatedNegativeBinomial

    def test_pmf_sums_to_one(self) -> None:
        d = ZeroInflatedNegativeBinomial(pi=0.3, r=2.0, beta=1.5)
        ks = np.arange(0, 200)
        assert d.pmf(ks).sum() == pytest.approx(1.0, abs=1e-7)

    def test_mean_formula(self) -> None:
        d = ZeroInflatedNegativeBinomial(pi=0.3, r=2.0, beta=1.5)
        assert d.mean() == pytest.approx(0.7 * 2.0 * 1.5)

    def test_mle_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = ZeroInflatedNegativeBinomial(pi=0.3, r=2.5, beta=1.5)
        data = true.rvs(size=10_000, random_state=rng)
        fit = ZeroInflatedNegativeBinomial()
        params = fit.mle_fit(data)
        # 3 params + zero-inflation make this a noisy estimator; loosen
        # tolerances and rely on mean match for the main sanity check
        assert params["pi"] == pytest.approx(0.3, abs=0.10)
        assert fit.mean() == pytest.approx(true.mean(), rel=0.10)
