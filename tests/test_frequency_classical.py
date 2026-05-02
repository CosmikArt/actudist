"""Tests for the classical frequency distributions: Poisson, Binomial,
NegativeBinomial, Geometric."""

from __future__ import annotations

import numpy as np
import pytest

from actudist import FREQUENCY_REGISTRY
from actudist.frequency.binomial import Binomial
from actudist.frequency.geometric import Geometric
from actudist.frequency.negative_binomial import NegativeBinomial
from actudist.frequency.poisson import Poisson


class TestPoisson:
    def test_registered(self) -> None:
        assert FREQUENCY_REGISTRY["Poisson"] is Poisson

    def test_pmf_sums_to_one(self) -> None:
        d = Poisson(lam=3.5)
        ks = np.arange(0, 50)
        assert d.pmf(ks).sum() == pytest.approx(1.0, abs=1e-9)

    def test_mean_matches_lambda(self) -> None:
        d = Poisson(lam=2.7)
        assert d.mean() == pytest.approx(2.7)

    def test_mle_recovers_lambda(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.poisson(lam=4.2, size=5_000)
        params = Poisson().mle_fit(data)
        assert params["lam"] == pytest.approx(4.2, rel=0.05)


class TestBinomial:
    def test_registered(self) -> None:
        assert FREQUENCY_REGISTRY["Binomial"] is Binomial

    def test_pmf_sums_to_one(self) -> None:
        d = Binomial(m=10, q=0.3)
        assert d.pmf(np.arange(0, 11)).sum() == pytest.approx(1.0, abs=1e-12)

    def test_mle_recovers_mean(self) -> None:
        # m is not identifiable from data alone; we fix m = max(observed)
        # and verify the resulting model has the correct mean.
        rng = np.random.default_rng(0)
        data = rng.binomial(n=20, p=0.4, size=2_000)
        fit = Binomial()
        fit.mle_fit(data)
        assert fit.mean() == pytest.approx(data.mean(), rel=1e-9)


class TestNegativeBinomial:
    def test_registered(self) -> None:
        assert FREQUENCY_REGISTRY["NegativeBinomial"] is NegativeBinomial

    def test_pmf_sums_to_one(self) -> None:
        d = NegativeBinomial(r=2.5, beta=1.5)
        ks = np.arange(0, 200)
        assert d.pmf(ks).sum() == pytest.approx(1.0, abs=1e-9)

    def test_mean_matches_r_beta(self) -> None:
        d = NegativeBinomial(r=2.0, beta=1.5)
        assert d.mean() == pytest.approx(3.0)

    def test_mle_recovers_params(self) -> None:
        rng = np.random.default_rng(0)
        true = NegativeBinomial(r=3.0, beta=2.0)
        data = true.rvs(size=4_000, random_state=rng)
        fit = NegativeBinomial()
        params = fit.mle_fit(data)
        assert params["r"] == pytest.approx(3.0, rel=0.20)
        assert params["beta"] == pytest.approx(2.0, rel=0.20)


class TestGeometric:
    def test_registered(self) -> None:
        assert FREQUENCY_REGISTRY["Geometric"] is Geometric

    def test_pmf_sums_to_one(self) -> None:
        d = Geometric(beta=2.0)
        ks = np.arange(0, 200)
        assert d.pmf(ks).sum() == pytest.approx(1.0, abs=1e-9)

    def test_matches_negative_binomial_with_r_one(self) -> None:
        beta = 2.0
        g = Geometric(beta=beta)
        nb = NegativeBinomial(r=1.0, beta=beta)
        for k in [0, 1, 2, 5, 10]:
            assert g.pmf(np.array([k]))[0] == pytest.approx(nb.pmf(np.array([k]))[0])

    def test_mle_recovers_beta(self) -> None:
        rng = np.random.default_rng(0)
        data = Geometric(beta=2.5).rvs(size=4_000, random_state=rng)
        params = Geometric().mle_fit(data)
        assert params["beta"] == pytest.approx(2.5, rel=0.10)
