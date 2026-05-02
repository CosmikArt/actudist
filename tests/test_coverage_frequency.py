"""Coverage tests for frequency distributions: init validation, cdf/ppf
paths, rvs(generator), pmf early returns, and the empty-args constructor
branch each frequency exposes for the registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from actudist.frequency.binomial import Binomial
from actudist.frequency.geometric import Geometric
from actudist.frequency.negative_binomial import NegativeBinomial
from actudist.frequency.poisson import Poisson
from actudist.frequency.zinb import ZeroInflatedNegativeBinomial
from actudist.frequency.zip import ZeroInflatedPoisson


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestInitValidation:
    def test_poisson_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            Poisson(lam=0.0)
        with pytest.raises(ValueError):
            Poisson(lam=-1.0)

    def test_binomial_partial_and_invalid(self) -> None:
        with pytest.raises(ValueError):
            Binomial(m=10)
        with pytest.raises(ValueError):
            Binomial(m=0, q=0.5)
        with pytest.raises(ValueError):
            Binomial(m=10, q=0.0)
        with pytest.raises(ValueError):
            Binomial(m=10, q=1.0)

    def test_negative_binomial_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            NegativeBinomial(r=1.0)
        with pytest.raises(ValueError):
            NegativeBinomial(r=-1.0, beta=1.0)

    def test_geometric_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            Geometric(beta=0.0)

    def test_zip_partial_and_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            ZeroInflatedPoisson(pi=0.5)
        with pytest.raises(ValueError):
            ZeroInflatedPoisson(pi=-0.1, lam=1.0)
        with pytest.raises(ValueError):
            ZeroInflatedPoisson(pi=1.0, lam=1.0)
        with pytest.raises(ValueError):
            ZeroInflatedPoisson(pi=0.5, lam=0.0)

    def test_zinb_partial_and_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            ZeroInflatedNegativeBinomial(pi=0.5, r=1.0)
        with pytest.raises(ValueError):
            ZeroInflatedNegativeBinomial(pi=-0.1, r=1.0, beta=1.0)
        with pytest.raises(ValueError):
            ZeroInflatedNegativeBinomial(pi=0.2, r=0.0, beta=1.0)


# ---------------------------------------------------------------------------
# cdf / ppf: happy paths and edge cases
# ---------------------------------------------------------------------------


class TestCdfPpfPaths:
    def test_poisson_cdf_and_ppf(self) -> None:
        d = Poisson(lam=3.0)
        c = d.cdf(np.array([-1.0, 0.0, 5.0, 100.0]))
        # cdf(-1) = 0, cdf(100) ≈ 1
        assert c[0] == 0.0
        assert c[3] == pytest.approx(1.0)
        # ppf goes back to 0..k
        q = d.ppf(np.array([0.1, 0.5, 0.9]))
        assert q.shape == (3,)
        assert np.all(q >= 0)

    def test_binomial_cdf_and_ppf(self) -> None:
        d = Binomial(m=10, q=0.4)
        c = d.cdf(np.arange(0, 12))
        assert np.all(np.diff(c) >= 0)
        q = d.ppf(np.array([0.1, 0.9]))
        assert np.all((q >= 0) & (q <= 10))

    def test_binomial_pmf_zero_outside_support(self) -> None:
        d = Binomial(m=5, q=0.5)
        # k > m or k < 0 returns 0; non-integer also returns 0
        assert d.pmf(np.array([6, -1]))[0] == 0.0
        assert d.pmf(np.array([6, -1]))[1] == 0.0
        assert d.pmf(np.array([2.5])) [0] == 0.0

    def test_negative_binomial_cdf_and_ppf(self) -> None:
        d = NegativeBinomial(r=2.0, beta=1.5)
        c = d.cdf(np.arange(0, 30))
        assert np.all(np.diff(c) >= 0)
        q = d.ppf(np.array([0.1, 0.5, 0.9]))
        assert np.all(q >= 0)

    def test_geometric_cdf_and_ppf(self) -> None:
        d = Geometric(beta=2.0)
        c = d.cdf(np.array([-1.0, 0.0, 5.0, 50.0]))
        assert c[0] == 0.0
        # ppf via scipy.geom-shifted
        q = d.ppf(np.array([0.1, 0.5, 0.9]))
        assert np.all(q >= 0)

    def test_geometric_pmf_zero_outside_support(self) -> None:
        d = Geometric(beta=2.0)
        # negative or non-integer
        assert d.pmf(np.array([-1, 2.5]))[0] == 0.0

    def test_zip_cdf_and_ppf(self) -> None:
        d = ZeroInflatedPoisson(pi=0.3, lam=2.0)
        c = d.cdf(np.array([-1.0, 0.0, 1.0, 5.0]))
        assert c[0] == 0.0
        q = d.ppf(np.array([0.1, 0.5, 0.99]))
        assert np.all(q >= 0)

    def test_zip_pmf_zero_outside_support(self) -> None:
        d = ZeroInflatedPoisson(pi=0.3, lam=2.0)
        assert d.pmf(np.array([-1, 2.5]))[0] == 0.0

    def test_zinb_cdf_and_ppf(self) -> None:
        d = ZeroInflatedNegativeBinomial(pi=0.2, r=2.0, beta=1.5)
        c = d.cdf(np.arange(0, 30))
        assert np.all(np.diff(c) >= 0)
        q = d.ppf(np.array([0.1, 0.5, 0.99]))
        assert np.all(q >= 0)

    def test_zinb_pmf_zero_outside_support(self) -> None:
        d = ZeroInflatedNegativeBinomial(pi=0.2, r=2.0, beta=1.5)
        assert d.pmf(np.array([-1]))[0] == 0.0


# ---------------------------------------------------------------------------
# rvs accepts a Generator instance directly (alternate branch)
# ---------------------------------------------------------------------------


class TestRvsGenerator:
    @pytest.mark.parametrize(
        "dist",
        [
            Poisson(lam=3.0),
            Binomial(m=10, q=0.4),
            NegativeBinomial(r=2.0, beta=1.5),
            Geometric(beta=2.0),
            ZeroInflatedPoisson(pi=0.3, lam=2.0),
            ZeroInflatedNegativeBinomial(pi=0.2, r=2.0, beta=1.5),
        ],
    )
    def test_rvs_with_generator(self, dist) -> None:
        rng = np.random.default_rng(0)
        out = dist.rvs(size=10, random_state=rng)
        assert out.shape == (10,)
        assert np.all(out >= 0)


# ---------------------------------------------------------------------------
# Empty-args construction (the default-None branch each __init__ exposes)
# ---------------------------------------------------------------------------


class TestEmptyConstruction:
    @pytest.mark.parametrize(
        "cls",
        [
            Poisson, Binomial, NegativeBinomial, Geometric,
            ZeroInflatedPoisson, ZeroInflatedNegativeBinomial,
        ],
    )
    def test_empty_args(self, cls) -> None:
        inst = cls()
        assert inst.params is None


# ---------------------------------------------------------------------------
# Mean and pmf edge: empty arrays
# ---------------------------------------------------------------------------


class TestPmfOnEmptyArray:
    @pytest.mark.parametrize(
        "dist",
        [
            Poisson(lam=2.0),
            Binomial(m=5, q=0.3),
            NegativeBinomial(r=1.5, beta=1.0),
            Geometric(beta=1.5),
            ZeroInflatedPoisson(pi=0.3, lam=2.0),
            ZeroInflatedNegativeBinomial(pi=0.2, r=2.0, beta=1.5),
        ],
    )
    def test_pmf_on_empty(self, dist) -> None:
        out = dist.pmf(np.array([], dtype=int))
        assert out.shape == (0,)
