"""Coverage tests for severity distributions: init validation, pdf-on-empty
input early returns, LEV ``d <= 0`` short circuits, and ``mean is infinite``
branches that the happy-path tests don't exercise.
"""

from __future__ import annotations

import numpy as np
import pytest

from actudist.severity.burrxii import BurrXII
from actudist.severity.exponential import Exponential
from actudist.severity.gamma import Gamma
from actudist.severity.inverse_gaussian import InverseGaussian
from actudist.severity.inverse_paralogistic import InverseParalogistic
from actudist.severity.inverse_transformed_gamma import InverseTransformedGamma
from actudist.severity.loglogistic import LogLogistic
from actudist.severity.lognormal import Lognormal
from actudist.severity.paralogistic import Paralogistic
from actudist.severity.pareto import Pareto
from actudist.severity.transformed_beta import TransformedBeta
from actudist.severity.transformed_gamma import TransformedGamma
from actudist.severity.weibull import Weibull


# ---------------------------------------------------------------------------
# Init validation: each dist rejects partial / non-positive parameters.
# ---------------------------------------------------------------------------


class TestInitValidation:
    def test_pareto_partial(self) -> None:
        with pytest.raises(ValueError):
            Pareto(alpha=1.0)
        with pytest.raises(ValueError):
            Pareto(alpha=0.0, theta=1.0)
        with pytest.raises(ValueError):
            Pareto(alpha=1.0, theta=-1.0)

    def test_lognormal_partial_and_bad_sigma(self) -> None:
        with pytest.raises(ValueError):
            Lognormal(mu=0.0)

    def test_weibull_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            Weibull(theta=1.0)
        with pytest.raises(ValueError):
            Weibull(theta=-1.0, tau=1.0)

    def test_gamma_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            Gamma(alpha=1.0)
        with pytest.raises(ValueError):
            Gamma(alpha=0.0, theta=1.0)

    def test_burrxii_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            BurrXII(alpha=1.0, theta=1.0)
        with pytest.raises(ValueError):
            BurrXII(alpha=0.0, theta=1.0, gamma=1.0)

    def test_loglogistic_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            LogLogistic(theta=1.0)
        with pytest.raises(ValueError):
            LogLogistic(theta=-1.0, gamma=1.0)

    def test_paralogistic_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            Paralogistic(alpha=1.0)
        with pytest.raises(ValueError):
            Paralogistic(alpha=-1.0, theta=1.0)

    def test_inverse_paralogistic_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            InverseParalogistic(tau=1.0)
        with pytest.raises(ValueError):
            InverseParalogistic(tau=-1.0, theta=1.0)

    def test_inverse_gaussian_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            InverseGaussian(mu=1.0)
        with pytest.raises(ValueError):
            InverseGaussian(mu=-1.0, beta=1.0)

    def test_transformed_gamma_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            TransformedGamma(alpha=1.0, theta=1.0)
        with pytest.raises(ValueError):
            TransformedGamma(alpha=-1.0, theta=1.0, tau=1.0)

    def test_inverse_transformed_gamma_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            InverseTransformedGamma(alpha=1.0, theta=1.0)
        with pytest.raises(ValueError):
            InverseTransformedGamma(alpha=-1.0, theta=1.0, tau=1.0)

    def test_transformed_beta_partial_and_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            TransformedBeta(alpha=1.0, theta=1.0, gamma=1.0)
        with pytest.raises(ValueError):
            TransformedBeta(alpha=-1.0, theta=1.0, gamma=1.0, tau=1.0)


# ---------------------------------------------------------------------------
# pdf: when input is all <= 0 (or all <= 0 for log-supported), we hit the
# early `not np.any(m)` return.
# ---------------------------------------------------------------------------


class TestPdfOnNonPositiveInput:
    @pytest.mark.parametrize(
        "dist",
        [
            BurrXII(alpha=2.0, theta=1.0, gamma=2.0),
            Gamma(alpha=2.0, theta=1.0),
            InverseGaussian(mu=1.0, beta=2.0),
            InverseParalogistic(tau=2.0, theta=1.0),
            InverseTransformedGamma(alpha=2.0, theta=1.0, tau=2.0),
            LogLogistic(theta=1.0, gamma=2.0),
            Lognormal(mu=0.0, sigma=1.0),
            Paralogistic(alpha=2.0, theta=1.0),
            TransformedBeta(alpha=2.0, theta=1.0, gamma=2.0, tau=1.0),
            TransformedGamma(alpha=2.0, theta=1.0, tau=2.0),
            Weibull(theta=1.0, tau=1.5),
        ],
    )
    def test_pdf_zero_for_nonpositive_array(self, dist) -> None:
        out = dist.pdf(np.array([-1.0, -2.0, 0.0]))
        assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# LEV at d=0 returns 0; mean=∞ branches.
# ---------------------------------------------------------------------------


class TestLevAndMeanEdgeBranches:
    @pytest.mark.parametrize(
        "dist",
        [
            Exponential(theta=1.0),
            Pareto(alpha=2.0, theta=1.0),
            Lognormal(mu=0.0, sigma=1.0),
            Weibull(theta=1.0, tau=1.0),
            Gamma(alpha=1.0, theta=1.0),
            BurrXII(alpha=2.0, theta=1.0, gamma=2.0),
            LogLogistic(theta=1.0, gamma=2.0),
            Paralogistic(alpha=2.0, theta=1.0),
            InverseParalogistic(tau=2.0, theta=1.0),
            InverseGaussian(mu=1.0, beta=2.0),
            TransformedGamma(alpha=2.0, theta=1.0, tau=2.0),
            InverseTransformedGamma(alpha=2.0, theta=1.0, tau=2.0),
        ],
    )
    def test_lev_at_zero(self, dist) -> None:
        assert dist.limited_expected_value(0.0) == 0.0
        assert dist.limited_expected_value(-1.0) == 0.0

    def test_pareto_mean_alpha_le_one(self) -> None:
        assert Pareto(alpha=1.0, theta=1.0).mean() == float("inf")

    def test_inverse_paralogistic_lev_falls_back_when_mean_infinite(self) -> None:
        d = InverseParalogistic(tau=0.7, theta=1.0)
        assert d.mean() == float("inf")
        assert d.limited_expected_value(5.0) > 0

    def test_paralogistic_lev_falls_back_when_mean_infinite(self) -> None:
        d = Paralogistic(alpha=0.7, theta=1.0)
        assert d.mean() == float("inf")
        assert d.limited_expected_value(5.0) > 0

    def test_loglogistic_lev_falls_back_when_mean_infinite(self) -> None:
        # already done by test_severity_loglogistic but covers redundantly
        d = LogLogistic(theta=1.0, gamma=0.7)
        assert d.limited_expected_value(5.0) > 0

    def test_inverse_transformed_gamma_lev_falls_back(self) -> None:
        d = InverseTransformedGamma(alpha=0.5, theta=1.0, tau=1.0)
        assert d.mean() == float("inf")
        assert d.limited_expected_value(5.0) > 0

    def test_transformed_beta_lev_falls_back_when_mean_infinite(self) -> None:
        # alpha*gamma <= 1 ⇒ mean infinite
        d = TransformedBeta(alpha=0.5, theta=1.0, gamma=1.0, tau=1.0)
        assert d.mean() == float("inf")
        assert d.limited_expected_value(5.0) > 0


# ---------------------------------------------------------------------------
# Random-state pass-through: rvs accepts a Generator instance directly
# (covers the alternate branch of the rvs(...) helper).
# ---------------------------------------------------------------------------


class TestRvsAcceptsGenerator:
    @pytest.mark.parametrize(
        "dist",
        [
            Exponential(theta=2.0),
            Pareto(alpha=3.0, theta=1.0),
            Lognormal(mu=0.0, sigma=1.0),
            Weibull(theta=1.5, tau=1.5),
            Gamma(alpha=2.0, theta=1.0),
            BurrXII(alpha=3.0, theta=1.0, gamma=2.0),
            LogLogistic(theta=1.0, gamma=2.0),
            Paralogistic(alpha=2.0, theta=1.0),
            InverseParalogistic(tau=2.0, theta=1.0),
            InverseGaussian(mu=1.0, beta=2.0),
            TransformedGamma(alpha=2.0, theta=1.0, tau=2.0),
            InverseTransformedGamma(alpha=2.0, theta=1.0, tau=2.0),
            TransformedBeta(alpha=2.0, theta=1.0, gamma=2.0, tau=1.0),
        ],
    )
    def test_rvs_with_generator_instance(self, dist) -> None:
        rng = np.random.default_rng(0)
        out = dist.rvs(size=10, random_state=rng)
        assert out.shape == (10,)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Lognormal and Pareto: cdf on negative input returns zero (covers the
# `m = x > 0`/`m = x >= 0` mask path explicitly).
# ---------------------------------------------------------------------------


class TestCdfOnNegativeInput:
    def test_lognormal_cdf_zero_for_negative(self) -> None:
        d = Lognormal(mu=0.0, sigma=1.0)
        assert np.all(d.cdf(np.array([-2.0, -1.0])) == 0.0)

    def test_pareto_cdf_zero_for_negative(self) -> None:
        d = Pareto(alpha=2.0, theta=1.0)
        assert np.all(d.cdf(np.array([-2.0, -1.0])) == 0.0)

    def test_weibull_cdf_zero_for_negative(self) -> None:
        d = Weibull(theta=1.0, tau=1.0)
        assert np.all(d.cdf(np.array([-2.0, -1.0])) == 0.0)
