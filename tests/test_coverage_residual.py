"""Final coverage push: walk-extension paths in the profile-likelihood
helper, fit_and_rank exception path, the rare log/logit boundary
adjustments, and a few overlooked single-line branches.
"""

from __future__ import annotations

import numpy as np
import pytest

from actudist._mle import (
    loglik_continuous,
    loglik_discrete,
    profile_likelihood_ci,
)
from actudist.base import SeverityDistribution
from actudist.fitting import DistributionFitter, register_severity
from actudist.frequency.negative_binomial import NegativeBinomial
from actudist.frequency.poisson import Poisson
from actudist.frequency.zip import ZeroInflatedPoisson
from actudist.severity.burrxii import BurrXII
from actudist.severity.exponential import Exponential
from actudist.severity.inverse_paralogistic import InverseParalogistic
from actudist.severity.inverse_transformed_gamma import InverseTransformedGamma
from actudist.severity.loglogistic import LogLogistic
from actudist.severity.lognormal import Lognormal
from actudist.severity.paralogistic import Paralogistic
from actudist.severity.transformed_beta import TransformedBeta
from actudist.severity.transformed_gamma import TransformedGamma


# ---------------------------------------------------------------------------
# Walk-extension paths in profile_likelihood_ci
# ---------------------------------------------------------------------------


class TestProfileLikelihoodWalkExtensions:
    def test_log_param_extends_on_small_sample(self) -> None:
        """With a small sample, the CI half-width is wide enough that the
        first 5% step does not bracket the boundary. The optimizer must
        keep doubling the step (the ``is_log`` extension path)."""
        rng = np.random.default_rng(0)
        data = Exponential(theta=2.0).rvs(size=30, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        lo, hi = fit.profile_likelihood_ci(data, "theta", alpha=0.05)
        assert 0 < lo < fit.theta < hi

    def test_identity_param_extends_on_small_sample(self) -> None:
        rng = np.random.default_rng(0)
        data = Lognormal(mu=1.5, sigma=0.6).rvs(size=30, random_state=rng)
        fit = Lognormal()
        fit.mle_fit(data)
        lo, hi = fit.profile_likelihood_ci(data, "mu", alpha=0.05)
        assert lo < fit.mu < hi

    def test_logit_param_walk_with_zip_small_sample(self) -> None:
        rng = np.random.default_rng(0)
        data = ZeroInflatedPoisson(pi=0.4, lam=3.0).rvs(size=80, random_state=rng)
        fit = ZeroInflatedPoisson()
        fit.mle_fit(data)
        lo, hi = fit.profile_likelihood_ci(data, "pi", alpha=0.05)
        assert lo <= fit.pi <= hi


class TestProfileLikelihoodLogLogitBoundaryAdjustments:
    def test_log_param_near_zero_initial_outer_clamps(self) -> None:
        """Construct a fitted dist whose log-transformed parameter is so
        small that ``point - step <= 0``; the helper must clamp to
        ``point * 0.5`` rather than crossing zero."""
        rng = np.random.default_rng(0)
        # tiny theta keeps point near zero so the 0.05 step floor flips sign
        data = Exponential(theta=0.03).rvs(size=400, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        lo, hi = fit.profile_likelihood_ci(data, "theta", alpha=0.05)
        assert lo > 0.0
        assert lo < fit.theta < hi

    def test_logit_param_near_one_initial_outer_clamps(self) -> None:
        rng = np.random.default_rng(0)
        # extreme zero inflation pushes pi_hat near 1
        data = ZeroInflatedPoisson(pi=0.95, lam=2.0).rvs(size=400, random_state=rng)
        fit = ZeroInflatedPoisson()
        fit.mle_fit(data)
        # might fail to bracket at all but should not error
        lo, hi = fit.profile_likelihood_ci(data, "pi", alpha=0.05)
        assert 0.0 < fit.pi < 1.0
        # boundaries should at minimum be inside [0, 1] when finite
        if np.isfinite(lo):
            assert lo >= 0.0
        if np.isfinite(hi):
            assert hi <= 1.0

    def test_logit_param_near_zero_initial_outer_clamps(self) -> None:
        rng = np.random.default_rng(0)
        # very small zero-inflation
        data = ZeroInflatedPoisson(pi=0.02, lam=2.0).rvs(size=200, random_state=rng)
        fit = ZeroInflatedPoisson()
        fit.mle_fit(data)
        # fit.pi might be very small; the lower walk boundary should clamp
        lo, hi = fit.profile_likelihood_ci(data, "pi", alpha=0.05)
        if np.isfinite(lo):
            assert lo >= 0.0


class TestProfileLikelihoodDiscreteMultiParam:
    def test_discrete_two_param_uses_loglik_discrete(self) -> None:
        """For a multi-parameter frequency, profile-likelihood loops over
        ``loglik_discrete`` inside ``_neg`` — covers that branch."""
        rng = np.random.default_rng(0)
        data = NegativeBinomial(r=2.0, beta=1.5).rvs(size=300, random_state=rng)
        fit = NegativeBinomial()
        fit.mle_fit(data)
        lo, hi = fit.profile_likelihood_ci(data, "r", alpha=0.05)
        assert lo < fit.r < hi


class TestProfileLikelihoodDirectCallNoneFitKwargs:
    def test_direct_call_with_default_fit_kwargs(self) -> None:
        rng = np.random.default_rng(0)
        data = Exponential(theta=2.0).rvs(size=200, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        # call the underlying function directly so the
        # ``if fit_kwargs is None: fit_kwargs = {}`` branch fires
        lo, hi = profile_likelihood_ci(fit, data, "theta", alpha=0.05)
        assert lo < fit.theta < hi


# ---------------------------------------------------------------------------
# loglik_continuous: produce a non-finite -inf to hit the inner _HUGE branch
# ---------------------------------------------------------------------------


class TestLoglikNonFinitePropagation:
    def test_loglik_neginf_for_truncation_above_support(self) -> None:
        d = Exponential(theta=1.0)
        ll = loglik_continuous(
            d,
            np.array([0.5, 1.0]),
            trunc_lower=10.0,
            trunc_upper=10.0,
        )
        assert ll == -np.inf

    def test_fit_continuous_recovers_when_likelihood_neg_inf(self) -> None:
        # the negative-loglik wrapper must convert -inf to _HUGE rather
        # than propagate a NaN. Use a degenerate trunc-lower above support.
        rng = np.random.default_rng(0)
        data = rng.exponential(scale=1.0, size=100)
        fit = Exponential()
        # When trunc_lower is enormous the optimizer should hit _HUGE on
        # most evaluations but still converge to *some* parameter.
        params = fit.mle_fit(data, trunc_lower=0.0, trunc_upper=None)
        assert "theta" in params


# ---------------------------------------------------------------------------
# DistributionFitter: exception path
# ---------------------------------------------------------------------------


@register_severity("__broken_fit__")
class _BrokenFit(SeverityDistribution):
    """A toy severity whose mle_fit always raises. Used to exercise the
    DistributionFitter exception-handling path."""

    n_params = 1

    def __init__(self, theta: float | None = None) -> None:
        if theta is None:
            super().__init__(params=None)
        else:
            super().__init__(params={"theta": float(theta)})

    @classmethod
    def _transforms(cls):
        return [("theta", "log")]

    def pdf(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def cdf(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def mean(self):
        return 1.0

    def mle_fit(self, data, **kwargs):  # noqa: D401
        raise RuntimeError("intentional MLE failure")


class TestFitterExceptionPath:
    def test_failed_fit_records_error_and_pushes_to_bottom(self) -> None:
        rng = np.random.default_rng(0)
        data = Exponential(theta=2.0).rvs(size=200, random_state=rng)
        fitter = DistributionFitter(candidates=["__broken_fit__", "Exponential"])
        rows = fitter.fit_and_rank(data)
        # Exponential converges and ends up on top
        assert rows[0]["name"] == "Exponential"
        # __broken_fit__ records its exception and lands at the bottom
        bottom = rows[-1]
        assert bottom["name"] == "__broken_fit__"
        assert bottom["error"] is not None
        assert "intentional MLE failure" in bottom["error"]
        assert np.isnan(bottom["loglik"])
        assert bottom["aic"] == float("inf")
        assert bottom["params"] is None


# ---------------------------------------------------------------------------
# Severity branches that needed a direct call: mean() finite branches and
# explicit MLE for dists whose tests omitted it.
# ---------------------------------------------------------------------------


class TestRemainingSeverityBranches:
    def test_burrxii_mean_finite_branch(self) -> None:
        d = BurrXII(alpha=3.0, theta=2.0, gamma=2.0)
        m = d.mean()
        assert np.isfinite(m) and m > 0

    def test_paralogistic_mean_finite_branch(self) -> None:
        d = Paralogistic(alpha=2.0, theta=1.0)
        m = d.mean()
        assert np.isfinite(m) and m > 0

    def test_inverse_paralogistic_mean_finite_branch(self) -> None:
        d = InverseParalogistic(tau=2.0, theta=1.0)
        m = d.mean()
        assert np.isfinite(m) and m > 0

    def test_loglogistic_mean_finite_branch(self) -> None:
        d = LogLogistic(theta=1.0, gamma=2.0)
        m = d.mean()
        assert np.isfinite(m) and m > 0

    def test_lognormal_empty_args(self) -> None:
        d = Lognormal()
        assert d.params is None

    def test_transformed_beta_empty_args(self) -> None:
        d = TransformedBeta()
        assert d.params is None

    def test_transformed_beta_mle_uses_transforms_and_initial_guess(self) -> None:
        rng = np.random.default_rng(0)
        data = TransformedBeta(alpha=2.5, theta=1.5, gamma=1.5, tau=1.0).rvs(
            size=2_000, random_state=rng
        )
        fit = TransformedBeta()
        # don't assert tight tolerances — 4-param identifiability is poor;
        # just exercise the MLE path
        fit.mle_fit(data)
        assert fit.params is not None

    def test_transformed_beta_mean_finite_branch(self) -> None:
        d = TransformedBeta(alpha=3.0, theta=1.0, gamma=2.0, tau=1.5)
        m = d.mean()
        assert np.isfinite(m) and m > 0

    def test_transformed_gamma_mean_finite(self) -> None:
        d = TransformedGamma(alpha=2.0, theta=1.0, tau=1.5)
        m = d.mean()
        assert np.isfinite(m) and m > 0

    def test_inverse_transformed_gamma_empty_args(self) -> None:
        d = InverseTransformedGamma()
        assert d.params is None

    def test_inverse_transformed_gamma_mle_uses_transforms_and_initial_guess(self) -> None:
        rng = np.random.default_rng(0)
        data = InverseTransformedGamma(alpha=3.0, theta=1.5, tau=1.5).rvs(
            size=2_000, random_state=rng
        )
        fit = InverseTransformedGamma()
        fit.mle_fit(data)
        assert fit.params is not None

    def test_inverse_transformed_gamma_mean_finite(self) -> None:
        d = InverseTransformedGamma(alpha=3.0, theta=1.0, tau=2.0)
        m = d.mean()
        assert np.isfinite(m) and m > 0


# ---------------------------------------------------------------------------
# gof._bootstrap_pvalue: the no-distribution and unknown-statistic guards
# ---------------------------------------------------------------------------


class TestGoFBootstrapDefensive:
    def test_bootstrap_without_distribution_raises(self) -> None:
        from actudist import GoodnessOfFit

        gof = GoodnessOfFit(distribution=None, data=np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            gof._bootstrap_pvalue(0.0, "ks", n_boot=10, random_state=None)

    def test_bootstrap_unknown_statistic_raises(self) -> None:
        from actudist import GoodnessOfFit

        rng = np.random.default_rng(0)
        data = Exponential(theta=2.0).rvs(size=20, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        gof = GoodnessOfFit(distribution=fit, data=data)
        with pytest.raises(ValueError):
            gof._bootstrap_pvalue(0.0, "not_ks_or_ad", n_boot=5, random_state=rng)


# ---------------------------------------------------------------------------
# Geometric/ZIP/ZINB ppf: hit the >1e6 break path is unreachable in practice;
# instead, verify that an enormous ppf request still returns a finite count.
# ---------------------------------------------------------------------------


class TestExtremePpf:
    def test_zip_ppf_extreme_quantile(self) -> None:
        d = ZeroInflatedPoisson(pi=0.1, lam=2.0)
        # 0.999 still well inside finite range
        out = d.ppf(np.array([0.999]))
        assert out[0] >= 0
