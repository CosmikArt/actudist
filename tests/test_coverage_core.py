"""Coverage tests for the core numerical and MLE plumbing.

Each test targets a specific defensive branch in
``actudist._numerics``, ``actudist._mle``, ``actudist.base``, or
``actudist.fitting`` that the happy-path tests do not exercise.
"""

from __future__ import annotations

import numpy as np
import pytest

from actudist._mle import (
    fit_continuous_mle,
    fit_discrete_mle,
    loglik_continuous,
    loglik_discrete,
)
from actudist._numerics import (
    from_unconstrained,
    numeric_lev,
    numeric_ppf,
    to_unconstrained,
)
from actudist.base import (
    ActuarialDistribution,
    FrequencyDistribution,
    SeverityDistribution,
)
from actudist.fitting import (
    DistributionFitter,
    _resolve,
    register_frequency,
    register_severity,
)
from actudist.frequency.poisson import Poisson
from actudist.severity.exponential import Exponential
from actudist.severity.lognormal import Lognormal


# ---------------------------------------------------------------------------
# _numerics.py
# ---------------------------------------------------------------------------


class TestNumericsEdgeCases:
    def test_numeric_ppf_returns_lower_for_q_zero(self) -> None:
        assert numeric_ppf(lambda x: x, 0.0, lower=0.0, upper=1.0) == 0.0

    def test_numeric_ppf_returns_upper_for_q_one(self) -> None:
        assert numeric_ppf(lambda x: x, 1.0, lower=0.0, upper=1.0) == 1.0

    def test_numeric_ppf_rejects_q_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            numeric_ppf(lambda x: x, -0.1)
        with pytest.raises(ValueError):
            numeric_ppf(lambda x: x, 1.1)

    def test_numeric_lev_returns_zero_for_d_le_zero(self) -> None:
        assert numeric_lev(lambda x: float(np.exp(-x)), 0.0) == 0.0
        assert numeric_lev(lambda x: float(np.exp(-x)), -1.0) == 0.0

    def test_to_unconstrained_log_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            to_unconstrained({"a": -1.0}, [("a", "log")])

    def test_to_unconstrained_logit_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            to_unconstrained({"a": 1.5}, [("a", "logit")])
        with pytest.raises(ValueError):
            to_unconstrained({"a": -0.1}, [("a", "logit")])

    def test_to_unconstrained_unknown_transform(self) -> None:
        with pytest.raises(ValueError):
            to_unconstrained({"a": 1.0}, [("a", "tanh")])

    def test_from_unconstrained_unknown_transform(self) -> None:
        with pytest.raises(ValueError):
            from_unconstrained(np.array([0.0]), [("a", "tanh")])


# ---------------------------------------------------------------------------
# _mle.py
# ---------------------------------------------------------------------------


class TestLoglikContinuous:
    def test_empty_data_returns_zero(self) -> None:
        d = Exponential(theta=1.0)
        assert loglik_continuous(d, np.array([], dtype=float)) == 0.0

    def test_censored_mask_shape_mismatch(self) -> None:
        d = Exponential(theta=1.0)
        with pytest.raises(ValueError):
            loglik_continuous(
                d,
                np.array([1.0, 2.0, 3.0]),
                censored=np.array([True, False]),
            )

    def test_truncation_returns_neg_inf_when_prob_zero(self) -> None:
        # trunc_lower above any positive support ⇒ S(lower) - S(upper) = 0
        d = Exponential(theta=1.0)
        ll = loglik_continuous(
            d,
            np.array([0.5]),
            trunc_lower=10.0,
            trunc_upper=10.0,
        )
        assert ll == -np.inf

    def test_truncation_lower_only(self) -> None:
        d = Exponential(theta=1.0)
        ll = loglik_continuous(d, np.array([1.0]), trunc_lower=0.5)
        assert np.isfinite(ll)

    def test_truncation_upper_only(self) -> None:
        d = Exponential(theta=1.0)
        ll = loglik_continuous(d, np.array([1.0]), trunc_upper=5.0)
        assert np.isfinite(ll)


class TestLoglikDiscrete:
    def test_empty_data_returns_zero(self) -> None:
        assert loglik_discrete(Poisson(lam=1.0), np.array([], dtype=int)) == 0.0


class TestFitContinuousMleNelderMeadFallback:
    def test_handles_optimizer_recovery(self) -> None:
        # All-equal data is degenerate for many distributions; the driver
        # should still produce some parameter set without raising.
        data = np.full(20, 2.0)
        params = fit_continuous_mle(Exponential, data)
        assert "theta" in params


class TestFitDiscreteMleNelderMeadFallback:
    def test_explicit_initial_params(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.poisson(lam=2.0, size=200)
        # exercising the initial_params kwarg branch
        params = fit_discrete_mle(Poisson, data, initial_params={"lam": 1.5})
        assert params["lam"] == pytest.approx(2.0, rel=0.1)


class TestProfileLikelihoodEdgeCases:
    def test_unknown_param_raises(self) -> None:
        rng = np.random.default_rng(0)
        data = Exponential(theta=2.0).rvs(size=200, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        with pytest.raises(KeyError):
            fit.profile_likelihood_ci(data, "not_a_param")

    def test_works_for_discrete_distribution(self) -> None:
        rng = np.random.default_rng(0)
        data = Poisson(lam=3.0).rvs(size=2_000, random_state=rng)
        fit = Poisson()
        fit.mle_fit(data)
        lo, hi = fit.profile_likelihood_ci(data, "lam", alpha=0.05)
        assert lo < 3.0 < hi


# ---------------------------------------------------------------------------
# base.py: abstract-method NotImplementedError paths
# ---------------------------------------------------------------------------


class TestActuarialDistributionAbstract:
    def test_cdf_raises(self) -> None:
        d = ActuarialDistribution()
        with pytest.raises(NotImplementedError):
            d.cdf(np.array([0.5]))

    def test_ppf_raises(self) -> None:
        d = ActuarialDistribution()
        with pytest.raises(NotImplementedError):
            d.ppf(np.array([0.5]))

    def test_rvs_raises(self) -> None:
        d = ActuarialDistribution()
        with pytest.raises(NotImplementedError):
            d.rvs(size=10)

    def test_loglik_raises(self) -> None:
        d = ActuarialDistribution()
        with pytest.raises(NotImplementedError):
            d.loglik(np.array([0.5]))

    def test_transforms_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            ActuarialDistribution._transforms()

    def test_initial_guess_default_falls_back(self) -> None:
        # subclass that declares transforms but doesn't override _initial_guess
        class _Tiny(ActuarialDistribution):
            @classmethod
            def _transforms(cls):
                return [("a", "log"), ("b", "identity")]

        guess = _Tiny._initial_guess(np.array([1.0, 2.0]))
        assert guess == {"a": 1.0, "b": 1.0}


class TestSeverityDistributionAbstract:
    def test_pdf_raises(self) -> None:
        d = SeverityDistribution()
        with pytest.raises(NotImplementedError):
            d.pdf(np.array([0.5]))

    def test_mean_raises(self) -> None:
        d = SeverityDistribution()
        with pytest.raises(NotImplementedError):
            d.mean()

    def test_hazard_rate_uses_pdf_and_survival(self) -> None:
        d = Exponential(theta=2.0)
        h = d.hazard_rate(np.array([1.0, 3.0]))
        # for exponential, hazard rate = 1/theta everywhere
        assert h[0] == pytest.approx(0.5, rel=1e-9)
        assert h[1] == pytest.approx(0.5, rel=1e-9)

    def test_default_limited_expected_value_uses_numeric_lev(self) -> None:
        # Build a custom severity that does NOT override LEV; default uses
        # numeric_lev. Use Exponential semantics for verification.
        class _ExpNoLev(SeverityDistribution):
            n_params = 1

            def __init__(self, theta=1.0):
                super().__init__(params={"theta": float(theta)})

            def cdf(self, x):
                x = np.asarray(x, dtype=float)
                out = np.zeros_like(x)
                out[x >= 0] = 1.0 - np.exp(-x[x >= 0] / self.theta)
                return out

            def pdf(self, x):
                x = np.asarray(x, dtype=float)
                out = np.zeros_like(x)
                out[x >= 0] = np.exp(-x[x >= 0] / self.theta) / self.theta
                return out

            def mean(self):
                return float(self.theta)

        d = _ExpNoLev(theta=2.0)
        # closed form: theta * (1 - exp(-d/theta)) = 2 * (1 - e^-1)
        assert d.limited_expected_value(2.0) == pytest.approx(
            2.0 * (1.0 - np.exp(-1.0)), rel=1e-6
        )

    def test_increased_limits_factor_zero_base_raises(self) -> None:
        d = Exponential(theta=2.0)
        with pytest.raises(ValueError):
            d.increased_limits_factor(d=1.0, base_d=0.0)


class TestFrequencyDistributionAbstract:
    def test_pmf_raises(self) -> None:
        d = FrequencyDistribution()
        with pytest.raises(NotImplementedError):
            d.pmf(np.array([0]))


# ---------------------------------------------------------------------------
# fitting.py
# ---------------------------------------------------------------------------


class TestFittingResolveAndRegister:
    def test_register_frequency_rejects_wrong_base(self) -> None:
        with pytest.raises(TypeError):

            @register_frequency("__bad_freq__")
            class _Bad:  # type: ignore[no-redef]
                pass

    def test_register_severity_rejects_wrong_base(self) -> None:
        with pytest.raises(TypeError):

            @register_severity("__bad_sev__")
            class _Bad:  # type: ignore[no-redef]
                pass

    def test_resolve_passes_through_instance(self) -> None:
        d = Exponential(theta=1.5)
        assert _resolve(d) is d

    def test_resolve_finds_frequency_by_name(self) -> None:
        inst = _resolve("Poisson")
        assert isinstance(inst, Poisson)

    def test_resolve_unknown_name_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            _resolve("NonExistent")

    def test_fitter_uses_instance_candidate(self) -> None:
        rng = np.random.default_rng(0)
        data = Exponential(theta=2.0).rvs(size=300, random_state=rng)
        fitter = DistributionFitter(candidates=[Lognormal(), Exponential()])
        rows = fitter.fit_and_rank(data)
        # the row name comes from type(inst).__name__ for instance candidates
        names = {r["name"] for r in rows}
        assert names == {"Lognormal", "Exponential"}


# ---------------------------------------------------------------------------
# gof.py
# ---------------------------------------------------------------------------


class TestGoFDefensive:
    def test_pp_plot_without_distribution_raises(self) -> None:
        from actudist import GoodnessOfFit

        gof = GoodnessOfFit(distribution=None, data=np.array([1.0]))
        with pytest.raises(ValueError):
            gof.pp_plot()

    def test_qq_plot_without_distribution_raises(self) -> None:
        from actudist import GoodnessOfFit

        gof = GoodnessOfFit(distribution=None, data=np.array([1.0]))
        with pytest.raises(ValueError):
            gof.qq_plot()

    def test_bootstrap_swallows_fit_failures(self) -> None:
        """Construct a degenerate case where MLE on the resampled draws
        sometimes fails, so the ``except: continue`` branch is exercised."""
        from actudist import GoodnessOfFit

        rng = np.random.default_rng(0)
        # Very small sample raises the chance of a degenerate bootstrap
        # sample; the GoF helper should still produce a p-value without
        # propagating the inner exception.
        true = Exponential(theta=2.0)
        data = true.rvs(size=20, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        gof = GoodnessOfFit(distribution=fit, data=data)
        out = gof.ks_test(n_boot=50, random_state=rng)
        assert 0.0 <= out["p_value"] <= 1.0
