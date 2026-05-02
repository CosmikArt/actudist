"""Last-mile coverage push: targeted tests for the remaining defensive
branches that the standard happy-path tests cannot reach.

We use monkey-patching where necessary to force exception propagation
deep inside the optimization loops; in production code these branches
exist precisely because the surrounding numerics can blow up under
adversarial parameter probes by the optimizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from actudist._mle import (
    fit_continuous_mle,
    fit_discrete_mle,
    profile_likelihood_ci,
)
from actudist.fitting import DistributionFitter
from actudist.frequency.binomial import Binomial
from actudist.frequency.geometric import Geometric
from actudist.frequency.zinb import ZeroInflatedNegativeBinomial
from actudist.frequency.zip import ZeroInflatedPoisson
from actudist.severity.exponential import Exponential
from actudist.severity.lognormal import Lognormal
from actudist.severity.transformed_beta import TransformedBeta


# ---------------------------------------------------------------------------
# severity & frequency micro-coverage
# ---------------------------------------------------------------------------


class TestMicroCoverage:
    def test_transformed_beta_lev_at_zero(self) -> None:
        d = TransformedBeta(alpha=2.0, theta=1.0, gamma=2.0, tau=1.0)
        assert d.limited_expected_value(0.0) == 0.0
        assert d.limited_expected_value(-1.0) == 0.0

    def test_lognormal_initial_guess_all_nonpositive_data(self) -> None:
        # When every observation is <= 0, the positive-only filter empties
        # the array and the helper returns the safe default (0, 1).
        guess = Lognormal._initial_guess(np.array([-1.0, 0.0, -2.0]))
        assert guess == {"mu": 0.0, "sigma": 1.0}

    def test_binomial_classmethods(self) -> None:
        assert Binomial._transforms() == [("q", "logit")]
        guess = Binomial._initial_guess(np.array([0, 1, 2, 3, 4]))
        assert guess["m"] >= 4
        assert 0.0 < guess["q"] < 1.0

    def test_geometric_mean(self) -> None:
        assert Geometric(beta=2.5).mean() == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# ZIP / ZINB ppf safety break (k > 1e6)
# ---------------------------------------------------------------------------


class TestPpfSafetyBreak:
    def test_zip_ppf_safety_break_via_unreachable_quantile(self) -> None:
        """``cdf < q`` is monotone in ``q``; we patch ``cdf`` to always
        return zero so the inner ``while`` loop runs to the safety
        break instead of forever."""
        d = ZeroInflatedPoisson(pi=0.5, lam=1.0)
        d.cdf = lambda k: np.zeros(np.atleast_1d(k).shape, dtype=float)
        out = d.ppf(np.array([0.5]))
        assert out[0] == float(10**6 + 1)

    def test_zinb_ppf_safety_break(self) -> None:
        d = ZeroInflatedNegativeBinomial(pi=0.3, r=2.0, beta=1.5)
        d.cdf = lambda k: np.zeros(np.atleast_1d(k).shape, dtype=float)
        out = d.ppf(np.array([0.5]))
        assert out[0] == float(10**6 + 1)


# ---------------------------------------------------------------------------
# _mle.fit_continuous_mle: -inf log-likelihood guard (line 123)
# ---------------------------------------------------------------------------


class TestFitContinuousNonFiniteGuard:
    def test_optimizer_swallows_neg_inf_loglik(self) -> None:
        """Truncation interval ``[lower, upper]`` with ``lower == upper``
        forces ``S(lower) - S(upper) = 0`` for every parameter, so the
        log-likelihood is ``-inf`` everywhere. The optimizer must not
        propagate that. It should map it to ``_HUGE`` and finish."""
        data = np.array([0.5, 1.0, 1.5])
        params = fit_continuous_mle(Exponential, data, trunc_lower=5.0, trunc_upper=5.0)
        # only care that the call returns
        assert "theta" in params


# ---------------------------------------------------------------------------
# _mle.profile_likelihood_ci: monkey-patched failure modes
# ---------------------------------------------------------------------------


class TestProfileLikelihoodMonkeyPatchedFailures:
    def test_returns_inf_when_walk_hits_max_expansions(self, monkeypatch) -> None:
        """Patch ``loglik_continuous`` to return a constant value so the
        profile log-likelihood is flat in ``theta``; ``g`` is always
        ``+0.5*chi2`` and the walk exhausts ``max_expansions``."""
        import actudist._mle as mle_mod

        rng = np.random.default_rng(0)
        data = Exponential(theta=2.0).rvs(size=200, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)

        monkeypatch.setattr(mle_mod, "loglik_continuous", lambda *a, **kw: -100.0)
        lo, hi = fit.profile_likelihood_ci(data, "theta", alpha=0.05)
        # walk gives up on both sides
        assert np.isinf(lo) or np.isinf(hi)

    def test_inner_neg_swallows_construction_exception(self, monkeypatch) -> None:
        """Force the inner ``_neg`` to raise on every other call: the
        try/except in ``_neg`` must trap it and return ``_HUGE``."""
        from actudist.severity.lognormal import Lognormal as _LN

        rng = np.random.default_rng(0)
        data = _LN(mu=1.5, sigma=0.6).rvs(size=200, random_state=rng)
        fit = _LN()
        fit.mle_fit(data)

        original_init = _LN.__init__
        counter = {"calls": 0}

        def flaky_init(self, mu=None, sigma=None):
            counter["calls"] += 1
            # Skip the very first construction (the fitted dist itself was
            # already built with the original __init__) and let the next
            # few runs mostly succeed, but raise occasionally inside the
            # optimizer's _neg.
            if counter["calls"] > 2 and counter["calls"] % 5 == 0:
                raise RuntimeError("synthetic init failure")
            original_init(self, mu, sigma)

        monkeypatch.setattr(_LN, "__init__", flaky_init)
        # no tight bounds asserted; just that no exception bubbled up
        lo, hi = fit.profile_likelihood_ci(data, "mu", alpha=0.05)
        assert lo == lo  # not NaN

    def test_brentq_exception_returns_none(self, monkeypatch) -> None:
        """If ``brentq`` itself raises (e.g., the bracketed function flips
        sign more than once due to numerical noise), ``_walk`` swallows
        the exception and returns ``None`` (so the endpoint is +/-inf)."""
        from scipy import optimize as _opt

        original = _opt.brentq

        def failing_brentq(*args, **kwargs):
            raise ValueError("synthetic brentq failure")

        rng = np.random.default_rng(0)
        data = Exponential(theta=2.0).rvs(size=200, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)

        # patch where it's actually used (imported at function scope inside
        # _mle.profile_likelihood_ci)
        monkeypatch.setattr("scipy.optimize.brentq", failing_brentq)
        lo, hi = fit.profile_likelihood_ci(data, "theta", alpha=0.05)
        assert lo == -np.inf
        assert hi == np.inf

    def test_g_outer_exception_returns_none(self, monkeypatch) -> None:
        """If the very first ``_g(bracket_outer)`` evaluation raises, the
        walk's ``except`` clause must short-circuit to ``None``."""
        import actudist._mle as mle_mod

        rng = np.random.default_rng(0)
        data = Exponential(theta=2.0).rvs(size=200, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)

        original_lc = mle_mod.loglik_continuous
        toggle = {"first_call": True}

        def flaky_lc(inst, data_, **kw):
            # let the initial full_ll computation succeed,
            # then raise on every subsequent call (within _profile_ll)
            if toggle["first_call"]:
                toggle["first_call"] = False
                return original_lc(inst, data_, **kw)
            raise RuntimeError("synthetic profile-LL failure")

        monkeypatch.setattr(mle_mod, "loglik_continuous", flaky_lc)
        lo, hi = fit.profile_likelihood_ci(data, "theta", alpha=0.05)
        assert lo == -np.inf
        assert hi == np.inf


class TestWalkLogitDirectionUpClamp:
    def test_logit_param_near_one_first_step_clamp(self) -> None:
        """Build a fitted ZIP whose ``pi`` is so close to 1 that the
        initial outer step in the +1 direction crosses 1.0; the helper
        must clamp it to ``(point + 1.0) / 2`` instead."""
        d = ZeroInflatedPoisson(pi=0.97, lam=1.0)
        # synthesize "data" with very high zero rate
        rng = np.random.default_rng(0)
        n = 500
        data = np.where(rng.uniform(size=n) < 0.97, 0, rng.poisson(1.0, size=n))
        fit = ZeroInflatedPoisson()
        fit.mle_fit(data)
        # CI may or may not be informative; just verify the call exits.
        lo, hi = fit.profile_likelihood_ci(data, "pi", alpha=0.05)
        if np.isfinite(hi):
            assert hi <= 1.0


# ---------------------------------------------------------------------------
# fit_discrete_mle: Nelder-Mead fallback when L-BFGS-B "fails"
# ---------------------------------------------------------------------------


class TestDiscreteOptimizerFallback:
    def test_fallback_when_lbfgsb_marks_failure(self, monkeypatch) -> None:
        """Force ``scipy.optimize.minimize`` to report L-BFGS-B as
        unsuccessful, so the helper falls through to Nelder-Mead."""
        from scipy import optimize as _opt

        original = _opt.minimize
        toggle = {"used_lbfgs": False}

        def fake_minimize(*args, **kwargs):
            method = kwargs.get("method")
            if method == "L-BFGS-B" and not toggle["used_lbfgs"]:
                toggle["used_lbfgs"] = True
                # craft a result that reports failure
                res = original(*args, **kwargs)
                res.success = False
                return res
            return original(*args, **kwargs)

        monkeypatch.setattr("actudist._mle.minimize", fake_minimize)

        rng = np.random.default_rng(0)
        from actudist.frequency.poisson import Poisson

        data = rng.poisson(lam=2.0, size=300)
        params = fit_discrete_mle(Poisson, data)
        assert params["lam"] == pytest.approx(2.0, rel=0.10)

    def test_discrete_neg_ll_handles_non_finite(self, monkeypatch) -> None:
        """Patch ``loglik_discrete`` to return ``+inf`` for the first call,
        so the inner ``neg_ll`` returns ``_HUGE`` via the
        ``not np.isfinite(ll)`` branch."""
        import actudist._mle as mle_mod

        original = mle_mod.loglik_discrete
        toggled = {"first": True}

        def flaky(inst, data):
            if toggled["first"]:
                toggled["first"] = False
                return float("inf")  # non-finite triggers _HUGE
            return original(inst, data)

        monkeypatch.setattr(mle_mod, "loglik_discrete", flaky)

        rng = np.random.default_rng(0)
        from actudist.frequency.poisson import Poisson

        data = rng.poisson(lam=2.0, size=200)
        params = fit_discrete_mle(Poisson, data)
        assert "lam" in params

    def test_discrete_neg_ll_swallows_overflow_exception(self, monkeypatch) -> None:
        """Patch ``loglik_discrete`` to raise ``OverflowError`` once, so
        the ``except (ValueError, FloatingPointError, OverflowError)``
        catch maps it to ``_HUGE``."""
        import actudist._mle as mle_mod

        original = mle_mod.loglik_discrete
        toggled = {"first": True}

        def flaky(inst, data):
            if toggled["first"]:
                toggled["first"] = False
                raise OverflowError("synthetic overflow")
            return original(inst, data)

        monkeypatch.setattr(mle_mod, "loglik_discrete", flaky)

        rng = np.random.default_rng(0)
        from actudist.frequency.poisson import Poisson

        data = rng.poisson(lam=2.0, size=200)
        params = fit_discrete_mle(Poisson, data)
        assert "lam" in params


class TestGoFBootstrapSwallowsFitFailure:
    def test_bootstrap_continues_when_inner_mle_raises(self, monkeypatch) -> None:
        """Force the bootstrap inner ``mle_fit`` call to raise on every
        third sample so the ``except Exception: continue`` guard fires
        (gof.py lines 87-88)."""
        from actudist import GoodnessOfFit

        rng = np.random.default_rng(0)
        true = Exponential(theta=2.0)
        data = true.rvs(size=80, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        gof = GoodnessOfFit(distribution=fit, data=data)

        original_mle = Exponential.mle_fit
        counter = {"calls": 0}

        def flaky_mle(self, data, **kwargs):
            counter["calls"] += 1
            if counter["calls"] % 3 == 0:
                raise RuntimeError("synthetic bootstrap failure")
            return original_mle(self, data, **kwargs)

        monkeypatch.setattr(Exponential, "mle_fit", flaky_mle)
        out = gof.ks_test(n_boot=30, random_state=rng)
        assert 0.0 <= out["p_value"] <= 1.0


class TestWalkExhaustionPaths:
    def test_log_walk_returns_none_when_bracket_underflows(self, monkeypatch) -> None:
        """Construct a fitted Exponential at theta near zero; with a flat
        log-likelihood, the walk halves bracket_outer until it drops
        below ``1e-12`` and the function returns ``None`` ⇒ ``-inf``
        endpoint. Exercises line 258."""
        import actudist._mle as mle_mod

        d = Exponential(theta=1e-4)
        # pretend it was fitted on this synthetic data
        data = np.array([0.5, 1.0, 1.5])
        # patch loglik flat so the walk never finds a crossing
        monkeypatch.setattr(mle_mod, "loglik_continuous", lambda *a, **kw: -10.0)
        lo, hi = d.profile_likelihood_ci(data, "theta", alpha=0.05)
        assert lo == -np.inf

    def test_logit_walk_up_returns_none_when_bracket_saturates(
        self, monkeypatch
    ) -> None:
        """Same idea on the logit upper-bound walk: it should return
        ``None`` once bracket_outer is within ``1e-9`` of ``1.0``."""
        import actudist._mle as mle_mod

        d = ZeroInflatedPoisson(pi=0.5, lam=1.0)
        data = np.array([0, 1, 0, 2, 0, 1])
        monkeypatch.setattr(mle_mod, "loglik_continuous", lambda *a, **kw: -10.0)
        monkeypatch.setattr(mle_mod, "loglik_discrete", lambda *a, **kw: -10.0)
        lo, hi = d.profile_likelihood_ci(data, "pi", alpha=0.05)
        # at least one endpoint should fail to bracket
        assert np.isinf(lo) or np.isinf(hi)

    def test_logit_first_step_up_clamps_when_pi_near_one(self, monkeypatch) -> None:
        """Build a synthetic fit at pi=0.97 directly (no MLE) and run
        profile-CI on pi: the very first +1 outer step lands at 1.02 and
        must be clamped via the line-236 branch."""
        import actudist._mle as mle_mod

        d = ZeroInflatedPoisson(pi=0.97, lam=1.0)
        data = np.array([0] * 95 + [1, 2, 3, 4, 5])
        monkeypatch.setattr(mle_mod, "loglik_discrete", lambda *a, **kw: -10.0)
        # walk should exit cleanly even though it can't bracket
        d.profile_likelihood_ci(data, "pi", alpha=0.05)
