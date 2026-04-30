"""Smoke tests for the new modular layout: imports, base hierarchy, registry
mechanics, and MLE-helper sanity. Concrete distributions are tested in the
``test_severity/`` and ``test_frequency/`` packages once they land.
"""

from __future__ import annotations

import numpy as np

import actudist
from actudist.base import (
    ActuarialDistribution,
    FrequencyDistribution,
    SeverityDistribution,
)
from actudist.fitting import (
    DistributionFitter,
    FREQUENCY_REGISTRY,
    SEVERITY_REGISTRY,
    register_frequency,
    register_severity,
)
from actudist.gof import GoodnessOfFit
from actudist._numerics import (
    from_unconstrained,
    numeric_lev,
    numeric_ppf,
    safe_log,
    to_unconstrained,
)
from actudist._mle import (
    fit_continuous_mle,
    fit_discrete_mle,
    loglik_continuous,
    loglik_discrete,
)


class TestVersion:
    def test_version_is_str(self) -> None:
        assert isinstance(actudist.__version__, str)


class TestPublicSurface:
    def test_top_level_reexports(self) -> None:
        assert actudist.ActuarialDistribution is ActuarialDistribution
        assert actudist.SeverityDistribution is SeverityDistribution
        assert actudist.FrequencyDistribution is FrequencyDistribution
        assert actudist.DistributionFitter is DistributionFitter
        assert actudist.GoodnessOfFit is GoodnessOfFit
        assert actudist.SEVERITY_REGISTRY is SEVERITY_REGISTRY
        assert actudist.FREQUENCY_REGISTRY is FREQUENCY_REGISTRY


class TestBaseHierarchy:
    def test_severity_extends_base(self) -> None:
        assert issubclass(SeverityDistribution, ActuarialDistribution)

    def test_frequency_extends_base(self) -> None:
        assert issubclass(FrequencyDistribution, ActuarialDistribution)

    def test_severity_and_frequency_are_disjoint(self) -> None:
        assert not issubclass(SeverityDistribution, FrequencyDistribution)
        assert not issubclass(FrequencyDistribution, SeverityDistribution)

    def test_information_criteria_use_loglik_and_n_params(self) -> None:
        # Build a one-off subclass with a constant log-likelihood and check
        # AIC / BIC arithmetic.
        class _Const(ActuarialDistribution):
            n_params = 3

            def loglik(self, data, **kw):  # type: ignore[override]
                return -10.0

        d = _Const()
        data = np.arange(100)
        assert d.aic(data) == 26.0  # -2*(-10) + 2*3
        # BIC = 20 + 3*log(100)
        assert abs(d.bic(data) - (20.0 + 3.0 * np.log(100))) < 1e-12


class TestRegistryDecorators:
    def test_register_severity_inserts(self) -> None:
        @register_severity("__test_severity__")
        class _D(SeverityDistribution):
            pass

        assert SEVERITY_REGISTRY["__test_severity__"] is _D
        del SEVERITY_REGISTRY["__test_severity__"]

    def test_register_frequency_inserts(self) -> None:
        @register_frequency("__test_frequency__")
        class _D(FrequencyDistribution):
            pass

        assert FREQUENCY_REGISTRY["__test_frequency__"] is _D
        del FREQUENCY_REGISTRY["__test_frequency__"]

    def test_register_severity_rejects_wrong_base(self) -> None:
        import pytest

        with pytest.raises(TypeError):

            @register_severity("__bad__")
            class _Bad:  # type: ignore[no-redef]
                pass


class TestNumericHelpers:
    def test_safe_log_handles_zero(self) -> None:
        out = safe_log(np.array([0.0, 1.0, np.e]))
        assert out[0] < -100
        assert abs(out[1]) < 1e-12
        assert abs(out[2] - 1.0) < 1e-12

    def test_to_from_unconstrained_roundtrip(self) -> None:
        transforms = [("alpha", "log"), ("mu", "identity"), ("p", "logit")]
        params = {"alpha": 2.5, "mu": -1.3, "p": 0.4}
        u = to_unconstrained(params, transforms)
        back = from_unconstrained(u, transforms)
        for k in params:
            assert abs(back[k] - params[k]) < 1e-12

    def test_numeric_lev_of_exponential_matches_closed_form(self) -> None:
        # S(x) = exp(-x); LEV(d) = 1 - exp(-d)
        for d in (0.5, 1.0, 3.0):
            num = numeric_lev(lambda x: float(np.exp(-x)), d)
            assert abs(num - (1.0 - np.exp(-d))) < 1e-8

    def test_numeric_ppf_inverts_uniform_cdf(self) -> None:
        # F(x) = x on [0, 1]; F^{-1}(0.3) = 0.3
        val = numeric_ppf(lambda x: x, 0.3, lower=0.0, upper=1.0)
        assert abs(val - 0.3) < 1e-9


class TestDistributionFitter:
    def test_construction(self) -> None:
        fitter = DistributionFitter(candidates=[])
        assert fitter.candidates == []
        assert fitter.results_ is None

    def test_fit_and_rank_not_yet_implemented(self) -> None:
        import pytest

        fitter = DistributionFitter(candidates=[])
        with pytest.raises(NotImplementedError):
            fitter.fit_and_rank([1.0, 2.0])


class TestGoodnessOfFit:
    def test_construction_with_none_distribution(self) -> None:
        gof = GoodnessOfFit(distribution=None, data=np.array([1.0, 2.0, 3.0]))
        assert gof.distribution is None
        assert len(gof.data) == 3

    def test_methods_pending_phase_3(self) -> None:
        import pytest

        gof = GoodnessOfFit(distribution=None, data=np.array([1.0]))
        with pytest.raises(NotImplementedError):
            gof.ks_test()
        with pytest.raises(NotImplementedError):
            gof.anderson_darling_test()


class TestMleHelpersExist:
    """The driver functions should be importable; exhaustive testing waits
    until concrete distributions can be plugged into them in Phase 1."""

    def test_callables(self) -> None:
        for f in (
            loglik_continuous,
            loglik_discrete,
            fit_continuous_mle,
            fit_discrete_mle,
        ):
            assert callable(f)
