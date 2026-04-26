"""Smoke tests for actudist — import and basic instantiation."""

import actudist
from actudist.core import (
    ActuarialDistribution,
    BurrXII,
    DistributionFitter,
    GeneralizedPareto,
    GoodnessOfFit,
    TransformedBeta,
    ZeroInflatedPoisson,
)


class TestImports:
    """Verify that the package and its public API are importable."""

    def test_version_exists(self) -> None:
        assert hasattr(actudist, "__version__")
        assert isinstance(actudist.__version__, str)

    def test_top_level_imports(self) -> None:
        assert actudist.ActuarialDistribution is ActuarialDistribution
        assert actudist.BurrXII is BurrXII
        assert actudist.GeneralizedPareto is GeneralizedPareto
        assert actudist.TransformedBeta is TransformedBeta
        assert actudist.ZeroInflatedPoisson is ZeroInflatedPoisson
        assert actudist.DistributionFitter is DistributionFitter
        assert actudist.GoodnessOfFit is GoodnessOfFit


class TestInstantiation:
    """Verify that core classes can be instantiated without errors."""

    def test_base_distribution(self) -> None:
        dist = ActuarialDistribution()
        assert dist.params is None
        assert dist._fitted is False

    def test_burr_xii_no_params(self) -> None:
        dist = BurrXII()
        assert dist.params is None

    def test_burr_xii_with_params(self) -> None:
        dist = BurrXII(alpha=2.0, gamma=1.5, theta=1000.0)
        assert dist.params == {"alpha": 2.0, "gamma": 1.5, "theta": 1000.0}

    def test_generalized_pareto_no_params(self) -> None:
        dist = GeneralizedPareto()
        assert dist.params is None

    def test_generalized_pareto_with_params(self) -> None:
        dist = GeneralizedPareto(xi=0.5, sigma=100.0)
        assert dist.params == {"xi": 0.5, "sigma": 100.0}

    def test_transformed_beta_no_params(self) -> None:
        dist = TransformedBeta()
        assert dist.params is None

    def test_transformed_beta_with_params(self) -> None:
        dist = TransformedBeta(alpha=2.0, tau=3.0, gamma=1.0, theta=500.0)
        assert dist.params == {
            "alpha": 2.0,
            "tau": 3.0,
            "gamma": 1.0,
            "theta": 500.0,
        }

    def test_zero_inflated_poisson_no_params(self) -> None:
        dist = ZeroInflatedPoisson()
        assert dist.params is None

    def test_zero_inflated_poisson_with_params(self) -> None:
        dist = ZeroInflatedPoisson(lam=3.0, pi=0.2)
        assert dist.params == {"lam": 3.0, "pi": 0.2}

    def test_distribution_fitter(self) -> None:
        fitter = DistributionFitter(candidates=["BurrXII", "Lognormal"])
        assert len(fitter.candidates) == 2
        assert fitter.results_ is None

    def test_goodness_of_fit(self) -> None:
        import numpy as np

        dist = BurrXII(alpha=2.0, gamma=1.5, theta=1000.0)
        data = np.array([100.0, 200.0, 500.0])
        gof = GoodnessOfFit(distribution=dist, data=data)
        assert gof.distribution is dist
        assert len(gof.data) == 3
