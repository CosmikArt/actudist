"""Tests for goodness-of-fit (KS, Anderson-Darling, chi-squared, PP/QQ)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from actudist import GoodnessOfFit
from actudist.severity.exponential import Exponential
from actudist.severity.lognormal import Lognormal


class TestStatistics:
    def test_ks_passes_for_correctly_specified_model(self) -> None:
        rng = np.random.default_rng(0)
        true = Exponential(theta=2.0)
        data = true.rvs(size=600, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        gof = GoodnessOfFit(distribution=fit, data=data)
        out = gof.ks_test(n_boot=200, random_state=rng)
        assert 0.0 <= out["p_value"] <= 1.0
        # the model is correctly specified ⇒ p should not be tiny
        assert out["p_value"] > 0.05

    def test_ks_rejects_misspecified_model(self) -> None:
        rng = np.random.default_rng(0)
        # Lognormal data, claim it is Exponential ⇒ misspecified
        data = Lognormal(mu=0.0, sigma=1.0).rvs(size=400, random_state=rng)
        wrong = Exponential()
        wrong.mle_fit(data)
        gof = GoodnessOfFit(distribution=wrong, data=data)
        out = gof.ks_test(n_boot=200, random_state=rng)
        assert out["p_value"] < 0.05

    def test_anderson_darling_runs(self) -> None:
        rng = np.random.default_rng(0)
        true = Exponential(theta=2.0)
        data = true.rvs(size=400, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        gof = GoodnessOfFit(distribution=fit, data=data)
        out = gof.anderson_darling_test(n_boot=200, random_state=rng)
        assert out["statistic"] >= 0
        assert 0.0 <= out["p_value"] <= 1.0

    def test_chi_squared_runs(self) -> None:
        rng = np.random.default_rng(0)
        true = Exponential(theta=2.0)
        data = true.rvs(size=2000, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        gof = GoodnessOfFit(distribution=fit, data=data)
        out = gof.chi_squared_test(n_bins=10)
        assert out["statistic"] >= 0
        assert 0.0 <= out["p_value"] <= 1.0
        assert out["df"] == 8  # 10 - 1 - 1


class TestPlots:
    def test_pp_plot_returns_axes(self) -> None:
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        true = Exponential(theta=2.0)
        data = true.rvs(size=200, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        gof = GoodnessOfFit(distribution=fit, data=data)
        ax = gof.pp_plot()
        assert ax is not None
        plt.close("all")

    def test_qq_plot_returns_axes(self) -> None:
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        true = Exponential(theta=2.0)
        data = true.rvs(size=200, random_state=rng)
        fit = Exponential()
        fit.mle_fit(data)
        gof = GoodnessOfFit(distribution=fit, data=data)
        ax = gof.qq_plot()
        assert ax is not None
        plt.close("all")
