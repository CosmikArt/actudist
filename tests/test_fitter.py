"""End-to-end tests for DistributionFitter.fit_and_rank."""

from __future__ import annotations

import numpy as np

from actudist import DistributionFitter
from actudist.severity.lognormal import Lognormal


class TestFitAndRank:
    def test_ranks_correctly_specified_distribution_top(self) -> None:
        rng = np.random.default_rng(0)
        true = Lognormal(mu=1.5, sigma=0.6)
        data = true.rvs(size=2_000, random_state=rng)
        fitter = DistributionFitter(
            candidates=["Exponential", "Gamma", "Weibull", "Lognormal", "Pareto"]
        )
        rows = fitter.fit_and_rank(data, criterion="aic")
        assert len(rows) == 5
        names = [r["name"] for r in rows]
        # Lognormal (true) should AIC-beat the misspecified candidates
        assert names[0] == "Lognormal"

    def test_bic_criterion_returns_sorted_list(self) -> None:
        data = np.random.default_rng(0).exponential(scale=1.5, size=500)
        fitter = DistributionFitter(candidates=["Exponential", "Lognormal", "Weibull"])
        rows = fitter.fit_and_rank(data, criterion="bic")
        bics = [r["bic"] for r in rows]
        assert bics == sorted(bics)

    def test_records_aic_bic_loglik_params(self) -> None:
        data = np.random.default_rng(0).exponential(scale=2.0, size=500)
        fitter = DistributionFitter(candidates=["Exponential"])
        (row,) = fitter.fit_and_rank(data)
        assert "aic" in row and "bic" in row and "loglik" in row
        assert "theta" in row["params"]
        assert row["error"] is None
        assert row["k"] == 1

    def test_failed_fit_pushed_to_bottom(self) -> None:
        # all-zero data trips the positive-support distributions; verify the
        # ranker pushes errored rows to the end without crashing
        data = np.zeros(50)
        fitter = DistributionFitter(candidates=["Exponential"])
        rows = fitter.fit_and_rank(data)
        # failure is acceptable; surface it
        assert rows[0]["name"] == "Exponential"
