"""Goodness-of-fit tests and diagnostic plots.

Kolmogorov-Smirnov, upper-tail emphasized Anderson-Darling, and chi-squared,
plus PP and QQ plot helpers. KS and AD p-values come from a parametric
bootstrap so the Lilliefors correction is automatic when parameters were
estimated from the same sample.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from actudist.base import ActuarialDistribution


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _ks_statistic(sorted_data: np.ndarray, cdf: np.ndarray) -> float:
    n = sorted_data.size
    i = np.arange(1, n + 1, dtype=float)
    d_plus = np.max(i / n - cdf)
    d_minus = np.max(cdf - (i - 1.0) / n)
    return float(max(d_plus, d_minus))


def _anderson_darling_statistic(sorted_data: np.ndarray, cdf: np.ndarray) -> float:
    n = sorted_data.size
    eps = 1e-15
    cdf = np.clip(cdf, eps, 1.0 - eps)
    i = np.arange(1, n + 1, dtype=float)
    log_f = np.log(cdf)
    log_s = np.log(1.0 - cdf[::-1])
    return float(-n - np.sum((2.0 * i - 1.0) / n * (log_f + log_s)))


# ---------------------------------------------------------------------------
# GoodnessOfFit
# ---------------------------------------------------------------------------


class GoodnessOfFit:
    """GoF wrapper for a fitted distribution. Exposes KS and AD tests
    (parametric-bootstrap p-values), chi-squared, and PP/QQ plots."""

    def __init__(
        self,
        distribution: ActuarialDistribution | None,
        data: ArrayLike,
    ) -> None:
        self.distribution = distribution
        self.data = np.asarray(data, dtype=float)

    # -- private bootstrap helper -----------------------------------------

    def _bootstrap_pvalue(
        self,
        observed_stat: float,
        statistic: str,
        n_boot: int,
        random_state: int | np.random.Generator | None,
    ) -> float:
        if self.distribution is None:
            raise ValueError("distribution is required for bootstrap p-values")
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        n = self.data.size
        cls = type(self.distribution)
        params = dict(self.distribution.params or {})

        ge = 0
        for _ in range(n_boot):
            sample = self.distribution.rvs(size=n, random_state=rng)
            try:
                inst = cls(**params)
                inst.mle_fit(sample)
            except Exception:
                continue
            sorted_s = np.sort(np.asarray(sample, dtype=float))
            cdf_s = np.asarray(inst.cdf(sorted_s), dtype=float)
            if statistic == "ks":
                stat = _ks_statistic(sorted_s, cdf_s)
            elif statistic == "ad":
                stat = _anderson_darling_statistic(sorted_s, cdf_s)
            else:
                raise ValueError(statistic)
            if stat >= observed_stat:
                ge += 1
        return (ge + 1) / (n_boot + 1)

    # -- public tests ------------------------------------------------------

    def ks_test(
        self,
        *,
        n_boot: int = 1000,
        random_state: int | np.random.Generator | None = None,
    ) -> dict[str, float]:
        if self.distribution is None:
            raise ValueError("KS test requires a fitted distribution")
        sorted_data = np.sort(self.data)
        cdf_vals = np.asarray(self.distribution.cdf(sorted_data), dtype=float)
        stat = _ks_statistic(sorted_data, cdf_vals)
        p_value = self._bootstrap_pvalue(stat, "ks", n_boot, random_state)
        return {"statistic": stat, "p_value": p_value, "n_boot": float(n_boot)}

    def anderson_darling_test(
        self,
        *,
        n_boot: int = 1000,
        random_state: int | np.random.Generator | None = None,
    ) -> dict[str, Any]:
        if self.distribution is None:
            raise ValueError("AD test requires a fitted distribution")
        sorted_data = np.sort(self.data)
        cdf_vals = np.asarray(self.distribution.cdf(sorted_data), dtype=float)
        stat = _anderson_darling_statistic(sorted_data, cdf_vals)
        p_value = self._bootstrap_pvalue(stat, "ad", n_boot, random_state)
        return {"statistic": stat, "p_value": p_value, "n_boot": float(n_boot)}

    def chi_squared_test(self, n_bins: int = 10) -> dict[str, float]:
        if self.distribution is None:
            raise ValueError("Chi-squared test requires a fitted distribution")
        from scipy.stats import chi2

        n = self.data.size
        # equiprobable bin edges from the fitted distribution
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.asarray(self.distribution.ppf(qs[1:-1]), dtype=float)
        edges = np.concatenate([[-np.inf], edges, [np.inf]])
        observed, _ = np.histogram(self.data, bins=edges)
        expected = n / n_bins
        stat = float(np.sum((observed - expected) ** 2 / expected))
        df = max(n_bins - 1 - getattr(self.distribution, "n_params", 0), 1)
        p_value = float(1.0 - chi2.cdf(stat, df=df))
        return {
            "statistic": stat,
            "p_value": p_value,
            "df": float(df),
            "n_bins": float(n_bins),
        }

    # -- diagnostic plots --------------------------------------------------

    def pp_plot(self, ax: Any | None = None) -> Any:
        if self.distribution is None:
            raise ValueError("PP plot requires a fitted distribution")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        sorted_data = np.sort(self.data)
        n = sorted_data.size
        emp = (np.arange(1, n + 1) - 0.5) / n
        theo = np.asarray(self.distribution.cdf(sorted_data), dtype=float)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.scatter(theo, emp, s=10)
        ax.set_xlabel("Theoretical CDF")
        ax.set_ylabel("Empirical CDF")
        ax.set_title("PP plot")
        return ax

    def qq_plot(self, ax: Any | None = None) -> Any:
        if self.distribution is None:
            raise ValueError("QQ plot requires a fitted distribution")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        sorted_data = np.sort(self.data)
        n = sorted_data.size
        plot_pos = (np.arange(1, n + 1) - 0.5) / n
        theo = np.asarray(self.distribution.ppf(plot_pos), dtype=float)
        finite = np.isfinite(theo) & np.isfinite(sorted_data)
        ax.scatter(theo[finite], sorted_data[finite], s=10)
        if np.any(finite):
            lo = float(min(theo[finite].min(), sorted_data[finite].min()))
            hi = float(max(theo[finite].max(), sorted_data[finite].max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Theoretical quantile")
        ax.set_ylabel("Empirical quantile")
        ax.set_title("QQ plot")
        return ax
