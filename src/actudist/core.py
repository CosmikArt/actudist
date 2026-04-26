"""
Core distribution classes and fitting utilities for actudist.

This module provides the base ``ActuarialDistribution`` class, concrete
distribution stubs, a multi-candidate fitter, and goodness-of-fit testing.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ActuarialDistribution:
    """Base class for all actuarial probability distributions.

    Every distribution in *actudist* inherits from this class and must
    implement the abstract-style methods below.  The interface is intentionally
    richer than ``scipy.stats`` — it includes actuarial layer statistics
    (limited expected value, excess pure premium) and fitting helpers that
    handle censored / truncated observations.

    Parameters
    ----------
    params : dict[str, float] | None
        Named distribution parameters.  ``None`` until the distribution is
        either constructed with known parameters or fitted via
        :meth:`mle_fit`.
    """

    def __init__(self, params: dict[str, float] | None = None) -> None:
        self.params: dict[str, float] | None = params
        self._fitted: bool = False

    # -- Core statistical functions ----------------------------------------

    def pdf(self, x: ArrayLike) -> np.ndarray:
        """Probability density function (severity) or probability mass
        function (frequency) evaluated at *x*.

        Parameters
        ----------
        x : array_like
            Quantiles.

        Returns
        -------
        np.ndarray
            Density / mass values.
        """
        raise NotImplementedError

    def cdf(self, x: ArrayLike) -> np.ndarray:
        """Cumulative distribution function evaluated at *x*.

        Parameters
        ----------
        x : array_like
            Quantiles.

        Returns
        -------
        np.ndarray
            Cumulative probabilities.
        """
        raise NotImplementedError

    def ppf(self, q: ArrayLike) -> np.ndarray:
        """Percent-point (quantile) function — inverse of :meth:`cdf`.

        Parameters
        ----------
        q : array_like
            Probabilities in [0, 1].

        Returns
        -------
        np.ndarray
            Quantiles corresponding to *q*.
        """
        raise NotImplementedError

    def rvs(self, size: int = 1, random_state: int | None = None) -> np.ndarray:
        """Generate random variates.

        Parameters
        ----------
        size : int
            Number of variates to draw.
        random_state : int | None
            Seed for reproducibility.

        Returns
        -------
        np.ndarray
            Random sample of length *size*.
        """
        raise NotImplementedError

    # -- Fitting -----------------------------------------------------------

    def mle_fit(
        self,
        data: ArrayLike,
        *,
        censored: ArrayLike | None = None,
        truncation_lower: float | None = None,
        truncation_upper: float | None = None,
    ) -> dict[str, float]:
        """Fit the distribution to *data* via maximum likelihood estimation.

        Parameters
        ----------
        data : array_like
            Observed loss amounts (severity) or claim counts (frequency).
        censored : array_like | None
            Boolean array of the same length as *data*.  ``True`` indicates
            the observation is right-censored at the reported value (e.g.,
            policy limit reached).
        truncation_lower : float | None
            Left-truncation point (e.g., deductible).  Observations below
            this threshold were never recorded.
        truncation_upper : float | None
            Right-truncation point, if applicable.

        Returns
        -------
        dict[str, float]
            Fitted parameter estimates stored in :attr:`params`.
        """
        raise NotImplementedError

    # -- Likelihood & information criteria ---------------------------------

    def loglik(self, data: ArrayLike) -> float:
        """Log-likelihood of the current parameters given *data*.

        Parameters
        ----------
        data : array_like
            Observed values.

        Returns
        -------
        float
            Total log-likelihood.
        """
        raise NotImplementedError

    def aic(self, data: ArrayLike) -> float:
        """Akaike Information Criterion.

        .. math::
            \\mathrm{AIC} = -2\\,\\ell(\\hat\\theta) + 2\\,k

        Parameters
        ----------
        data : array_like
            Observed values used to compute the log-likelihood.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def bic(self, data: ArrayLike) -> float:
        """Bayesian Information Criterion.

        .. math::
            \\mathrm{BIC} = -2\\,\\ell(\\hat\\theta) + k\\,\\ln(n)

        Parameters
        ----------
        data : array_like
            Observed values.

        Returns
        -------
        float
        """
        raise NotImplementedError

    # -- Actuarial layer statistics ----------------------------------------

    def limited_expected_value(self, limit: float) -> float:
        """Limited expected value (LEV) at *limit*.

        .. math::
            \\operatorname{LEV}(d) = E[\\min(X, d)]
            = \\int_0^d S(x)\\,dx

        where *S* is the survival function.

        Parameters
        ----------
        limit : float
            Policy limit or layer attachment point.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def excess_pure_premium(
        self,
        attachment: float,
        limit: float | None = None,
    ) -> float:
        """Excess pure premium for a layer.

        For an excess-of-loss layer attaching at *attachment* with width
        *limit* (``None`` for unlimited):

        .. math::
            \\operatorname{EPP}(a, a+l) = \\operatorname{LEV}(a+l)
            - \\operatorname{LEV}(a)

        Parameters
        ----------
        attachment : float
            Layer attachment point (deductible).
        limit : float | None
            Layer width.  ``None`` means unlimited cover above *attachment*.

        Returns
        -------
        float
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Severity distributions
# ---------------------------------------------------------------------------

class BurrXII(ActuarialDistribution):
    """Burr Type XII (Singh-Maddala) distribution.

    A three-parameter heavy-tail distribution widely used for modeling
    insurance claim severities.  Nests the Pareto, Loglogistic, and
    Paralogistic as special / limiting cases.

    **Parameterization** (Klugman, Panjer & Willmot convention)::

        f(x) = (alpha * gamma / theta) * (x / theta)^(gamma - 1)
               / (1 + (x / theta)^gamma)^(alpha + 1),   x > 0

    Parameters
    ----------
    alpha : float
        Shape parameter (tail weight), alpha > 0.
    gamma : float
        Shape parameter, gamma > 0.
    theta : float
        Scale parameter, theta > 0.
    """

    def __init__(
        self,
        alpha: float | None = None,
        gamma: float | None = None,
        theta: float | None = None,
    ) -> None:
        params = None
        if alpha is not None and gamma is not None and theta is not None:
            params = {"alpha": alpha, "gamma": gamma, "theta": theta}
        super().__init__(params=params)

    def pdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def ppf(self, q: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def rvs(self, size: int = 1, random_state: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def mle_fit(self, data: ArrayLike, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError

    def loglik(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def aic(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def bic(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def limited_expected_value(self, limit: float) -> float:
        raise NotImplementedError

    def excess_pure_premium(
        self, attachment: float, limit: float | None = None
    ) -> float:
        raise NotImplementedError


class GeneralizedPareto(ActuarialDistribution):
    """Generalized Pareto Distribution (GPD).

    Used in extreme value theory for modeling exceedances over a high
    threshold.  Central to peaks-over-threshold (POT) analysis of
    catastrophe losses.

    **Parameterization**::

        F(x) = 1 - (1 + xi * x / sigma)^(-1/xi),   x >= 0

    where *xi* is the shape (tail index) and *sigma* is the scale.  When
    *xi > 0* the distribution is heavy-tailed (Pareto-like); *xi = 0*
    gives the exponential; *xi < 0* gives a bounded support.

    Parameters
    ----------
    xi : float
        Shape (tail index).
    sigma : float
        Scale, sigma > 0.
    """

    def __init__(
        self,
        xi: float | None = None,
        sigma: float | None = None,
    ) -> None:
        params = None
        if xi is not None and sigma is not None:
            params = {"xi": xi, "sigma": sigma}
        super().__init__(params=params)

    def pdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def ppf(self, q: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def rvs(self, size: int = 1, random_state: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def mle_fit(self, data: ArrayLike, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError

    def loglik(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def aic(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def bic(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def limited_expected_value(self, limit: float) -> float:
        raise NotImplementedError

    def excess_pure_premium(
        self, attachment: float, limit: float | None = None
    ) -> float:
        raise NotImplementedError


class TransformedBeta(ActuarialDistribution):
    """Transformed Beta distribution (4-parameter).

    The transformed beta family is a flexible four-parameter system that
    nests the Burr XII, Inverse Burr, Paralogistic, Inverse Paralogistic,
    Loglogistic, and Pareto distributions as special cases.

    **Parameterization** (Klugman, Panjer & Willmot)::

        f(x) = (gamma / (x * B(alpha, tau)))
               * ((x/theta)^(gamma*tau))
               / (1 + (x/theta)^gamma)^(alpha + tau),   x > 0

    where B(alpha, tau) is the beta function.

    Parameters
    ----------
    alpha : float
        Shape parameter, alpha > 0.
    tau : float
        Shape parameter, tau > 0.
    gamma : float
        Shape parameter, gamma > 0.
    theta : float
        Scale parameter, theta > 0.
    """

    def __init__(
        self,
        alpha: float | None = None,
        tau: float | None = None,
        gamma: float | None = None,
        theta: float | None = None,
    ) -> None:
        params = None
        if all(p is not None for p in (alpha, tau, gamma, theta)):
            params = {
                "alpha": alpha,
                "tau": tau,
                "gamma": gamma,
                "theta": theta,
            }
        super().__init__(params=params)

    def pdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def ppf(self, q: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def rvs(self, size: int = 1, random_state: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def mle_fit(self, data: ArrayLike, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError

    def loglik(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def aic(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def bic(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def limited_expected_value(self, limit: float) -> float:
        raise NotImplementedError

    def excess_pure_premium(
        self, attachment: float, limit: float | None = None
    ) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Frequency distributions
# ---------------------------------------------------------------------------

class ZeroInflatedPoisson(ActuarialDistribution):
    """Zero-Inflated Poisson (ZIP) distribution.

    Models count data with excess zeros — common in insurance frequency
    data where a large proportion of policies produce no claims.

    **Parameterization**::

        P(X=0) = pi + (1 - pi) * exp(-lambda)
        P(X=k) = (1 - pi) * exp(-lambda) * lambda^k / k!,   k >= 1

    Parameters
    ----------
    lam : float
        Poisson rate parameter, lambda > 0.
    pi : float
        Zero-inflation probability, 0 <= pi < 1.
    """

    def __init__(
        self,
        lam: float | None = None,
        pi: float | None = None,
    ) -> None:
        params = None
        if lam is not None and pi is not None:
            params = {"lam": lam, "pi": pi}
        super().__init__(params=params)

    def pdf(self, x: ArrayLike) -> np.ndarray:
        """Probability mass function."""
        raise NotImplementedError

    def cdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def ppf(self, q: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def rvs(self, size: int = 1, random_state: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def mle_fit(self, data: ArrayLike, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError

    def loglik(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def aic(self, data: ArrayLike) -> float:
        raise NotImplementedError

    def bic(self, data: ArrayLike) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fitting utilities
# ---------------------------------------------------------------------------

class DistributionFitter:
    """Fit multiple candidate distributions and rank by information criteria.

    Given a list of distribution names (or instances), ``DistributionFitter``
    fits each one to the supplied data via MLE and produces a ranking table
    sorted by AIC, BIC, or another criterion.

    Parameters
    ----------
    candidates : Sequence[str | ActuarialDistribution]
        Distribution names (e.g., ``"BurrXII"``, ``"Lognormal"``) or
        pre-configured ``ActuarialDistribution`` instances.

    Examples
    --------
    >>> fitter = DistributionFitter(["BurrXII", "Lognormal", "Pareto"])
    >>> rankings = fitter.fit_and_rank(data, criterion="aic")
    """

    def __init__(
        self,
        candidates: Sequence[str | ActuarialDistribution],
    ) -> None:
        self.candidates = list(candidates)
        self.results_: list[dict[str, Any]] | None = None

    def fit_and_rank(
        self,
        data: ArrayLike,
        *,
        criterion: str = "aic",
        **fit_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fit all candidates and return a ranking.

        Parameters
        ----------
        data : array_like
            Observed data.
        criterion : str
            ``"aic"`` or ``"bic"``.
        **fit_kwargs
            Keyword arguments forwarded to each distribution's
            :meth:`~ActuarialDistribution.mle_fit`.

        Returns
        -------
        list[dict]
            Sorted list of dicts with keys ``distribution``, ``params``,
            ``aic``, ``bic``, ``loglik``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Goodness-of-fit
# ---------------------------------------------------------------------------

class GoodnessOfFit:
    """Goodness-of-fit testing suite for a fitted distribution.

    Provides Kolmogorov-Smirnov, Anderson-Darling, and Chi-squared tests
    as well as PP and QQ diagnostic plots.

    Parameters
    ----------
    distribution : ActuarialDistribution
        A fitted distribution (i.e., :attr:`params` is not ``None``).
    data : array_like
        The data the distribution was fitted to.
    """

    def __init__(
        self,
        distribution: ActuarialDistribution,
        data: ArrayLike,
    ) -> None:
        self.distribution = distribution
        self.data = np.asarray(data)

    def ks_test(self) -> dict[str, float]:
        """Kolmogorov-Smirnov test.

        Returns
        -------
        dict
            ``{"statistic": float, "p_value": float}``.
        """
        raise NotImplementedError

    def anderson_darling_test(self) -> dict[str, float]:
        """Anderson-Darling test.

        Returns
        -------
        dict
            ``{"statistic": float, "critical_values": list, "p_value": float}``.
        """
        raise NotImplementedError

    def chi_squared_test(self, n_bins: int = 10) -> dict[str, float]:
        """Chi-squared goodness-of-fit test.

        Parameters
        ----------
        n_bins : int
            Number of equiprobable bins.

        Returns
        -------
        dict
            ``{"statistic": float, "p_value": float, "df": int}``.
        """
        raise NotImplementedError

    def pp_plot(self, ax: Any | None = None) -> Any:
        """Probability-probability plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Axes to plot on.  If ``None``, a new figure is created.

        Returns
        -------
        matplotlib.axes.Axes
        """
        raise NotImplementedError

    def qq_plot(self, ax: Any | None = None) -> Any:
        """Quantile-quantile plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Axes to plot on.  If ``None``, a new figure is created.

        Returns
        -------
        matplotlib.axes.Axes
        """
        raise NotImplementedError
