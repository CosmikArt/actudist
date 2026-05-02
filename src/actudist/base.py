"""Base classes for actuarial distributions.

:class:`ActuarialDistribution` is the abstract root. It holds the parameter
dict, defines ``cdf`` / ``ppf`` / ``rvs`` / ``loglik`` as abstract, and
implements AIC, BIC, and a profile-likelihood CI on top of them.

:class:`SeverityDistribution` is continuous on the non-negative reals. It
adds ``pdf``, survival function, hazard rate, layer statistics (LEV, EPP,
ILF), and an MLE driver that accepts a right-censoring mask and one- or
two-sided truncation bounds.

:class:`FrequencyDistribution` is discrete on the non-negative integers
and deliberately omits layer statistics.

Concrete subclasses declare ``_transforms()`` (parameter to log / identity /
logit) so the MLE driver in :mod:`actudist._mle` can optimize in
unconstrained space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class ActuarialDistribution:
    """Abstract root.

    Subclasses implement :meth:`cdf`, :meth:`ppf`, :meth:`rvs`, and
    :meth:`loglik`. AIC and BIC follow from :meth:`loglik` and
    :attr:`n_params`.
    """

    #: Number of free parameters. Subclasses override.
    n_params: int = 0

    def __init__(self, params: dict[str, float] | None = None) -> None:
        self.params: dict[str, float] | None = params
        self._fitted: bool = False
        if params is not None:
            for name, value in params.items():
                setattr(self, name, value)

    # -- Statistical functions (must be overridden) ------------------------

    def cdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def ppf(self, q: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def rvs(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        raise NotImplementedError

    def loglik(self, data: ArrayLike, **kwargs: Any) -> float:
        raise NotImplementedError

    # -- Information criteria ---------------------------------------------

    def aic(self, data: ArrayLike, **kwargs: Any) -> float:
        r""":math:`\mathrm{AIC} = -2\,\ell(\hat\theta) + 2k`."""
        return -2.0 * self.loglik(data, **kwargs) + 2.0 * self.n_params

    def bic(self, data: ArrayLike, **kwargs: Any) -> float:
        r""":math:`\mathrm{BIC} = -2\,\ell(\hat\theta) + k\ln n`."""
        n = int(np.asarray(data).size)
        return -2.0 * self.loglik(data, **kwargs) + self.n_params * np.log(n)

    def profile_likelihood_ci(
        self,
        data: ArrayLike,
        param: str,
        *,
        alpha: float = 0.05,
        **fit_kwargs: Any,
    ) -> tuple[float, float]:
        """Profile-likelihood :math:`(1-\\alpha)` CI for ``param``.

        Requires that ``self`` is already MLE-fitted. The other parameters
        are re-optimized at each grid point of ``param``.
        """
        from actudist._mle import profile_likelihood_ci as _impl

        return _impl(
            self,
            np.asarray(data, dtype=float),
            param,
            alpha=alpha,
            fit_kwargs=fit_kwargs,
        )

    # -- Required hooks for _mle driver -----------------------------------

    @classmethod
    def _transforms(cls) -> list[tuple[str, str]]:
        """Return ``[(param_name, transform), ...]`` with ``transform`` in
        ``{"log", "identity", "logit"}``."""
        raise NotImplementedError

    @classmethod
    def _initial_guess(cls, data: ArrayLike) -> dict[str, float]:
        """Initial guess for MLE; defaults to ``1.0`` per parameter."""
        return {name: 1.0 for name, _ in cls._transforms()}


class SeverityDistribution(ActuarialDistribution):
    """Continuous distribution on :math:`x \\ge 0`. Adds pdf, survival
    function, hazard rate, and the actuarial layer statistics."""

    def pdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    # -- Derived statistical functions ------------------------------------

    def survival_function(self, x: ArrayLike) -> np.ndarray:
        return 1.0 - np.asarray(self.cdf(x))

    def hazard_rate(self, x: ArrayLike) -> np.ndarray:
        s = self.survival_function(x)
        return np.asarray(self.pdf(x)) / np.maximum(s, 1e-300)

    # -- Moments (override per distribution) -------------------------------

    def mean(self) -> float:
        raise NotImplementedError

    # -- Layer statistics --------------------------------------------------

    def limited_expected_value(self, d: float) -> float:
        r"""Returns :math:`E[\min(X, d)] = \int_0^d S(x)\,dx`. Default is
        quadrature; subclasses override with closed forms."""
        from actudist._numerics import numeric_lev

        return numeric_lev(lambda x: float(self.survival_function(x)), float(d))

    def excess_pure_premium(self, d: float) -> float:
        r"""Returns :math:`E[X] - \mathrm{LEV}(d)`, the pure premium of an
        unlimited excess layer attaching at *d*."""
        return float(self.mean()) - self.limited_expected_value(d)

    def increased_limits_factor(self, d: float, base_d: float) -> float:
        r""":math:`\mathrm{LEV}(d) / \mathrm{LEV}(\text{base\_d})`."""
        base = self.limited_expected_value(base_d)
        if base == 0:
            raise ValueError("Base LEV is zero; ILF undefined.")
        return self.limited_expected_value(d) / base

    # -- Likelihood + fitting ---------------------------------------------

    def loglik(
        self,
        data: ArrayLike,
        *,
        censored: ArrayLike | None = None,
        trunc_lower: float | None = None,
        trunc_upper: float | None = None,
    ) -> float:
        """Log-likelihood with right-censoring and truncation hooks; see
        :func:`actudist._mle.loglik_continuous` for the formula."""
        from actudist._mle import loglik_continuous

        return loglik_continuous(
            self,
            np.asarray(data, dtype=float),
            censored=censored,
            trunc_lower=trunc_lower,
            trunc_upper=trunc_upper,
        )

    def mle_fit(
        self,
        data: ArrayLike,
        *,
        censored: ArrayLike | None = None,
        trunc_lower: float | None = None,
        trunc_upper: float | None = None,
    ) -> dict[str, float]:
        """Fit by MLE; store fitted parameters on ``self``."""
        from actudist._mle import fit_continuous_mle

        params = fit_continuous_mle(
            type(self),
            np.asarray(data, dtype=float),
            censored=censored,
            trunc_lower=trunc_lower,
            trunc_upper=trunc_upper,
        )
        self.params = params
        self._fitted = True
        for name, value in params.items():
            setattr(self, name, value)
        return params


class FrequencyDistribution(ActuarialDistribution):
    """Discrete distribution on :math:`k \\in \\{0, 1, 2, \\dots\\}`. No
    layer statistics: LEV and EPP are severity concepts."""

    def pmf(self, k: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def loglik(self, data: ArrayLike, **kwargs: Any) -> float:
        from actudist._mle import loglik_discrete

        return loglik_discrete(self, np.asarray(data))

    def mle_fit(self, data: ArrayLike) -> dict[str, float]:
        from actudist._mle import fit_discrete_mle

        params = fit_discrete_mle(type(self), np.asarray(data))
        self.params = params
        self._fitted = True
        for name, value in params.items():
            setattr(self, name, value)
        return params
