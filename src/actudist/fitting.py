"""Distribution registry and multi-candidate MLE fitter.

Concrete distribution classes register themselves at import time with
:func:`register_severity` or :func:`register_frequency`, populating the
``SEVERITY_REGISTRY`` and ``FREQUENCY_REGISTRY`` dicts. The
:class:`DistributionFitter` resolves candidate names against these
registries and returns a ranking by AIC/BIC.

The ranking and goodness-of-fit columns are filled in once Phase 5 of the
roadmap is reached; the registry mechanics are needed earlier so each
Phase 1 / Phase 2 distribution can self-register on import.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from actudist.base import (
    ActuarialDistribution,
    FrequencyDistribution,
    SeverityDistribution,
)


SEVERITY_REGISTRY: dict[str, type[SeverityDistribution]] = {}
FREQUENCY_REGISTRY: dict[str, type[FrequencyDistribution]] = {}


def register_severity(name: str) -> Callable[[type], type]:
    """Class decorator that adds a severity distribution to the registry
    under ``name``."""

    def deco(cls: type) -> type:
        if not issubclass(cls, SeverityDistribution):
            raise TypeError(
                f"{cls.__name__} must inherit from SeverityDistribution to "
                f"register as a severity distribution."
            )
        SEVERITY_REGISTRY[name] = cls
        return cls

    return deco


def register_frequency(name: str) -> Callable[[type], type]:
    """Class decorator that adds a frequency distribution to the registry
    under ``name``."""

    def deco(cls: type) -> type:
        if not issubclass(cls, FrequencyDistribution):
            raise TypeError(
                f"{cls.__name__} must inherit from FrequencyDistribution to "
                f"register as a frequency distribution."
            )
        FREQUENCY_REGISTRY[name] = cls
        return cls

    return deco


def _resolve(name_or_instance: str | ActuarialDistribution) -> ActuarialDistribution:
    """Resolve a candidate to an instance. Strings are looked up in both
    registries; instances pass through."""
    if isinstance(name_or_instance, ActuarialDistribution):
        return name_or_instance
    name = name_or_instance
    if name in SEVERITY_REGISTRY:
        return SEVERITY_REGISTRY[name]()
    if name in FREQUENCY_REGISTRY:
        return FREQUENCY_REGISTRY[name]()
    raise KeyError(
        f"{name!r} is not a registered distribution. "
        f"Known severity: {sorted(SEVERITY_REGISTRY)!r}; "
        f"frequency: {sorted(FREQUENCY_REGISTRY)!r}."
    )


class DistributionFitter:
    """Fit several candidate distributions to data and rank by an
    information criterion (AIC or BIC). Each candidate is named (string
    looked up in the registry) or a pre-built ``ActuarialDistribution``
    instance.
    """

    def __init__(
        self,
        candidates: Sequence[str | ActuarialDistribution],
    ) -> None:
        self.candidates = list(candidates)
        self.results_: list[dict[str, Any]] | None = None

    def fit_and_rank(
        self,
        data: Any,
        *,
        criterion: str = "aic",
        **fit_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fit every candidate to ``data`` and return rows sorted by
        ``criterion`` (``"aic"`` or ``"bic"``), ascending. Failed fits
        are recorded with ``error`` and pushed to the bottom."""
        if criterion not in ("aic", "bic"):
            raise ValueError(f"criterion must be 'aic' or 'bic'; got {criterion!r}")

        import numpy as np

        rows: list[dict[str, Any]] = []
        for cand in self.candidates:
            inst = _resolve(cand)
            name = (
                cand if isinstance(cand, str) else type(inst).__name__
            )
            row: dict[str, Any] = {"name": name, "distribution": inst}
            try:
                inst.mle_fit(data, **fit_kwargs)
                ll = float(inst.loglik(data, **fit_kwargs))
                row["loglik"] = ll
                row["k"] = int(getattr(inst, "n_params", 0))
                row["aic"] = float(inst.aic(data, **fit_kwargs))
                row["bic"] = float(inst.bic(data, **fit_kwargs))
                row["params"] = dict(inst.params or {})
                row["error"] = None
            except Exception as exc:  # noqa: BLE001 — surface fit failure
                row["loglik"] = float("nan")
                row["aic"] = float("inf")
                row["bic"] = float("inf")
                row["params"] = None
                row["error"] = repr(exc)
            rows.append(row)

        rows.sort(key=lambda r: (np.isnan(r["loglik"]), r[criterion]))
        self.results_ = rows
        return rows
