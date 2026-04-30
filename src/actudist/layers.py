"""Layer statistics helpers (excess pure premium, increased limits factors,
layer expected value).

Each :class:`actudist.base.SeverityDistribution` carries its own
closed-form ``limited_expected_value``; this module wraps it in a small
set of distribution-agnostic helpers for use in pricing notebooks.

Phase 0 ships function signatures only; closed-form layer formulas land
once their underlying severity distributions exist (Phase 1) and the
public helpers are tested in Phase 4.
"""

from __future__ import annotations

from actudist.base import SeverityDistribution


def excess_pure_premium(dist: SeverityDistribution, attachment: float) -> float:
    r""":math:`E[X] - \mathrm{LEV}(d)`. Pure premium of an unlimited excess
    layer attaching at *attachment*."""
    return dist.excess_pure_premium(attachment)


def increased_limits_factor(
    dist: SeverityDistribution, d: float, base_d: float
) -> float:
    r""":math:`\mathrm{LEV}(d) / \mathrm{LEV}(\text{base\_d})`."""
    return dist.increased_limits_factor(d, base_d)


def layer_expected_value(
    dist: SeverityDistribution, lower: float, upper: float
) -> float:
    r""":math:`\mathrm{LEV}(\text{upper}) - \mathrm{LEV}(\text{lower})`. The
    expected loss falling into a finite layer ``[lower, upper]``."""
    if upper <= lower:
        raise ValueError("upper must exceed lower for a non-degenerate layer")
    return dist.limited_expected_value(upper) - dist.limited_expected_value(lower)
