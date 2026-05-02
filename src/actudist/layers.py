"""Layer-statistic helpers.

Thin wrappers around the methods on :class:`actudist.base.SeverityDistribution`,
in a shape convenient for pricing notebooks.
"""

from __future__ import annotations

from actudist.base import SeverityDistribution


def excess_pure_premium(dist: SeverityDistribution, attachment: float) -> float:
    r"""Returns :math:`E[X] - \mathrm{LEV}(d)` for an unlimited excess layer
    attaching at *attachment*."""
    return dist.excess_pure_premium(attachment)


def increased_limits_factor(
    dist: SeverityDistribution, d: float, base_d: float
) -> float:
    r"""Returns :math:`\mathrm{LEV}(d) / \mathrm{LEV}(\text{base\_d})`."""
    return dist.increased_limits_factor(d, base_d)


def layer_expected_value(
    dist: SeverityDistribution, lower: float, upper: float
) -> float:
    r"""Expected loss in the finite layer ``[lower, upper]``:
    :math:`\mathrm{LEV}(\text{upper}) - \mathrm{LEV}(\text{lower})`."""
    if upper <= lower:
        raise ValueError("upper must exceed lower for a non-degenerate layer")
    return dist.limited_expected_value(upper) - dist.limited_expected_value(lower)
