"""
actudist — Actuarial probability distributions.

Heavy-tail severity distributions, frequency models, compound distributions,
MLE fitting with profile likelihood CIs, and goodness-of-fit testing.
"""

__version__ = "0.0.1"

from actudist.core import (
    ActuarialDistribution,
    BurrXII,
    DistributionFitter,
    GeneralizedPareto,
    GoodnessOfFit,
    TransformedBeta,
    ZeroInflatedPoisson,
)

__all__ = [
    "ActuarialDistribution",
    "BurrXII",
    "GeneralizedPareto",
    "TransformedBeta",
    "ZeroInflatedPoisson",
    "DistributionFitter",
    "GoodnessOfFit",
]
