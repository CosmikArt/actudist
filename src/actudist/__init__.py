"""actudist: actuarial probability distributions for P&C loss modeling.

Severity and frequency distributions in Klugman's parameterizations.
MLE fitting accepts right-censored and truncated samples.

Concrete distributions live in :mod:`actudist.severity` and
:mod:`actudist.frequency` and self-register at import time.
"""

from __future__ import annotations

__version__ = "0.1.0"

from actudist.base import (
    ActuarialDistribution,
    FrequencyDistribution,
    SeverityDistribution,
)
from actudist.fitting import (
    DistributionFitter,
    FREQUENCY_REGISTRY,
    SEVERITY_REGISTRY,
    register_frequency,
    register_severity,
)
from actudist.gof import GoodnessOfFit

# Importing the sub-packages triggers concrete distributions to register.
from actudist import frequency as _frequency  # noqa: F401
from actudist import severity as _severity  # noqa: F401

__all__ = [
    "ActuarialDistribution",
    "SeverityDistribution",
    "FrequencyDistribution",
    "DistributionFitter",
    "GoodnessOfFit",
    "SEVERITY_REGISTRY",
    "FREQUENCY_REGISTRY",
    "register_severity",
    "register_frequency",
]
