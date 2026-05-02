"""Continuous severity distributions.

Klugman Appendix A parameterizations. Each module registers its class
with :data:`actudist.fitting.SEVERITY_REGISTRY` on import.
"""

from __future__ import annotations

from actudist.severity import exponential as _exponential  # noqa: F401
from actudist.severity import pareto as _pareto  # noqa: F401
from actudist.severity import lognormal as _lognormal  # noqa: F401
from actudist.severity import weibull as _weibull  # noqa: F401
from actudist.severity import gamma as _gamma  # noqa: F401
from actudist.severity import burrxii as _burrxii  # noqa: F401
from actudist.severity import loglogistic as _loglogistic  # noqa: F401
from actudist.severity import paralogistic as _paralogistic  # noqa: F401
from actudist.severity import inverse_paralogistic as _inverse_paralogistic  # noqa: F401
from actudist.severity import inverse_gaussian as _inverse_gaussian  # noqa: F401
from actudist.severity import transformed_gamma as _transformed_gamma  # noqa: F401
from actudist.severity import (  # noqa: F401
    inverse_transformed_gamma as _inverse_transformed_gamma,
)
from actudist.severity import transformed_beta as _transformed_beta  # noqa: F401
