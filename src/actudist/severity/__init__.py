"""Continuous heavy-tail severity distributions.

Each distribution lives in its own module and registers itself with
:data:`actudist.fitting.SEVERITY_REGISTRY` at import time. Phase 1 of the
v0.1.0 roadmap fills these in one by one (Klugman Appendix A
parameterizations).
"""

from __future__ import annotations

# Concrete distribution modules are imported here so their @register
# decorators populate SEVERITY_REGISTRY at package import time.

from actudist.severity import exponential as _exponential  # noqa: F401
from actudist.severity import pareto as _pareto  # noqa: F401
from actudist.severity import lognormal as _lognormal  # noqa: F401
from actudist.severity import weibull as _weibull  # noqa: F401
from actudist.severity import gamma as _gamma  # noqa: F401
from actudist.severity import burrxii as _burrxii  # noqa: F401
from actudist.severity import loglogistic as _loglogistic  # noqa: F401
