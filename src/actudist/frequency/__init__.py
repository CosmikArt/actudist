"""Discrete frequency distributions.

Each distribution lives in its own module and registers itself with
:data:`actudist.fitting.FREQUENCY_REGISTRY` at import time.
"""

from __future__ import annotations

from actudist.frequency import poisson as _poisson  # noqa: F401
from actudist.frequency import binomial as _binomial  # noqa: F401
from actudist.frequency import negative_binomial as _negative_binomial  # noqa: F401
from actudist.frequency import geometric as _geometric  # noqa: F401
