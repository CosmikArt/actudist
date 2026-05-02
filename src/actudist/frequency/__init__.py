"""Discrete frequency distributions (Klugman ch. 6). Each module registers
its class with :data:`actudist.fitting.FREQUENCY_REGISTRY` on import."""

from __future__ import annotations

from actudist.frequency import poisson as _poisson  # noqa: F401
from actudist.frequency import binomial as _binomial  # noqa: F401
from actudist.frequency import negative_binomial as _negative_binomial  # noqa: F401
from actudist.frequency import geometric as _geometric  # noqa: F401
from actudist.frequency import zip as _zip  # noqa: F401
from actudist.frequency import zinb as _zinb  # noqa: F401
