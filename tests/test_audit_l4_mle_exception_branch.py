"""AUDIT-L-4 coverage test.

Covers fit_continuous_mle's exception-handling branch:
``except (ValueError, FloatingPointError, OverflowError): return _HUGE``.
We force the optimizer to evaluate parameters that raise ValueError
inside the distribution constructor; the optimizer must absorb the
exception via _HUGE and finish without propagating it.
"""

from __future__ import annotations

import numpy as np

from actudist._mle import fit_continuous_mle
from actudist.severity.exponential import Exponential


class _RaisingExponential(Exponential):
    """Exponential clone whose constructor refuses every other call
    with a ValueError, forcing the optimizer to land on the
    exception-handling branch in fit_continuous_mle. After the trap
    bumps neg_ll to _HUGE, the optimizer steps elsewhere and the
    fit completes."""

    _calls = 0

    def __init__(self, theta: float | None = None) -> None:
        if theta is not None:
            type(self)._calls += 1
            if type(self)._calls % 2 == 0:
                raise ValueError("artificial trap to exercise except-branch")
        super().__init__(theta=theta)


def test_fit_continuous_traps_constructor_value_error():
    rng = np.random.default_rng(7)
    data = rng.exponential(scale=2.0, size=200)
    _RaisingExponential._calls = 0
    params = fit_continuous_mle(_RaisingExponential, data)
    # Without the trap branch the optimizer would bubble the
    # ValueError. The fact that it returns a sensible theta means
    # neg_ll mapped the failures to _HUGE and the optimizer kept
    # going.
    assert "theta" in params
    assert 0.5 < params["theta"] < 3.5
    assert _RaisingExponential._calls > 1, "trap was never exercised"
