"""Goodness-of-fit tests and diagnostic plots.

Phase 0 wires up the public class so downstream modules can import it; the
test implementations land in Phase 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from actudist.base import ActuarialDistribution


class GoodnessOfFit:
    """KS, Anderson-Darling, chi-squared tests and PP/QQ diagnostics for a
    fitted distribution.

    Parameters
    ----------
    distribution : ActuarialDistribution | None
        Fitted distribution. ``None`` is permitted for construction-only
        tests; calling a test method requires a real distribution.
    data : array_like
        The sample to test against.
    """

    def __init__(
        self,
        distribution: ActuarialDistribution | None,
        data: ArrayLike,
    ) -> None:
        self.distribution = distribution
        self.data = np.asarray(data)

    def ks_test(self, *, n_boot: int = 1000) -> dict[str, float]:
        raise NotImplementedError("KS test is implemented in Phase 3.")

    def anderson_darling_test(self, *, n_boot: int = 1000) -> dict[str, Any]:
        raise NotImplementedError("AD test is implemented in Phase 3.")

    def chi_squared_test(self, n_bins: int = 10) -> dict[str, float]:
        raise NotImplementedError("Chi-squared test is implemented in Phase 3.")

    def pp_plot(self, ax: Any | None = None) -> Any:
        raise NotImplementedError("PP plot is implemented in Phase 3.")

    def qq_plot(self, ax: Any | None = None) -> Any:
        raise NotImplementedError("QQ plot is implemented in Phase 3.")
