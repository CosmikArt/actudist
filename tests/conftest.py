"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy Generator for tests that need reproducibility."""
    return np.random.default_rng(42)
