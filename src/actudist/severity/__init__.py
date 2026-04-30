"""Continuous heavy-tail severity distributions.

Each distribution lives in its own module and registers itself with
:data:`actudist.fitting.SEVERITY_REGISTRY` at import time. Phase 1 of the
v0.1.0 roadmap fills these in one by one (Klugman Appendix A
parameterizations).
"""

from __future__ import annotations

# Concrete distribution modules will be imported here as they land in
# Phase 1. Importing this package therefore triggers their @register
# decorators and populates SEVERITY_REGISTRY.
