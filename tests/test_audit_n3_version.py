"""AUDIT-N-3: assert __version__ matches the package metadata.

Catches drift between src/actudist/__init__.py and pyproject.toml at
release time, before the wheel is built.
"""

from __future__ import annotations

from importlib.metadata import version

import actudist


def test_version_matches_package_metadata():
    declared = actudist.__version__
    metadata = version("actudist")
    assert declared == metadata, (
        f"actudist.__version__ = {declared!r} but installed metadata "
        f"says {metadata!r}. Update src/actudist/__init__.py and "
        f"pyproject.toml together."
    )
