"""Smoke tests for the Phase 0 project foundation."""

from riskchallenger import __version__


def test_package_imports() -> None:
    """The production package is importable and exposes a version."""
    assert __version__ == "0.1.0"
