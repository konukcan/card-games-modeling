"""Root conftest: register custom pytest markers."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that run actual enumeration (may take minutes)")
