"""Tests for T-201's environment guard.

The guard exists so a wrong interpreter fails on line one of an entry point
rather than as a subtle divergence later. These check it actually refuses a
mismatch, and that the smoke-test report names the packages Phase 2 imports.
"""

import sys

import pytest

from src import env_check as E


def test_require_python_passes_on_the_running_interpreter():
    """The suite runs on 3.11, so the real check must be a no-op here."""
    assert sys.version_info[:2] == (3, 11)
    E.require_python()  # does not raise


def test_require_python_rejects_a_near_miss():
    """3.12 mostly works, which is exactly why it must be refused."""
    with pytest.raises(E.EnvironmentError_) as exc:
        E.require_python((3, 12))
    assert "3.12 is required" in str(exc.value)
    assert "env/bin/activate" in str(exc.value)


def test_require_python_rejects_a_major_mismatch():
    with pytest.raises(E.EnvironmentError_):
        E.require_python((4, 0))


def test_check_env_reports_the_interpreter_and_the_training_stack():
    report = E.check_env()
    assert report["python"].startswith("3.11.")
    for label in ("torch", "numpy", "scikit-learn", "PyYAML", "transformers"):
        assert label in report["packages"]
        assert report["packages"][label] != "unknown"


def test_check_env_reports_cuda_state_without_assuming_a_gpu():
    report = E.check_env()
    assert isinstance(report["cuda_available"], bool)
    if report["cuda_available"]:
        assert report["device_count"] >= 1
    else:
        assert report["device_count"] == 0


def test_smoke_main_exits_zero_on_this_environment():
    assert E.main() == 0
