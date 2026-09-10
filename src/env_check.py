"""Entry-point environment guard (CLAUDE2.md T-201).

Phase 2 shares the Phase 1 venv, so the interpreter and the pinned libraries
must be exactly what the corpus was built against. Every Phase 2 entry point
calls :func:`require_python` before doing any work, so a wrong interpreter
fails loudly on line one instead of surfacing as a confusing error pages later:

    from src.env_check import require_python
    require_python()

Run as a module for the smoke test — it checks the interpreter, imports the
whole training stack, and reports versions and CUDA state:

    python -m src.env_check

Exits non-zero if the interpreter is wrong or any required package is missing,
so "environment is set up" is an automated check rather than an eyeball.
"""

from __future__ import annotations

import importlib
import sys

# CLAUDE.md §3: "Python 3.11 exactly." Not >=; the corpus and its determinism
# guarantees were established on 3.11 and nothing re-establishes them elsewhere.
REQUIRED_PYTHON = (3, 11)

# Import name -> human label. These are the packages Phase 2 code imports
# directly; their transitive deps are covered by requirements.txt.
REQUIRED_PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "transformers": "transformers",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
}


class EnvironmentError_(RuntimeError):
    """Raised when the runtime is not the one Phase 2 must run on."""


def require_python(required: tuple[int, int] = REQUIRED_PYTHON) -> None:
    """Abort unless the running interpreter is exactly ``required``.

    Called at the top of every Phase 2 entry point. Raises rather than warns:
    a near-miss like 3.12 mostly works and then silently diverges, which is the
    failure mode determinism (§4 rule 7) exists to prevent.
    """
    actual = sys.version_info[:2]
    if actual != tuple(required):
        want = ".".join(str(p) for p in required)
        got = ".".join(str(p) for p in actual)
        raise EnvironmentError_(
            f"Python {want} is required, but this is {got} "
            f"({sys.executable}). Activate the project venv: "
            f"`source env/bin/activate`."
        )


def check_env() -> dict[str, object]:
    """Verify the interpreter and import the training stack.

    Returns a report dict. Raises :class:`EnvironmentError_` on a wrong
    interpreter and :class:`ImportError` on a missing package, so a caller can
    treat any exception as "environment not ready".
    """
    require_python()

    versions: dict[str, str] = {}
    for import_name, label in REQUIRED_PACKAGES.items():
        module = importlib.import_module(import_name)
        versions[label] = getattr(module, "__version__", "unknown")

    import torch

    cuda = bool(torch.cuda.is_available())
    report: dict[str, object] = {
        "python": ".".join(str(p) for p in sys.version_info[:3]),
        "executable": sys.executable,
        "packages": versions,
        "cuda_available": cuda,
        "device_count": torch.cuda.device_count() if cuda else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda else None,
    }
    return report


def _format(report: dict[str, object]) -> str:
    lines = [f"{'python':<14}{report['python']}  ({report['executable']})"]
    for label, version in report["packages"].items():  # type: ignore[union-attr]
        lines.append(f"{label:<14}{version}")
    if report["cuda_available"]:
        lines.append(
            f"{'cuda':<14}{report['device_count']}x {report['device_name']}"
        )
    else:
        lines.append(f"{'cuda':<14}not available (CPU only)")
    return "\n".join(lines)


def main() -> int:
    try:
        report = check_env()
    except (EnvironmentError_, ImportError) as exc:
        print(f"environment check FAILED: {exc}", file=sys.stderr)
        return 1
    print(_format(report))
    print("\nenvironment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
