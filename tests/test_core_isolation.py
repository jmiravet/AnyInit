"""The core must not depend on any framework.

This is the invariant that makes optional dependencies work.  If a core module ever
imports torch, a user with only JAX installed gets an ImportError from a library that
had no business needing it -- so the rule is checked mechanically rather than trusted.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

FRAMEWORKS = {"torch", "tensorflow", "keras", "tf_keras", "jax", "jaxlib", "flax"}
CORE = pathlib.Path(__import__("anyinit").__file__).parent / "core"
FRAMEWORK_FREE = ["config.py", "errors.py", "report.py", "core", "backends/base.py"]


def _imported_roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted(CORE.glob("*.py")), ids=lambda p: p.name)
def test_core_module_imports_no_framework(path):
    offenders = _imported_roots(path) & FRAMEWORKS
    assert not offenders, f"{path.name} imports {sorted(offenders)}"


def test_framework_free_modules_import_no_framework():
    package = pathlib.Path(__import__("anyinit").__file__).parent
    for relative in FRAMEWORK_FREE:
        target = package / relative
        files = sorted(target.glob("*.py")) if target.is_dir() else [target]
        for path in files:
            offenders = _imported_roots(path) & FRAMEWORKS
            assert not offenders, f"{path} imports {sorted(offenders)}"


def test_importing_anyinit_loads_no_framework():
    """Check that importing and using the core loads no framework.

    Covers importing the package, listing backends and profiling a builtin.
    """
    import subprocess

    script = (
        "import sys, anyinit\n"
        "anyinit.available_backends()\n"
        "anyinit.activation_profile('relu').chi\n"
        "loaded = {m.split('.')[0] for m in sys.modules} & "
        f"{FRAMEWORKS!r}\n"
        "print(sorted(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", f"frameworks were imported: {result.stdout}"
