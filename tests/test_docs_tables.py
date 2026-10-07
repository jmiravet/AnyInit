"""The tables in the docs are the output of the code, not copies."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tables():
    path = ROOT / "docs" / "experiments" / "stability_table.py"
    spec = importlib.util.spec_from_file_location("stability_table", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stability_table_matches_the_code(tables):
    guide = (ROOT / "docs" / "guide" / "stability.md").read_text(encoding="utf-8")
    assert "\n".join(tables.stability_table()) in guide


def test_reference_moment_table_matches_the_code(tables):
    reference = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
    assert "\n".join(tables.moment_table()) in reference
