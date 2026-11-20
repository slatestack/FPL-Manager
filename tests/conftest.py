from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def bootstrap_payload() -> dict[str, Any]:
    return load_fixture("bootstrap_static.json")


@pytest.fixture
def fixtures_payload() -> list[dict[str, Any]]:
    return load_fixture("fixtures.json")
