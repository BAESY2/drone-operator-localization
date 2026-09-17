"""pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def seoul_city_hall():
    return (37.5665, 126.9780)
