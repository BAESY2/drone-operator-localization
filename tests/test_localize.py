"""Geometry tests — no mock geospatial datasets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.localize import create_search_sector
from src.utils import add_distance_to_coord, distance_between_coords, normalize_bearing


def test_add_distance_to_coord():
    center = (50.45, 30.52)
    south = add_distance_to_coord(center, 180, 1)
    assert abs(south[0] - (50.45 - 0.009)) < 0.002
    east = add_distance_to_coord(center, 90, 1)
    assert abs(east[1] - (30.52 + 0.014)) < 0.003


def test_distance_between_coords():
    dist = distance_between_coords((50.45, 30.52), (50.46, 30.53))
    assert 1 < dist < 3


def test_normalize_bearing():
    assert normalize_bearing(360) == 0
    assert normalize_bearing(-10) == 350


def test_create_search_sector():
    sector = create_search_sector(50.45, 30.52, 235, 10)
    assert sector["type"] == "Polygon"
    coords = sector["coordinates"][0]
    assert len(coords) > 50
    assert coords[0] == coords[-1]
