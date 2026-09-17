"""Terrain filter on OSM-shaped records."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.localize import create_search_sector
from src.terrain_filter import check_line_of_sight, filter_candidates


def test_filter_candidates(seoul_city_hall):
    lat, lng = seoul_city_hall
    sector = create_search_sector(lat, lng, 90, 3)
    candidates = [
        {
            "id": "a",
            "lat": 37.5665,
            "lng": 126.990,
            "dist_to_road_m": 80,
            "elevation_m": 30,
        },
        {
            "id": "b",
            "lat": 37.5665,
            "lng": 126.990,
            "dist_to_road_m": 4000,
            "elevation_m": 10,
        },
    ]
    filtered = filter_candidates(
        sector,
        candidates,
        center=seoul_city_hall,
        bearing_deg=90,
        range_km=3,
    )
    assert all(c["dist_to_road_m"] < 1500 for c in filtered)


def test_check_line_of_sight():
    updated = check_line_of_sight(
        [
            {"lat": 37.56, "lng": 126.97, "elevation_m": 10},
            {"lat": 37.56, "lng": 126.97, "elevation_m": 40},
        ]
    )
    assert "los_clear" in updated[0]
