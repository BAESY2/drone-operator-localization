"""Scoring on OSM-shaped records."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring import calculate_score, score_all_candidates


def test_calculate_score(seoul_city_hall):
    building = {
        "lat": 37.565,
        "lng": 126.985,
        "elevation_m": 45,
        "los_clear": True,
        "dist_to_road_m": 80,
        "building_type": "residential",
        "cell_tower_nearby": True,
        "military": False,
    }
    score, reasoning = calculate_score(building, seoul_city_hall)
    assert score > 0
    assert "elevation" in reasoning
    assert "military" in reasoning


def test_score_all_candidates(seoul_city_hall):
    candidates = [
        {
            "id": "osm_way_1",
            "lat": 37.565,
            "lng": 126.985,
            "elevation_m": 40,
            "los_clear": True,
            "dist_to_road_m": 50,
            "building_type": "residential",
            "cell_tower_nearby": False,
        },
        {
            "id": "osm_way_2",
            "lat": 37.567,
            "lng": 126.99,
            "elevation_m": 20,
            "los_clear": True,
            "dist_to_road_m": 200,
            "building_type": "commercial",
            "cell_tower_nearby": True,
        },
    ]
    ranked = score_all_candidates(candidates, seoul_city_hall, num_results=2)
    assert ranked[0]["rank"] == 1
    assert ranked[0]["probability"] >= ranked[1]["probability"]
