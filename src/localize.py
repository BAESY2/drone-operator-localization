"""부채꼴 생성 및 OSM 실시간 후보 로컬라이제이션."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BEARING_TOLERANCE, DEFAULT_RANGE_MIN_KM
from src.providers.context import fetch_geospatial_context
from src.scoring import score_all_candidates
from src.terrain_filter import filter_candidates
from src.utils import add_distance_to_coord


def create_search_sector(
    center_lat: float,
    center_lng: float,
    bearing_deg: float,
    range_km: float,
    bearing_tolerance_deg: float = DEFAULT_BEARING_TOLERANCE,
    num_samples: int = 64,
    radius_min_km: float = DEFAULT_RANGE_MIN_KM,
) -> Dict[str, Any]:
    """부채꼴 영역 GeoJSON Polygon."""
    bearing_min = (bearing_deg - bearing_tolerance_deg) % 360
    bearing_max = (bearing_deg + bearing_tolerance_deg) % 360
    bearing_range = bearing_max - bearing_min
    if bearing_range < 0:
        bearing_range += 360

    outer_arc: List[List[float]] = []
    inner_arc: List[List[float]] = []
    for i in range(num_samples + 1):
        bearing = (bearing_min + (bearing_range * i / num_samples)) % 360
        lat_max, lng_max = add_distance_to_coord(
            (center_lat, center_lng), bearing, range_km
        )
        outer_arc.append([lng_max, lat_max])
        lat_min, lng_min = add_distance_to_coord(
            (center_lat, center_lng), bearing, radius_min_km
        )
        inner_arc.append([lng_min, lat_min])

    inner_arc.reverse()
    polygon_coords = outer_arc + inner_arc + [outer_arc[0]]
    return {"type": "Polygon", "coordinates": [polygon_coords]}


def localize_drone_operator(
    center_lat: float,
    center_lng: float,
    bearing_deg: float,
    drone_range_km: float = 10,
    num_results: int = 10,
    bearing_tolerance_deg: float = DEFAULT_BEARING_TOLERANCE,
) -> Dict[str, Any]:
    """실시간 OSM 건물 + 고도 + 위성 URL → 상위 N 후보."""
    sector = create_search_sector(
        center_lat,
        center_lng,
        bearing_deg,
        drone_range_km,
        bearing_tolerance_deg=bearing_tolerance_deg,
    )
    geo = fetch_geospatial_context(
        (center_lat, center_lng),
        bearing_deg,
        drone_range_km,
        bearing_tolerance_deg,
    )
    filtered = filter_candidates(
        sector,
        geo["candidates"],
        center=(center_lat, center_lng),
        bearing_deg=bearing_deg,
        range_km=max(drone_range_km, geo["query_range_km"]),
        bearing_tolerance_deg=bearing_tolerance_deg,
        num_results=80,
    )
    ranked = score_all_candidates(
        filtered,
        center_pos=(center_lat, center_lng),
        num_results=num_results,
    )
    return {
        "search_sector": sector,
        "primary_target": ranked[0] if ranked else None,
        "top_10_candidates": ranked,
        "filtered_count": len(filtered),
        "osm_building_count": len(geo["candidates"]),
        "terrain_grid": geo["terrain_grid"],
        "imagery": geo["imagery"],
        "bbox": geo["bbox"],
        "sources": geo["sources"],
        "attribution": geo["attribution"],
        "query_range_km": geo["query_range_km"],
    }
