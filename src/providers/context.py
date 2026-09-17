"""건물 + 고도 그리드 + 위성 레이어 — 전부 실시간 공개 API."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.providers.elevation import fetch_elevations, sample_terrain_grid
from src.providers.imagery import satellite_layers
from src.providers.osm import fetch_osm_layer
from src.terrain_filter import check_line_of_sight
from src.utils import sector_bounding_box

Coord = Tuple[float, float]

ATTRIBUTION = [
    "OpenStreetMap contributors (ODbL)",
    "Elevation: Open-Meteo / OpenTopoData SRTM",
    "Imagery: Esri World Imagery, NASA GIBS, OpenTopoMap",
]


def fetch_geospatial_context(
    center: Coord,
    bearing_deg: float,
    range_km: float,
    bearing_tolerance_deg: float,
) -> Dict[str, Any]:
    query_range = min(max(range_km, 0.8), 3.0)
    bbox = sector_bounding_box(center, bearing_deg, query_range, bearing_tolerance_deg)
    osm = fetch_osm_layer(center, bearing_deg, query_range)
    buildings: List[Dict[str, Any]] = osm["buildings"]

    if len(buildings) > 120:
        buildings = buildings[:120]

    if buildings:
        points = [(b["lat"], b["lng"]) for b in buildings]
        elevations, elev_source = fetch_elevations(points)
        for building, elev in zip(buildings, elevations):
            building["ground_elevation_m"] = elev
            building["elevation_m"] = elev + float(building.get("height_m") or 0)
        buildings = check_line_of_sight(buildings)
    else:
        elev_source = "none"

    grid = sample_terrain_grid(center, bearing_deg, query_range, size=7)
    imagery = satellite_layers(center[0], center[1], bbox, zoom=15)

    return {
        "candidates": buildings,
        "bbox": bbox,
        "terrain_grid": grid,
        "imagery": imagery,
        "query_range_km": query_range,
        "sources": {
            "buildings": "openstreetmap-overpass",
            "roads": f"{len(osm['roads'])} osm ways",
            "towers": f"{len(osm['towers'])} osm nodes",
            "elevation": elev_source,
            "terrain": grid.get("source"),
            "imagery": list(imagery.keys()),
        },
        "attribution": ATTRIBUTION,
        "live": True,
    }
