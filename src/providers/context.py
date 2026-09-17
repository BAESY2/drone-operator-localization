"""건물 + 고도 그리드 + 위성 레이어 — 전부 실시간 공개 API."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.providers.elevation import fetch_elevations, sample_terrain_grid
from src.providers.imagery import satellite_layers
from src.providers.osm import fetch_osm_layer
from src.terrain_filter import check_line_of_sight
from src.utils import add_distance_to_coord, sector_bounding_box

Coord = Tuple[float, float]

ATTRIBUTION = [
    "OpenStreetMap contributors (ODbL)",
    "Elevation: Open-Meteo / OpenTopoData SRTM",
    "Imagery: Esri World Imagery, NASA GIBS, OpenTopoMap",
]


def _synthetic_sector_candidates(
    center: Coord,
    bearing_deg: float,
    range_km: float,
    bearing_tolerance_deg: float,
) -> List[Dict[str, Any]]:
    """
    When Overpass is down: seed bearing-aligned pseudo-OP points so FIX still runs.
    Marked source=synthetic_sector — not OSM truth.
    """
    out: List[Dict[str, Any]] = []
    # denser near mid-range (typical OP standoff)
    dist_fracs = (0.25, 0.40, 0.55, 0.70, 0.85, 0.95)
    half = max(8.0, min(float(bearing_tolerance_deg), 35.0))
    offs = (-half, -half * 0.5, 0.0, half * 0.5, half)
    idx = 0
    for d_frac in dist_fracs:
        for off in offs:
            brg = (bearing_deg + off) % 360.0
            lat, lng = add_distance_to_coord(center, brg, range_km * d_frac)
            idx += 1
            out.append(
                {
                    "id": f"syn_sector_{idx}",
                    "lat": lat,
                    "lng": lng,
                    "building_type": "unknown",
                    "height_m": 12.0,
                    "name": None,
                    "source": "synthetic_sector",
                    "military": False,
                    "rooftop_access": False,
                    "dist_to_road_m": None,
                    "cell_tower_nearby": False,
                }
            )
    return out


def fetch_geospatial_context(
    center: Coord,
    bearing_deg: float,
    range_km: float,
    bearing_tolerance_deg: float,
) -> Dict[str, Any]:
    query_range = min(max(range_km, 0.8), 4.0)
    bbox = sector_bounding_box(center, bearing_deg, query_range, bearing_tolerance_deg)
    osm = fetch_osm_layer(center, bearing_deg, query_range)
    buildings: List[Dict[str, Any]] = list(osm.get("buildings") or [])
    osm_degraded = bool(osm.get("degraded") or osm.get("error"))
    osm_error = osm.get("error")

    if not buildings:
        buildings = _synthetic_sector_candidates(
            center, bearing_deg, query_range, bearing_tolerance_deg
        )
        osm_degraded = True

    if len(buildings) > 280:
        buildings = buildings[:280]

    if buildings:
        points = [(b["lat"], b["lng"]) for b in buildings]
        try:
            elevations, elev_source = fetch_elevations(points)
            for building, elev in zip(buildings, elevations):
                building["ground_elevation_m"] = elev
                building["elevation_m"] = elev + float(building.get("height_m") or 0)
            buildings = check_line_of_sight(buildings)
        except Exception:  # noqa: BLE001
            elev_source = "unavailable"
            for building in buildings:
                building.setdefault("ground_elevation_m", 0.0)
                building.setdefault(
                    "elevation_m", float(building.get("height_m") or 12.0)
                )
    else:
        elev_source = "none"

    try:
        grid = sample_terrain_grid(center, bearing_deg, query_range, size=7)
    except Exception:  # noqa: BLE001
        grid = {"source": "unavailable", "cells": []}
    imagery = satellite_layers(center[0], center[1], bbox, zoom=15)

    sources = {
        "buildings": (
            "synthetic_sector_fallback"
            if any(b.get("source") == "synthetic_sector" for b in buildings)
            else "openstreetmap-overpass"
        ),
        "roads": f"{len(osm.get('roads') or [])} osm ways",
        "towers": f"{len(osm.get('towers') or [])} osm nodes",
        "elevation": elev_source,
        "terrain": grid.get("source"),
        "imagery": list(imagery.keys()),
        "osm_degraded": osm_degraded,
    }
    if osm_error:
        sources["osm_error"] = str(osm_error)[:120]

    return {
        "candidates": buildings,
        "bbox": bbox,
        "terrain_grid": grid,
        "imagery": imagery,
        "query_range_km": query_range,
        "sources": sources,
        "attribution": ATTRIBUTION,
        "live": not osm_degraded,
        "osm_degraded": osm_degraded,
    }
