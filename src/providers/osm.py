"""OpenStreetMap Overpass — 실제 건물·도로·통신탑 (키 불필요)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode

from src.providers.http import get_json
from src.utils import add_distance_to_coord, nearest_distance_m

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

RESIDENTIAL = {
    "yes",
    "house",
    "apartments",
    "residential",
    "detached",
    "terrace",
    "semidetached_house",
    "dormitory",
    "bungalow",
}
COMMERCIAL = {"commercial", "retail", "office", "warehouse", "hotel", "supermarket"}
INDUSTRIAL = {"industrial", "factory", "manufacture", "hangar"}
Coord = Tuple[float, float]


def _classify_building(tags: Dict[str, Any]) -> str:
    value = str(tags.get("building") or tags.get("building:use") or "yes").lower()
    if value in RESIDENTIAL:
        return "residential"
    if value in COMMERCIAL:
        return "commercial"
    if value in INDUSTRIAL:
        return "industrial"
    return "unknown"


def _height_m(tags: Dict[str, Any]) -> float:
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        try:
            return float(str(raw).lower().replace("m", "").split()[0])
        except ValueError:
            pass
    levels = tags.get("building:levels")
    if levels:
        try:
            return float(str(levels).split(";")[0]) * 3.0
        except ValueError:
            pass
    return 10.0


def _center(el: Dict[str, Any]) -> Coord | None:
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])
    center = el.get("center") or {}
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def _sample_points(center: Coord, bearing_deg: float, range_km: float) -> List[Coord]:
    """방위각을 따라 실제 OSM around 쿼리 지점."""
    max_d = min(max(range_km, 0.8), 3.0)
    points: List[Coord] = []
    dist = 0.45
    while dist <= max_d:
        points.append(add_distance_to_coord(center, bearing_deg, dist))
        dist += 0.7
    return points


def _query(center: Coord, bearing_deg: float, range_km: float) -> str:
    parts = []
    for lat, lng in _sample_points(center, bearing_deg, range_km):
        parts.append(f'way["building"](around:480,{lat:.5f},{lng:.5f});')
        parts.append(
            f'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]'
            f"(around:480,{lat:.5f},{lng:.5f});"
        )
        parts.append(
            f'node["man_made"~"^(mast|tower|communications_tower)$"](around:800,{lat:.5f},{lng:.5f});'
        )
    joined = "\n  ".join(parts)
    return f"""
[out:json][timeout:25];
(
  {joined}
);
out center tags;
""".strip()


def fetch_osm_layer(
    center: Coord,
    bearing_deg: float,
    range_km: float,
) -> Dict[str, List]:
    query = _query(center, bearing_deg, range_km)
    body = urlencode({"data": query}).encode("utf-8")
    last_error = None
    payload: Dict[str, Any] = {}

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            payload = get_json(
                endpoint,
                method="POST",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                cache_key=f"overpass:{query}",
                ttl_sec=6 * 3600,
                timeout=28,
            )
            if payload.get("elements") is not None:
                break
        except RuntimeError as exc:
            last_error = exc
            continue

    if not payload:
        raise RuntimeError(f"Overpass unavailable: {last_error}")

    buildings: List[Dict[str, Any]] = []
    roads: List[Coord] = []
    towers: List[Coord] = []
    seen = set()

    for el in payload.get("elements") or []:
        tags = el.get("tags") or {}
        pos = _center(el)
        if not pos:
            continue
        lat, lng = pos
        key = (el.get("type"), el.get("id"))
        if key in seen:
            continue
        seen.add(key)

        if "building" in tags:
            buildings.append(
                {
                    "id": f"osm_{el.get('type')}_{el.get('id')}",
                    "lat": lat,
                    "lng": lng,
                    "building_type": _classify_building(tags),
                    "height_m": _height_m(tags),
                    "name": tags.get("name"),
                    "source": "openstreetmap",
                    "osm_id": el.get("id"),
                    "military": tags.get("military") or tags.get("landuse") == "military",
                    "rooftop_access": True,
                }
            )
        elif tags.get("highway"):
            roads.append((lat, lng))
        else:
            towers.append((lat, lng))

    for building in buildings:
        point = (building["lat"], building["lng"])
        building["dist_to_road_m"] = nearest_distance_m(point, roads)
        building["cell_tower_nearby"] = nearest_distance_m(point, towers) < 800

    return {"buildings": buildings, "roads": roads, "towers": towers}
