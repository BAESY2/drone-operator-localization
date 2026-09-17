"""OpenStreetMap Overpass — 실제 건물·도로·통신탑 (키 불필요)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode

from src.providers.http import cached_json_stale, get_json
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
    """Denser samples along bearing for more building coverage."""
    max_d = min(max(range_km, 0.8), 4.0)
    points: List[Coord] = [center]
    dist = 0.28
    while dist <= max_d:
        points.append(add_distance_to_coord(center, bearing_deg, dist))
        # slight left/right offsets to catch flank buildings
        points.append(
            add_distance_to_coord(center, (bearing_deg - 9) % 360, dist * 0.92)
        )
        points.append(
            add_distance_to_coord(center, (bearing_deg + 9) % 360, dist * 0.92)
        )
        dist += 0.45
    return points


def _query(center: Coord, bearing_deg: float, range_km: float) -> str:
    parts = []
    for lat, lng in _sample_points(center, bearing_deg, range_km):
        parts.append(f'way["building"](around:420,{lat:.5f},{lng:.5f});')
        parts.append(
            f'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]'
            f"(around:420,{lat:.5f},{lng:.5f});"
        )
        parts.append(
            f'node["man_made"~"^(mast|tower|communications_tower)$"](around:700,{lat:.5f},{lng:.5f});'
        )
    joined = "\n  ".join(parts)
    return f"""
[out:json][timeout:35];
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
    """
    Fetch OSM buildings/roads/towers.
    Fail-open: on Overpass outage return empty lists (+ optional stale cache).
    Never raise — callers must keep producing a defensive fix.
    """
    query = _query(center, bearing_deg, range_km)
    body = urlencode({"data": query}).encode("utf-8")
    cache_key = f"overpass:{query}"
    last_error = None
    payload: Dict[str, Any] = {}
    degraded = False

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            payload = get_json(
                endpoint,
                method="POST",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                cache_key=cache_key,
                ttl_sec=6 * 3600,
                timeout=10,
                allow_stale=True,
            )
            if payload.get("elements") is not None:
                break
        except RuntimeError as exc:
            last_error = exc
            continue

    if not payload or payload.get("elements") is None:
        stale = cached_json_stale(cache_key)
        if stale and stale.get("elements") is not None:
            payload = stale
            degraded = True
        else:
            return {
                "buildings": [],
                "roads": [],
                "towers": [],
                "degraded": True,
                "error": str(last_error or "Overpass unavailable")[:160],
            }

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

    out: Dict[str, Any] = {"buildings": buildings, "roads": roads, "towers": towers}
    if degraded or last_error:
        out["degraded"] = degraded or False
        if last_error and not buildings:
            out["error"] = str(last_error)[:160]
    return out
