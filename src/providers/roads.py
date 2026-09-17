"""OSM road segments (geometry) for egress routing sketches."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode

from src.providers.http import get_json

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

Coord = Tuple[float, float]


def fetch_road_segments(
    center: Coord,
    radius_m: float = 3500,
) -> List[Dict[str, Any]]:
    """Return highway ways with lat/lng polylines near center."""
    lat, lng = center
    r = int(min(max(radius_m, 800), 8000))
    query = f"""
[out:json][timeout:25];
way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|track|path|cycleway|footway)$"](around:{r},{lat:.5f},{lng:.5f});
out geom tags;
""".strip()
    body = urlencode({"data": query}).encode("utf-8")
    payload: Dict[str, Any] = {}
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            payload = get_json(
                endpoint,
                method="POST",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                cache_key=f"roads:{lat:.4f},{lng:.4f}:{r}",
                ttl_sec=6 * 3600,
                timeout=28,
            )
            if payload.get("elements") is not None:
                break
        except RuntimeError as exc:
            last_error = exc
            continue
    if not payload:
        raise RuntimeError(f"Overpass roads unavailable: {last_error}")

    segments: List[Dict[str, Any]] = []
    for el in payload.get("elements") or []:
        tags = el.get("tags") or {}
        hw = tags.get("highway")
        geom = el.get("geometry") or []
        if not hw or len(geom) < 2:
            continue
        coords = [(float(p["lat"]), float(p["lon"])) for p in geom if "lat" in p and "lon" in p]
        if len(coords) < 2:
            continue
        segments.append(
            {
                "id": el.get("id"),
                "highway": hw,
                "name": tags.get("name"),
                "coords": coords,
            }
        )
    return segments
