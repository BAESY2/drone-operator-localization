"""지도 측정 — 거리·방위·고도 샘플 (전 세계 좌표)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.providers.elevation import fetch_elevations
from src.utils import distance_between_coords, initial_bearing

Coord = Tuple[float, float]


def measure_segment(
    a: Coord,
    b: Coord,
    include_elevation: bool = True,
) -> Dict[str, Any]:
    dist_km = distance_between_coords(a, b)
    bearing = initial_bearing(a, b)
    back = initial_bearing(b, a)
    result: Dict[str, Any] = {
        "from": {"latitude": a[0], "longitude": a[1]},
        "to": {"latitude": b[0], "longitude": b[1]},
        "distance_km": round(dist_km, 4),
        "distance_m": round(dist_km * 1000.0, 1),
        "bearing_deg": round(bearing, 2),
        "back_bearing_deg": round(back, 2),
    }
    if include_elevation:
        elevs, src = fetch_elevations([a, b])
        result["elevation"] = {
            "from_m": elevs[0] if elevs else None,
            "to_m": elevs[1] if len(elevs) > 1 else None,
            "delta_m": (
                round(elevs[1] - elevs[0], 1) if len(elevs) > 1 else None
            ),
            "source": src,
        }
    return result


def measure_path(
    points: List[Coord],
    include_elevation: bool = True,
) -> Dict[str, Any]:
    if len(points) < 2:
        return {"success": False, "error": "need >= 2 points"}
    segments = []
    total = 0.0
    for i in range(1, len(points)):
        seg = measure_segment(points[i - 1], points[i], include_elevation=False)
        segments.append(seg)
        total += seg["distance_km"]
    elev = None
    if include_elevation:
        elevs, src = fetch_elevations(points)
        elev = {
            "points_m": elevs,
            "min_m": min(elevs) if elevs else None,
            "max_m": max(elevs) if elevs else None,
            "source": src,
        }
    return {
        "success": True,
        "point_count": len(points),
        "total_distance_km": round(total, 4),
        "total_distance_m": round(total * 1000.0, 1),
        "segments": segments,
        "elevation_profile": elev,
    }
