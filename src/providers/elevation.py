"""무료 고도 API — Open-Meteo, OpenTopoData(SRTM)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from src.providers.http import get_json
from src.utils import add_distance_to_coord, chunked

Coord = Tuple[float, float]


def _open_meteo(points: Sequence[Coord]) -> List[float]:
    out: List[float] = []
    for group in chunked(list(points), 80):
        lats = ",".join(f"{p[0]:.5f}" for p in group)
        lngs = ",".join(f"{p[1]:.5f}" for p in group)
        data = get_json(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lats, "longitude": lngs},
            cache_key=f"meteo:{lats}:{lngs}",
            ttl_sec=24 * 3600,
            timeout=15,
        )
        elevations = data.get("elevation") or []
        if len(elevations) != len(group):
            raise RuntimeError("Open-Meteo elevation count mismatch")
        out.extend(float(v) for v in elevations)
    return out


def _opentopo(points: Sequence[Coord]) -> List[float]:
    out: List[float] = []
    for group in chunked(list(points), 90):
        loc = "|".join(f"{p[0]:.5f},{p[1]:.5f}" for p in group)
        data = get_json(
            "https://api.opentopodata.org/v1/srtm90m",
            params={"locations": loc},
            cache_key=f"srtm90:{loc}",
            ttl_sec=24 * 3600,
            timeout=20,
        )
        results = data.get("results") or []
        for item in results:
            elev = item.get("elevation")
            out.append(float(elev) if elev is not None else 0.0)
    if len(out) != len(points):
        raise RuntimeError("OpenTopoData elevation count mismatch")
    return out


def fetch_elevations(points: Sequence[Coord]) -> Tuple[List[float], str]:
    """점 고도(m). 성공한 소스 이름을 함께 반환."""
    if not points:
        return [], "none"
    try:
        return _open_meteo(points), "open-meteo"
    except RuntimeError:
        return _opentopo(points), "opentopodata-srtm90m"


def sample_terrain_grid(
    center: Coord,
    bearing_deg: float,
    range_km: float,
    size: int = 8,
) -> dict:
    """부채꼴 방향의 간이 3D 고도 그리드."""
    half = range_km / 2.0
    origin = add_distance_to_coord(center, bearing_deg, half)
    step_km = max(range_km / max(size - 1, 1), 0.2)
    points: List[Coord] = []
    for row in range(size):
        for col in range(size):
            north = (row - (size - 1) / 2.0) * step_km
            east = (col - (size - 1) / 2.0) * step_km
            p = add_distance_to_coord(origin, 0, north)
            p = add_distance_to_coord(p, 90, east)
            points.append(p)
    elevations, source = fetch_elevations(points)
    return {
        "size": size,
        "source": source,
        "points": [
            {"lat": lat, "lng": lng, "elevation_m": elev}
            for (lat, lng), elev in zip(points, elevations)
        ],
        "min_m": min(elevations) if elevations else 0,
        "max_m": max(elevations) if elevations else 0,
    }
