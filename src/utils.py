"""좌표/방위각 유틸리티."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

EARTH_RADIUS_KM = 6371.0
Coord = Tuple[float, float]


def add_distance_to_coord(
    center: Coord,
    bearing_deg: float,
    distance_km: float,
) -> Coord:
    """거리와 방향으로부터 새 좌표 계산 (Haversine)."""
    lat_rad = math.radians(center[0])
    lng_rad = math.radians(center[1])
    bearing_rad = math.radians(bearing_deg)

    lat_new_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance_km / EARTH_RADIUS_KM)
        + math.cos(lat_rad)
        * math.sin(distance_km / EARTH_RADIUS_KM)
        * math.cos(bearing_rad)
    )
    lng_new_rad = lng_rad + math.atan2(
        math.sin(bearing_rad)
        * math.sin(distance_km / EARTH_RADIUS_KM)
        * math.cos(lat_rad),
        math.cos(distance_km / EARTH_RADIUS_KM)
        - math.sin(lat_rad) * math.sin(lat_new_rad),
    )
    return (math.degrees(lat_new_rad), math.degrees(lng_new_rad))


def bearing_to_vector(bearing_deg: float, magnitude: float = 1.0) -> Coord:
    """도수각을 (east, north) 벡터로 변환."""
    bearing_rad = math.radians(bearing_deg)
    return (magnitude * math.sin(bearing_rad), magnitude * math.cos(bearing_rad))


def normalize_bearing(bearing_deg: float) -> float:
    """도수각 정규화 (0~359)."""
    return bearing_deg % 360


def distance_between_coords(coord1: Coord, coord2: Coord) -> float:
    """두 좌표 사이의 거리 (km)."""
    lat1, lng1 = coord1
    lat2, lng2 = coord2
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_KM * c


def initial_bearing(coord1: Coord, coord2: Coord) -> float:
    """coord1 → coord2 초기 방위각 (0–359)."""
    lat1, lng1 = map(math.radians, coord1)
    lat2, lng2 = map(math.radians, coord2)
    dlon = lng2 - lng1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return normalize_bearing(math.degrees(math.atan2(y, x)))


def point_in_bearing_sector(
    center: Coord,
    point: Coord,
    bearing_deg: float,
    tolerance_deg: float,
    min_km: float,
    max_km: float,
) -> bool:
    """점이 부채꼴 범위 안인지 판별."""
    dist = distance_between_coords(center, point)
    if dist < min_km or dist > max_km:
        return False
    diff = abs((initial_bearing(center, point) - bearing_deg + 180) % 360 - 180)
    return diff <= tolerance_deg


def sector_bounding_box(
    center: Coord,
    bearing_deg: float,
    range_km: float,
    bearing_tolerance_deg: float,
    radius_min_km: float = 0.3,
) -> Dict[str, float]:
    """부채꼴을 감싸는 WGS84 bbox."""
    samples: List[Coord] = [center]
    for frac in (0.0, 0.5, 1.0):
        dist = radius_min_km + (range_km - radius_min_km) * frac
        for offset in (-bearing_tolerance_deg, 0.0, bearing_tolerance_deg):
            samples.append(
                add_distance_to_coord(center, (bearing_deg + offset) % 360, dist)
            )
    lats = [p[0] for p in samples]
    lngs = [p[1] for p in samples]
    pad = 0.002
    return {
        "south": min(lats) - pad,
        "north": max(lats) + pad,
        "west": min(lngs) - pad,
        "east": max(lngs) + pad,
    }


def nearest_distance_m(point: Coord, others: Sequence[Coord]) -> float:
    """점과 다른 점들 사이 최근접 거리(m)."""
    if not others:
        return 500.0
    return min(distance_between_coords(point, other) * 1000.0 for other in others)


def deg2num(lat: float, lng: float, zoom: int) -> Tuple[int, int]:
    """WGS84 → slippy map tile (x, y)."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
