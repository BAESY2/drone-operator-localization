"""지형/후보 필터링."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BEARING_TOLERANCE, DEFAULT_RANGE_MIN_KM
from src.utils import point_in_bearing_sector


def filter_candidates(
    sector: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    center: Optional[Tuple[float, float]] = None,
    bearing_deg: float = 0,
    range_km: float = 10,
    bearing_tolerance_deg: float = DEFAULT_BEARING_TOLERANCE,
    filters: Optional[Dict[str, Any]] = None,
    num_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    부채꼴 내 후보 필터링.

    GeoJSON/DEM이 없을 때도 dict 후보 리스트로 동작.
    """
    _ = sector  # GeoJSON clip 경로용 예약
    if filters is None:
        filters = {
            "building": True,
            "los_to_sky": True,
            "elevation": "high",
            "road_access": True,
            "open_space": True,
        }

    filtered: List[Dict[str, Any]] = []

    for c in candidates:
        lat = c.get("lat")
        lng = c.get("lng")
        if lat is None or lng is None:
            # GeoJSON-like
            geom = c.get("geometry") or {}
            coords = geom.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lng, lat = coords[0], coords[1]
                c = {**c, "lat": lat, "lng": lng}
            else:
                continue

        if center is not None:
            if not point_in_bearing_sector(
                center,
                (lat, lng),
                bearing_deg,
                bearing_tolerance_deg,
                DEFAULT_RANGE_MIN_KM,
                range_km,
            ):
                continue

        if filters.get("road_access") and c.get("dist_to_road_m") is not None:
            if c["dist_to_road_m"] >= 1500:
                continue

        if filters.get("los_to_sky") and c.get("los_clear") is False:
            # 완전 제거 대신 후순위 — 일단 통과시키되 표시
            pass

        filtered.append(c)

    # 고도 우선 정렬 후 상위 N
    if filters.get("elevation") == "high":
        filtered.sort(key=lambda x: x.get("elevation_m", 0), reverse=True)

    return filtered[:num_results]


def check_line_of_sight(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """주변보다 현저히 낮은 후보의 LOS를 false로 표시."""
    result = []
    for building in candidates:
        elev = building.get("elevation_m", 0)
        lat = building.get("lat", 0)
        lng = building.get("lng", 0)
        nearby_higher = 0
        for other in candidates:
            if other is building:
                continue
            dlat = abs(other.get("lat", 0) - lat)
            dlng = abs(other.get("lng", 0) - lng)
            if dlat < 0.0005 and dlng < 0.0005:
                if other.get("elevation_m", 0) > elev * 1.2:
                    nearby_higher += 1
        updated = dict(building)
        updated["los_clear"] = nearby_higher == 0
        result.append(updated)
    return result
