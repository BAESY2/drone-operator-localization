"""후보 풀 집계: 확률·고도·위치·이유를 한 레코드로 묶는다."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.precision import candidate_precision_score, confidence_radius_m
from src.scoring import calculate_score
from src.utils import distance_between_coords, initial_bearing

Coord = Tuple[float, float]


def _reason_codes(detail: Dict[str, float], building: Dict[str, Any]) -> List[str]:
    """Machine codes — UI translates via I18N (EN/RU/UK). No Korean hardcode."""
    reasons: List[str] = []
    be = detail.get("bearing_error_deg")
    if be is not None:
        reasons.append(f"bearing_err:{be:.1f}")
    if detail.get("L_aoa", 0) >= 0.7:
        reasons.append("aoa_high")
    elif detail.get("L_aoa", 0) < 0.3:
        reasons.append("aoa_low")
    if detail.get("L_rssi", 0) >= 0.6:
        reasons.append("rssi_match")
    if detail.get("L_los", 0) >= 0.8:
        reasons.append("terrain_los_ok")
    elif detail.get("L_los", 0) < 0.4:
        reasons.append("terrain_los_poor")
    if building.get("los_clear"):
        reasons.append("building_los_clear")
    h = building.get("height_m")
    if h and float(h) >= 15:
        reasons.append(f"height_m:{float(h):.0f}")
    elev = building.get("elevation_m") or building.get("ground_elevation_m")
    if elev is not None:
        reasons.append(f"elev_m:{float(elev):.0f}")
    if building.get("cell_tower_nearby"):
        reasons.append("tower_near")
    if building.get("dist_to_road_m") is not None and building["dist_to_road_m"] < 200:
        reasons.append("road_access")
    if building.get("building_type"):
        reasons.append(f"type:{building['building_type']}")
    if building.get("military"):
        reasons.append("military_tag")
    return reasons


def aggregate_candidate(
    building: Dict[str, Any],
    observer: Coord,
    bearing_deg: float,
    rssi_dbm: Optional[float],
    terrain_grid: Optional[Dict[str, Any]] = None,
    fix: Optional[Coord] = None,
    n_observers: int = 1,
) -> Dict[str, Any]:
    """단일 후보의 위치·고도·확률성분·이유를 집계."""
    lat = float(building["lat"])
    lng = float(building["lng"])
    heuristic, heuristic_parts = calculate_score(building, observer)
    precision, detail = candidate_precision_score(
        building, observer, bearing_deg, rssi_dbm, terrain_grid, fix
    )
    dist_km = distance_between_coords(observer, (lat, lng))
    brg = initial_bearing(observer, (lat, lng))
    radius = confidence_radius_m(
        detail.get("bearing_error_deg", 10.0),
        dist_km,
        n_observers,
        agent_agreement=0.5,
    )
    return {
        "id": building.get("id"),
        "osm_id": building.get("osm_id"),
        "source": building.get("source", "openstreetmap"),
        "name": building.get("name"),
        "location": {
            "latitude": lat,
            "longitude": lng,
            "bearing_from_observer_deg": round(brg, 2),
            "distance_km": round(dist_km, 3),
            "distance_m": round(dist_km * 1000.0, 1),
        },
        "elevation": {
            "ground_m": building.get("ground_elevation_m"),
            "roof_m": building.get("elevation_m"),
            "height_m": building.get("height_m"),
            "los_clear": building.get("los_clear"),
        },
        "attributes": {
            "building_type": building.get("building_type", "unknown"),
            "dist_to_road_m": building.get("dist_to_road_m"),
            "cell_tower_nearby": bool(building.get("cell_tower_nearby")),
            "military": bool(building.get("military")),
        },
        "scores": {
            "heuristic": round(heuristic, 4),
            "precision_likelihood": round(precision, 6),
            "components": {
                **{f"h_{k}": v for k, v in heuristic_parts.items()},
                **{f"p_{k}": v for k, v in detail.items()},
            },
        },
        "reasons": _reason_codes(detail, building),
        "confidence_radius_m": round(radius, 1),
        # carry-through for later stages
        "lat": lat,
        "lng": lng,
        "latitude": lat,
        "longitude": lng,
        "precision_score": precision,
        "heuristic_score": heuristic,
        "building_type": building.get("building_type", "unknown"),
        "height_m": building.get("height_m"),
        "ground_elevation_m": building.get("ground_elevation_m"),
        "elevation_m": building.get("elevation_m"),
    }


def aggregate_pool(
    buildings: List[Dict[str, Any]],
    observer: Coord,
    bearing_deg: float,
    rssi_dbm: Optional[float],
    terrain_grid: Optional[Dict[str, Any]] = None,
    fix: Optional[Coord] = None,
    n_observers: int = 1,
) -> List[Dict[str, Any]]:
    pool = [
        aggregate_candidate(
            b, observer, bearing_deg, rssi_dbm, terrain_grid, fix, n_observers
        )
        for b in buildings
    ]
    # normalize preliminary probability from precision mass
    total = sum(max(c["precision_score"], 0.0) for c in pool) or 1.0
    for c in pool:
        c["probability_raw"] = round(c["precision_score"] / total, 5)
        c["probability"] = c["probability_raw"]
    pool.sort(key=lambda x: x["probability"], reverse=True)
    return pool
