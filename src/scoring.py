"""후보 점수 함수 — AoA/RSSI/중층 우선, 최고층·군사 태그 편향 최소화."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SCORING_WEIGHTS
from src.accuracy_model import (
    bearing_fit_score,
    mid_elevation_preference,
    rssi_range_consistency,
)
from src.utils import distance_between_coords, initial_bearing


def assess_tactical_position(
    building: Dict[str, Any],
    center_pos: Tuple[float, float],
) -> float:
    """관측점과의 거리 기반 전술 가치 (0~1). 1–8 km 유리."""
    lat = building.get("lat")
    lng = building.get("lng")
    if lat is None or lng is None:
        return 0.5
    dist_km = distance_between_coords(center_pos, (lat, lng))
    if 1.0 <= dist_km <= 8.0:
        return 1.0
    if dist_km < 1.0:
        return 0.4
    return max(0.2, 1.0 - (dist_km - 8.0) / 20.0)


def calculate_score(
    building: Dict[str, Any],
    center_pos: Tuple[float, float],
    *,
    approach_bearing: Optional[float] = None,
    rssi_dbm: Optional[float] = None,
    bearing_tol_deg: float = 18.0,
) -> Tuple[float, Dict[str, float]]:
    """한 후보에 점수 부여. Returns (score, reasoning)."""
    score = 0.0
    reasoning: Dict[str, float] = {}

    lat = building.get("lat")
    lng = building.get("lng")
    dist_km = None
    brg_err = building.get("bearing_error_deg")
    if lat is not None and lng is not None:
        dist_km = distance_between_coords(center_pos, (float(lat), float(lng)))
        if brg_err is None and approach_bearing is not None:
            brg = initial_bearing(center_pos, (float(lat), float(lng)))
            brg_err = abs((brg - float(approach_bearing) + 180) % 360 - 180)

    # AoA fit (primary)
    bf = bearing_fit_score(brg_err, bearing_tol_deg)
    bearing_term = bf * SCORING_WEIGHTS.get("bearing_fit", 0.2)
    reasoning["bearing_fit"] = round(bearing_term, 3)
    score += bearing_term

    # RSSI ↔ range consistency
    rr = rssi_range_consistency(rssi_dbm, dist_km)
    rssi_term = rr * SCORING_WEIGHTS.get("rssi_range", 0.15)
    reasoning["rssi_range"] = round(rssi_term, 3)
    score += rssi_term

    # Mid elevation (NOT tallest)
    height_m = building.get("height_m")
    elev_pref = mid_elevation_preference(
        float(height_m) if height_m is not None else None
    )
    elevation_score = elev_pref * SCORING_WEIGHTS["elevation"]
    reasoning["elevation"] = round(elevation_score, 3)
    score += elevation_score

    los_clear = building.get("los_clear", True)
    los_score = (
        SCORING_WEIGHTS["los"] if los_clear else SCORING_WEIGHTS["los"] * 0.3
    )
    reasoning["los"] = round(los_score, 3)
    score += los_score

    # Tiny military tag — not dominant
    military_score = (
        SCORING_WEIGHTS["military_history"] if building.get("military") else 0.0
    )
    reasoning["military"] = round(military_score, 3)
    score += military_score

    tactical_score = (
        assess_tactical_position(building, center_pos) * SCORING_WEIGHTS["tactical"]
    )
    reasoning["tactical"] = round(tactical_score, 3)
    score += tactical_score

    dist_to_road = float(building.get("dist_to_road_m", 500) or 500)
    # Sweet spot 40–350 m (egress access)
    if 40 <= dist_to_road <= 350:
        road_raw = 1.0
    elif dist_to_road < 40:
        road_raw = 0.55
    else:
        road_raw = max(0.15, 1.0 - (dist_to_road - 350) / 2000.0)
    road_score = road_raw * SCORING_WEIGHTS["road"]
    reasoning["road"] = round(road_score, 3)
    score += road_score

    building_type = building.get("building_type", "unknown")
    if building_type == "residential":
        building_type_score = SCORING_WEIGHTS["building_type"]
    elif building_type == "commercial":
        building_type_score = SCORING_WEIGHTS["building_type"] * 0.55
    else:
        building_type_score = SCORING_WEIGHTS["building_type"] * 0.25
    reasoning["building_type"] = round(building_type_score, 3)
    score += building_type_score

    cell_tower = bool(building.get("cell_tower_nearby", False))
    cell_score = SCORING_WEIGHTS["cell_tower"] if cell_tower else 0.0
    reasoning["cell_tower"] = round(cell_score, 3)
    score += cell_score

    return score, reasoning


def score_all_candidates(
    candidates: List[Dict[str, Any]],
    center_pos: Tuple[float, float],
    num_results: int = 10,
    *,
    approach_bearing: Optional[float] = None,
    rssi_dbm: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """모든 후보 점수 매기고 확률 정규화 후 상위 N 반환."""
    scored: List[Dict[str, Any]] = []

    for idx, building in enumerate(candidates):
        score, reasoning = calculate_score(
            building,
            center_pos,
            approach_bearing=approach_bearing,
            rssi_dbm=rssi_dbm,
        )
        lat = building.get("lat", 0)
        lng = building.get("lng", 0)
        scored.append(
            {
                "rank": 0,
                "lat": lat,
                "lng": lng,
                "latitude": lat,
                "longitude": lng,
                "score": score,
                "probability": 0.0,
                "reasoning": reasoning,
                "building_type": building.get("building_type", "unknown"),
                "name": building.get("name"),
                "height_m": building.get("height_m"),
                "ground_elevation_m": building.get("ground_elevation_m"),
                "elevation_m": building.get("elevation_m"),
                "source": building.get("source", "openstreetmap"),
                "osm_id": building.get("osm_id"),
                "confidence_radius_m": 120,
                "id": building.get("id", f"cand_{idx}"),
            }
        )

    if not scored:
        return []

    max_score = max(s["score"] for s in scored) or 1.0
    for s in scored:
        s["probability"] = round(s["score"] / max_score, 3)

    ranked = sorted(scored, key=lambda x: x["probability"], reverse=True)[
        :num_results
    ]
    for i, item in enumerate(ranked, start=1):
        item["rank"] = i
    return ranked
