"""후보 점수 함수."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SCORING_WEIGHTS
from src.utils import distance_between_coords


def assess_tactical_position(
    building: Dict[str, Any],
    center_pos: Tuple[float, float],
) -> float:
    """관측점과의 거리 기반 전술 가치 (0~1)."""
    lat = building.get("lat")
    lng = building.get("lng")
    if lat is None or lng is None:
        return 0.5
    dist_km = distance_between_coords(center_pos, (lat, lng))
    # 1~8km 구간이 상대적으로 유리
    if 1.0 <= dist_km <= 8.0:
        return 1.0
    if dist_km < 1.0:
        return 0.4
    return max(0.2, 1.0 - (dist_km - 8.0) / 20.0)


def calculate_score(
    building: Dict[str, Any],
    center_pos: Tuple[float, float],
) -> Tuple[float, Dict[str, float]]:
    """한 후보에 점수 부여. Returns (score, reasoning)."""
    score = 0.0
    reasoning: Dict[str, float] = {}

    elevation_m = float(building.get("elevation_m", 0) or 0)
    elevation_percentile = min(elevation_m / 500.0, 1.0)
    elevation_score = elevation_percentile * SCORING_WEIGHTS["elevation"]
    reasoning["elevation"] = round(elevation_score, 3)
    score += elevation_score

    los_clear = building.get("los_clear", True)
    los_score = (
        SCORING_WEIGHTS["los"] if los_clear else SCORING_WEIGHTS["los"] * 0.3
    )
    reasoning["los"] = round(los_score, 3)
    score += los_score

    if building.get("military"):
        military_score = SCORING_WEIGHTS["military_history"]
    else:
        military_score = 0.0
    reasoning["military"] = round(military_score, 3)
    score += military_score

    tactical_score = (
        assess_tactical_position(building, center_pos) * SCORING_WEIGHTS["tactical"]
    )
    reasoning["tactical"] = round(tactical_score, 3)
    score += tactical_score

    dist_to_road = float(building.get("dist_to_road_m", 500) or 500)
    road_score = max(SCORING_WEIGHTS["road"] - (dist_to_road / 10000.0), 0.0)
    reasoning["road"] = round(road_score, 3)
    score += road_score

    building_type = building.get("building_type", "unknown")
    if building_type == "residential":
        building_type_score = SCORING_WEIGHTS["building_type"]
    elif building_type == "commercial":
        building_type_score = SCORING_WEIGHTS["building_type"] * 0.5
    else:
        building_type_score = SCORING_WEIGHTS["building_type"] * 0.2
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
) -> List[Dict[str, Any]]:
    """모든 후보 점수 매기고 확률 정규화 후 상위 N 반환."""
    scored: List[Dict[str, Any]] = []

    for idx, building in enumerate(candidates):
        score, reasoning = calculate_score(building, center_pos)
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
                "confidence_radius_m": 450,
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
