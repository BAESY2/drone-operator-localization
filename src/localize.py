"""부채꼴 + OSM → 집계 → 경쟁 에이전트 → 재검토 축소."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BEARING_TOLERANCE, DEFAULT_RANGE_MIN_KM
from src.aggregate import aggregate_pool
from src.agents import run_competing_agents
from src.kinematics import extract_kinematics_features
from src.precision import narrow_candidates, triangulation_fix
from src.providers.context import fetch_geospatial_context
from src.human_ops import algorithmic_shortlist, apply_human_behavior_prior
from src.refine import progressive_refine
from src.review import rereview_candidates
from src.rf_models import constants as research_constants
from src.terrain_filter import filter_candidates
from src.utils import add_distance_to_coord


def create_search_sector(
    center_lat: float,
    center_lng: float,
    bearing_deg: float,
    range_km: float,
    bearing_tolerance_deg: float = DEFAULT_BEARING_TOLERANCE,
    num_samples: int = 64,
    radius_min_km: float = DEFAULT_RANGE_MIN_KM,
) -> Dict[str, Any]:
    bearing_min = (bearing_deg - bearing_tolerance_deg) % 360
    bearing_max = (bearing_deg + bearing_tolerance_deg) % 360
    bearing_range = bearing_max - bearing_min
    if bearing_range < 0:
        bearing_range += 360

    outer_arc: List[List[float]] = []
    inner_arc: List[List[float]] = []
    for i in range(num_samples + 1):
        bearing = (bearing_min + (bearing_range * i / num_samples)) % 360
        lat_max, lng_max = add_distance_to_coord(
            (center_lat, center_lng), bearing, range_km
        )
        outer_arc.append([lng_max, lat_max])
        lat_min, lng_min = add_distance_to_coord(
            (center_lat, center_lng), bearing, radius_min_km
        )
        inner_arc.append([lng_min, lat_min])
    inner_arc.reverse()
    return {
        "type": "Polygon",
        "coordinates": [outer_arc + inner_arc + [outer_arc[0]]],
    }


def _to_api_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    loc = c.get("location") or {}
    elev = c.get("elevation") or {}
    return {
        "rank": c.get("rank"),
        "id": c.get("id"),
        "osm_id": c.get("osm_id"),
        "source": c.get("source"),
        "name": c.get("name"),
        "latitude": loc.get("latitude", c.get("latitude")),
        "longitude": loc.get("longitude", c.get("longitude")),
        "lat": loc.get("latitude", c.get("lat")),
        "lng": loc.get("longitude", c.get("lng")),
        "probability": c.get("probability"),
        "probability_pct": c.get("probability_pct"),
        "confidence_radius_m": c.get("confidence_radius_m"),
        "bearing_error_deg": c.get("bearing_error_deg"),
        "distance_m": c.get("distance_m"),
        "distance_km": c.get("distance_km"),
        "fix_quality": c.get("fix_quality"),
        "location": c.get("location"),
        "elevation": elev,
        "attributes": c.get("attributes"),
        "reasons": c.get("reasons"),
        "scores": c.get("scores"),
        "human_score": c.get("human_score"),
        "human_parts": c.get("human_parts"),
        "human_reasons": c.get("human_reasons"),
        "behavior_fusion": c.get("behavior_fusion"),
        "agent_votes": c.get("agent_votes"),
        "agent_agreement": c.get("agent_agreement"),
        "cluster_size": c.get("cluster_size", 1),
        "summary": c.get("summary"),
        "building_type": c.get("building_type"),
        "height_m": elev.get("height_m") or c.get("height_m"),
        "ground_elevation_m": elev.get("ground_m") or c.get("ground_elevation_m"),
        "elevation_m": elev.get("roof_m") or c.get("elevation_m"),
    }


def localize_drone_operator(
    center_lat: float,
    center_lng: float,
    bearing_deg: float,
    drone_range_km: float = 10,
    num_results: int = 8,
    bearing_tolerance_deg: float = DEFAULT_BEARING_TOLERANCE,
    signal_strength_dbm: Optional[float] = -65,
    additional_observers: Optional[List[Dict[str, Any]]] = None,
    flight_track: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    파이프라인:
      OSM 수집 → 부채꼴 필터 → 정밀 가능도 축소
      → 확률/고도/위치/이유 집계 → 병렬 에이전트 경쟁(+궤적)
      → 재검토로 최종 후보지군 축소
    """
    observer = (center_lat, center_lng)
    sector = create_search_sector(
        center_lat,
        center_lng,
        bearing_deg,
        drone_range_km,
        bearing_tolerance_deg=bearing_tolerance_deg,
    )
    geo = fetch_geospatial_context(
        observer, bearing_deg, drone_range_km, bearing_tolerance_deg
    )

    n_obs = 1 + len(additional_observers or [])
    fix_info = {"fix": None, "points": [], "method": None}
    if additional_observers:
        fix_info = triangulation_fix(
            {
                "latitude": center_lat,
                "longitude": center_lng,
                "bearing_degrees": bearing_deg,
            },
            additional_observers,
        )
    fix = None
    if fix_info.get("fix"):
        fix = (
            float(fix_info["fix"]["latitude"]),
            float(fix_info["fix"]["longitude"]),
        )

    kin_features = None
    if flight_track:
        kin_features = extract_kinematics_features(flight_track)

    filtered = filter_candidates(
        sector,
        geo["candidates"],
        center=observer,
        bearing_deg=bearing_deg,
        range_km=max(drone_range_km, geo["query_range_km"]),
        bearing_tolerance_deg=bearing_tolerance_deg,
        num_results=220,
    )

    narrowed = narrow_candidates(
        filtered,
        observer,
        bearing_deg,
        signal_strength_dbm,
        geo.get("terrain_grid"),
        fix,
        top_n=research_constants().get("max_candidates_precision", 80),
    )

    aggregated = aggregate_pool(
        narrowed,
        observer,
        bearing_deg,
        signal_strength_dbm,
        geo.get("terrain_grid"),
        fix,
        n_observers=n_obs,
    )

    contest = run_competing_agents(
        aggregated,
        observer,
        bearing_deg,
        signal_strength_dbm,
        fix,
        kinematics_features=kin_features,
        range_km=float(drone_range_km or 5.0),
    )

    # Human-operator causal prior (concealment / mid-elev / egress / standoff)
    # — independent of any external AI API.
    humanized = apply_human_behavior_prior(
        contest["pool"],
        observer,
        bearing_deg,
        float(drone_range_km or 5.0),
    )

    review = rereview_candidates(
        humanized, observer, max_keep=max(num_results * 3, 48)
    )

    # Wide formula pool then algorithmic shortlist (AI optional later).
    pool_cap = max(num_results, 36)
    refined = progressive_refine(
        review["final"],
        observer,
        bearing_deg,
        rssi_dbm=signal_strength_dbm,
        n_observers=n_obs,
        max_keep=pool_cap,
    )
    formula_pool = [_to_api_candidate(c) for c in refined["candidates"]]
    short = algorithmic_shortlist(
        formula_pool, max_keep=max(num_results, 16), min_sep_m=38.0
    )
    final = short["shortlist"]

    return {
        "search_sector": sector,
        "primary_target": final[0] if final else None,
        "top_10_candidates": final,
        "formula_pool": formula_pool,
        "formula_pool_count": len(formula_pool),
        "algorithm_shortlist": {
            "count": short["count"],
            "notes": short["notes"],
            "method": "human_ops_causal_v1",
        },
        "aggregated_count": len(aggregated),
        "pre_review_count": len(contest["pool"]),
        "filtered_count": len(filtered),
        "narrowed_count": len(narrowed),
        "refined_count": refined["after_count"],
        "osm_building_count": len(geo["candidates"]),
        "terrain_grid": geo["terrain_grid"],
        "imagery": geo["imagery"],
        "bbox": geo["bbox"],
        "sources": geo["sources"],
        "attribution": geo["attribution"],
        "query_range_km": geo["query_range_km"],
        "triangulation": fix_info,
        "kinematics": kin_features,
        "departure_tight": refined.get("departure_tight"),
        "agents": {
            "names": contest["agent_names"],
            "winning_agent": contest["winning_agent"],
            "rankings_top3": {
                name: ranks[:3] for name, ranks in contest["rankings"].items()
            },
        },
        "review": {
            "before_count": review["before_count"],
            "after_count": review["after_count"],
            "rejected_count": review["rejected_count"],
            "notes": review["review_notes"],
            "rejected_sample": review["rejected"][:10],
        },
        "refine": {
            "before_count": refined["before_count"],
            "after_count": refined["after_count"],
            "rejected_count": refined["rejected_count"],
            "notes": refined["notes"],
            "aoa_limit_deg": refined.get("aoa_limit_deg"),
            "rejected_sample": refined["rejected"][:12],
        },
        "human_ops": {
            "engine": "causal_behavior_v1",
            "ai_required": False,
            "factors": [
                "rf_geometry",
                "concealment",
                "mid_elevation",
                "egress",
                "standoff",
                "los_usable",
                "cluster",
            ],
        },
    }
