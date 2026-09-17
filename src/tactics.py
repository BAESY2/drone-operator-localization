"""방어용 위협·항로·패턴 분석 (폭발물/공격 수행 지침 없음)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.utils import add_distance_to_coord, distance_between_coords, initial_bearing

Coord = Tuple[float, float]


def _prob_color(p: float) -> str:
    """확률 → 지도 색상 (낮음=파랑 → 높음=빨강)."""
    p = max(0.0, min(1.0, p))
    if p >= 0.75:
        return "#ff2d2d"
    if p >= 0.5:
        return "#ff8c1a"
    if p >= 0.3:
        return "#f0c14b"
    if p >= 0.15:
        return "#5dade2"
    return "#7f8c8d"


def approach_corridor(
    observer: Coord,
    bearing_deg: float,
    range_km: float,
    half_width_deg: float = 7.0,
    steps: int = 20,
) -> Dict[str, Any]:
    """Approach sector polygon + woven (non-straight) center path."""
    centerline: List[List[float]] = []
    left: List[List[float]] = []
    right: List[List[float]] = []
    for i in range(steps + 1):
        t = i / steps
        d = max((range_km * i) / steps, 0.05)
        # S-curve lateral weave — drones rarely fly perfect radials
        weave_m = 55.0 * math.sin(t * math.pi * 2.2) * (0.2 + 0.8 * t)
        clat, clng = add_distance_to_coord(observer, bearing_deg, d)
        if abs(weave_m) > 0.5:
            clat, clng = add_distance_to_coord(
                (clat, clng), (bearing_deg + 90) % 360, weave_m / 1000.0
            )
        centerline.append([clng, clat])
        llat, llng = add_distance_to_coord(
            observer, (bearing_deg - half_width_deg) % 360, d
        )
        rlat, rlng = add_distance_to_coord(
            observer, (bearing_deg + half_width_deg) % 360, d
        )
        left.append([llng, llat])
        right.append([rlng, rlat])
    ring = left + list(reversed(right)) + [left[0]]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"kind": "corridor", "bearing_deg": bearing_deg},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            },
            {
                "type": "Feature",
                "properties": {
                    "kind": "centerline",
                    "woven": True,
                    "note": "Primary weave hypothesis (not radial straight)",
                },
                "geometry": {"type": "LineString", "coordinates": centerline},
            },
        ],
        "eta_minutes_estimate": {
            "slow_fpv_5mps": round((range_km * 1000) / 5 / 60, 1),
            "cruise_15mps": round((range_km * 1000) / 15 / 60, 1),
            "fast_30mps": round((range_km * 1000) / 30 / 60, 1),
        },
    }


def threat_impact_zones(
    observer: Coord,
    asset: Optional[Coord],
    primary: Optional[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    drone_range_km: float,
) -> Dict[str, Any]:
    """
    위협 영향권(방어 평가용):
    - self: 관측자/본인 위치 주변 위험 반경
    - asset: 보호 지점(지정 시)
    - departure: 추정 출발(조종) 지점 반경
    """
    focus = asset or observer
    # tighter observer watch ring (was up to 800m)
    self_r = min(320.0, max(80.0, drone_range_km * 35))
    zones = [
        {
            "id": "self_threat",
            "label": "OBS watch radius",
            "center": {"latitude": observer[0], "longitude": observer[1]},
            "radius_m": round(self_r, 1),
            "color": "#3d8bfd",
            "kind": "observer",
        }
    ]
    if asset and (
        abs(asset[0] - observer[0]) > 1e-6 or abs(asset[1] - observer[1]) > 1e-6
    ):
        zones.append(
            {
                "id": "asset_threat",
                "label": "Asset watch radius",
                "center": {"latitude": asset[0], "longitude": asset[1]},
                "radius_m": round(self_r * 1.05, 1),
                "color": "#9b59b6",
                "kind": "asset",
            }
        )

    departure_points = []
    for c in candidates[:5]:
        lat = c.get("latitude") or c.get("lat")
        lng = c.get("longitude") or c.get("lng")
        if lat is None or lng is None:
            continue
        p = float(c.get("probability") or 0)
        # prefer refined tight radius; never inflate with 200m default
        r = float(c.get("confidence_radius_m") or 80)
        r = max(12.0, min(r, 280.0))
        departure_points.append(
            {
                "id": c.get("id"),
                "rank": c.get("rank"),
                "label": f"DEP #{c.get('rank')} r={r:.0f}m ±{c.get('bearing_error_deg', '?')}°",
                "center": {"latitude": lat, "longitude": lng},
                "radius_m": round(r, 2),
                "probability": p,
                "probability_pct": c.get("probability_pct"),
                "bearing_error_deg": c.get("bearing_error_deg"),
                "distance_m": c.get("distance_m"),
                "ground_elevation_m": c.get("ground_elevation_m") or (c.get("elevation") or {}).get("ground_m"),
                "height_m": c.get("height_m"),
                "fix_quality": c.get("fix_quality"),
                "color": _prob_color(p),
                "kind": "departure",
                "reasons": c.get("reasons") or [],
            }
        )

    # predicted impact along approach toward asset/observer from primary departure
    predicted_impacts = []
    if primary:
        plat = primary.get("latitude") or primary.get("lat")
        plng = primary.get("longitude") or primary.get("lng")
        if plat is not None and plng is not None:
            inbound = initial_bearing((plat, plng), focus)
            # impact near protected point along last 300m of inbound
            impact = add_distance_to_coord(focus, (inbound + 180) % 360, 0.05)
            predicted_impacts.append(
                {
                    "id": "predicted_impact_primary",
                    "label": "Approach focus (defensive watch)",
                    "center": {"latitude": impact[0], "longitude": impact[1]},
                    "radius_m": 60,
                    "color": "#e74c3c",
                    "kind": "impact",
                    "inbound_bearing_deg": round(inbound, 2),
                    "from_departure": {"latitude": plat, "longitude": plng},
                    "to_focus": {"latitude": focus[0], "longitude": focus[1]},
                }
            )

    return {
        "observer_zones": zones,
        "departure_estimates": departure_points,
        "predicted_impacts": predicted_impacts,
        "legend": {
            "red": "p≥75%",
            "orange": "50–75%",
            "yellow": "30–50%",
            "blue": "15–30%",
            "gray": "<15%",
        },
        "disclaimer": "Defensive threat assessment only. Not munition or explosive guidance.",
    }


def classify_attack_pattern(
    observer: Coord,
    bearing_deg: float,
    range_km: float,
    signal_dbm: Optional[float],
    candidates: List[Dict[str, Any]],
    asset: Optional[Coord] = None,
) -> Dict[str, Any]:
    """기하·신호 기반 접근 패턴 분류 (방어 분석)."""
    focus = asset or observer
    patterns = []

    # direct inbound if bearing points roughly from sector toward focus
    patterns.append(
        {
            "name": "direct_approach",
            "score": 0.55,
            "description": "Inbound along observed bearing (primary hypothesis)",
        }
    )

    if signal_dbm is not None:
        if signal_dbm > -55:
            patterns.append(
                {
                    "name": "close_in_loiter",
                    "score": 0.7,
                    "description": "Strong RSSI → close-range loiter / near threat",
                }
            )
        elif signal_dbm < -75:
            patterns.append(
                {
                    "name": "standoff_control",
                    "score": 0.65,
                    "description": "Weak RSSI → standoff control pattern",
                }
            )

    if candidates:
        primary = candidates[0]
        plat = primary.get("latitude") or primary.get("lat")
        plng = primary.get("longitude") or primary.get("lng")
        if plat is not None and plng is not None:
            brg_to_focus = initial_bearing((plat, plng), focus)
            delta = abs((brg_to_focus - bearing_deg + 180) % 360 - 180)
            if delta < 25:
                patterns.append(
                    {
                        "name": "aligned_vector",
                        "score": 0.8,
                        "description": "Departure→focus vector aligns with observed bearing",
                    }
                )
            elif delta > 60:
                patterns.append(
                    {
                        "name": "flanking_vector",
                        "score": 0.6,
                        "description": "Flanking / offset approach (bearing mismatch)",
                    }
                )

            elev = (primary.get("elevation") or {}).get("height_m") or primary.get(
                "height_m"
            )
            if elev and float(elev) >= 20:
                patterns.append(
                    {
                        "name": "elevated_launch",
                        "score": 0.55,
                        "description": "Elevated / rooftop launch site hypothesis",
                    }
                )

    if range_km >= 8:
        patterns.append(
            {
                "name": "long_range_link",
                "score": 0.5,
                "description": "Long-range C2 link feasible for type/environment",
            }
        )

    patterns.sort(key=lambda x: x["score"], reverse=True)
    top = patterns[0] if patterns else {
        "name": "unknown",
        "score": 0.3,
        "description": "Insufficient data",
    }

    # simple timeline of phases
    timeline = [
        {"phase": "detect", "note": "방위/신호 관측"},
        {"phase": "localize", "note": "출발/조종 후보 축소"},
        {"phase": "track_corridor", "note": "접근 항로 감시"},
        {"phase": "impact_watch", "note": "보호지점 영향권 경계"},
        {"phase": "egress", "note": "이탈 방향 확보"},
    ]

    return {
        "primary_pattern": top,
        "patterns": patterns[:6],
        "timeline": timeline,
        "recommended_watch_bearings": [
            int(bearing_deg % 360),
            int((bearing_deg - 30) % 360),
            int((bearing_deg + 30) % 360),
        ],
        "escape_bearing_deg": int((bearing_deg + 180) % 360),
    }


def colorize_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in candidates:
        item = dict(c)
        p = float(c.get("probability") or 0)
        item["color"] = _prob_color(p)
        item["map_style"] = {
            "color": _prob_color(p),
            "fillColor": _prob_color(p),
            "radius": 6 + int(10 * p),
            "fillOpacity": 0.35 + 0.5 * p,
        }
        out.append(item)
    return out


def egress_corridors(
    origin: Coord,
    approach_bearing_deg: float,
    range_km: float = 2.5,
) -> Dict[str, Any]:
    """Defensive egress fan opposite approach — thick drawable paths."""
    axis = (approach_bearing_deg + 180.0) % 360.0
    features = []
    colors = ["#ffcc00", "#e6b800", "#c9a000"]
    for i, off in enumerate((-30.0, 0.0, 30.0)):
        brg = (axis + off) % 360.0
        line = []
        steps = 14
        for s in range(steps + 1):
            d = (range_km * s) / steps
            lat, lng = add_distance_to_coord(origin, brg, d)
            line.append([lng, lat])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "egress",
                    "bearing_deg": round(brg, 2),
                    "color": colors[i],
                    "label": f"EGRESS {brg:.0f}°",
                },
                "geometry": {"type": "LineString", "coordinates": line},
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "egress_axis_deg": round(axis, 2),
    }


def approach_path_detailed(
    observer: Coord,
    bearing_deg: float,
    range_km: float,
    elev_samples: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Non-straight approach sketch: slight weave + altitude samples along path."""
    pts = []
    steps = 18
    for i in range(steps + 1):
        t = i / steps
        d = max(range_km * t, 0.02)
        # soft S-curve lateral offset (meters → km) — not always radial straight
        weave_m = 45.0 * math.sin(t * math.pi * 2.0) * (0.3 + 0.7 * t)
        lat, lng = add_distance_to_coord(observer, bearing_deg, d)
        if abs(weave_m) > 1:
            lat, lng = add_distance_to_coord(
                (lat, lng), (bearing_deg + 90) % 360, weave_m / 1000.0
            )
        elev = None
        if elev_samples and i < len(elev_samples):
            elev = elev_samples[i]
        pts.append(
            {
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "dist_km": round(d, 3),
                "agl_m": elev,
            }
        )
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "kind": "approach_path",
                    "color": "#ff3b3b",
                    "label": "Predicted inbound path",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p["lng"], p["lat"]] for p in pts],
                },
            }
        ],
        "samples": pts,
    }


def build_tactical_layers(
    observer: Coord,
    bearing_deg: float,
    drone_range_km: float,
    signal_dbm: Optional[float],
    candidates: List[Dict[str, Any]],
    primary: Optional[Dict[str, Any]],
    asset: Optional[Coord] = None,
) -> Dict[str, Any]:
    colored = colorize_candidates(candidates)
    span = min(drone_range_km, 4.0)
    return {
        "approach": approach_corridor(observer, bearing_deg, span),
        "approach_path": {"type": "FeatureCollection", "features": []},  # superseded by path_hypotheses
        "egress": {"type": "FeatureCollection", "features": [], "note": "use path_hypotheses egress"},
        "threat": threat_impact_zones(
            observer, asset, primary, colored, drone_range_km
        ),
        "attack_pattern": classify_attack_pattern(
            observer, bearing_deg, drone_range_km, signal_dbm, colored, asset
        ),
        "candidates_styled": colored,
        "layer_legend": [
            {"id": "approach", "color": "#e05a4a", "label": "Approach corridor"},
            {"id": "path_groups", "color": "#ff3b3b", "label": "G1/G2/G3 dodge + egress"},
            {"id": "dep", "color": "#ff8c1a", "label": "Departure radii"},
            {"id": "contacts", "color": "#f0c14b", "label": "Candidates"},
        ],
        "probability_legend": [
            {"min": 0.75, "color": "#ff2d2d", "label": "very high"},
            {"min": 0.5, "color": "#ff8c1a", "label": "high"},
            {"min": 0.3, "color": "#f0c14b", "label": "mid"},
            {"min": 0.15, "color": "#5dade2", "label": "low"},
            {"min": 0.0, "color": "#7f8c8d", "label": "very low"},
        ],
    }
