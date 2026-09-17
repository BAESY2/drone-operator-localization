"""
BGU Mashhadi et al. (CSCML 2020) 영감 — 비행궤적 피처 & 조종자 prior.

논문: 위치 궤적만으로 POV 분류 ~73% (시뮬). RF 불필요.
우리는 연속 공간 prior로 가공해 OSM 후보에 가중한다.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.utils import distance_between_coords, initial_bearing

Coord = Tuple[float, float]
TrackPoint = Dict[str, Any]


def _as_track(points: List[TrackPoint]) -> List[Dict[str, float]]:
    out = []
    for i, p in enumerate(points):
        try:
            out.append(
                {
                    "t": float(p.get("t", i)),
                    "lat": float(p["lat"] if "lat" in p else p["latitude"]),
                    "lng": float(p["lng"] if "lng" in p else p["longitude"]),
                    "alt_m": float(p.get("alt_m", p.get("altitude_m", 0)) or 0),
                    "yaw_deg": (
                        float(p["yaw_deg"])
                        if p.get("yaw_deg") is not None
                        else (
                            float(p["yaw"])
                            if p.get("yaw") is not None
                            else float("nan")
                        )
                    ),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda x: x["t"])
    return out


def extract_kinematics_features(track: List[TrackPoint]) -> Dict[str, Any]:
    """수직상승·하강·속도·요·경로·공격성 지수."""
    pts = _as_track(track)
    n = len(pts)
    if n < 2:
        return {
            "sample_count": n,
            "usable": False,
            "confidence": 0.0,
            "reason": "need >= 2 track points",
        }

    speeds = []
    climbs = []
    yaw_rates = []
    headings = []
    path_len_m = 0.0

    for i in range(1, n):
        dt = max(pts[i]["t"] - pts[i - 1]["t"], 1e-3)
        d_km = distance_between_coords(
            (pts[i - 1]["lat"], pts[i - 1]["lng"]),
            (pts[i]["lat"], pts[i]["lng"]),
        )
        d_m = d_km * 1000.0
        path_len_m += d_m
        speeds.append(d_m / dt)
        climbs.append((pts[i]["alt_m"] - pts[i - 1]["alt_m"]) / dt)
        hdg = initial_bearing(
            (pts[i - 1]["lat"], pts[i - 1]["lng"]),
            (pts[i]["lat"], pts[i]["lng"]),
        )
        headings.append(hdg)
        y0, y1 = pts[i - 1]["yaw_deg"], pts[i]["yaw_deg"]
        if not (math.isnan(y0) or math.isnan(y1)):
            dy = abs((y1 - y0 + 180) % 360 - 180)
            yaw_rates.append(dy / dt)

    def _stat(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "abs_mean": 0.0}
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        return {
            "mean": round(m, 4),
            "std": round(math.sqrt(var), 4),
            "max": round(max(xs), 4),
            "abs_mean": round(sum(abs(x) for x in xs) / len(xs), 4),
        }

    speed_s = _stat(speeds)
    climb_s = _stat(climbs)
    yaw_s = _stat(yaw_rates)

    # heading persistence: low std → straight inbound
    hdg_persist = 1.0
    if len(headings) >= 2:
        # circular-ish approx via linear unwrap of deltas
        deltas = []
        for i in range(1, len(headings)):
            deltas.append(abs((headings[i] - headings[i - 1] + 180) % 360 - 180))
        hdg_persist = max(0.0, 1.0 - (sum(deltas) / len(deltas)) / 45.0)

    # aggression: high speed std + climb abs + yaw rate
    aggression = min(
        1.0,
        0.35 * min(speed_s["std"] / 8.0, 1.0)
        + 0.35 * min(climb_s["abs_mean"] / 3.0, 1.0)
        + 0.30 * min(yaw_s["abs_mean"] / 40.0, 1.0),
    )

    # LOS vs FPV heuristic (BGU qualitative finding → rule prior)
    # LOS: more hover/climb near visual axis, lower sustained high-speed dashes
    # FPV: longer path, higher mean speed, more yaw activity
    los_score = 0.5
    los_score += 0.15 if speed_s["mean"] < 8 else -0.1
    los_score += 0.15 if climb_s["abs_mean"] > 0.5 else -0.05
    los_score += 0.1 if yaw_s["abs_mean"] < 25 else -0.1
    los_score += 0.1 if hdg_persist > 0.55 else -0.05
    los_score = max(0.05, min(0.95, los_score))
    fpv_score = 1.0 - los_score
    if fpv_score > los_score + 0.08:
        pilot_mode = "fpv"
    elif los_score > fpv_score + 0.08:
        pilot_mode = "los"
    else:
        pilot_mode = "unknown"

    # sample confidence (paper: more samples → better; press mentioned up to 8+)
    conf = min(1.0, 0.35 + 0.65 * min(n / 120.0, 1.0))
    if n >= 8:
        conf = min(1.0, conf + 0.1)

    start, end = pts[0], pts[-1]
    net_bearing = initial_bearing((start["lat"], start["lng"]), (end["lat"], end["lng"]))
    net_dist_km = distance_between_coords(
        (start["lat"], start["lng"]), (end["lat"], end["lng"])
    )

    return {
        "sample_count": n,
        "usable": True,
        "confidence": round(conf, 3),
        "paper_ref": "bgu2020_mashhadi",
        "paper_accuracy_note": "BGU sim: 73% pos / 74% pos+orient / press ~78%. Heuristic prior only until MLP trained.",
        "speed_mps": speed_s,
        "climb_mps": climb_s,
        "yaw_rate_dps": yaw_s,
        "path_length_m": round(path_len_m, 1),
        "heading_persistence": round(hdg_persist, 3),
        "aggression_index": round(aggression, 3),
        "pilot_mode": pilot_mode,
        "pilot_mode_scores": {"los": round(los_score, 3), "fpv": round(fpv_score, 3)},
        "track_start": {"lat": start["lat"], "lng": start["lng"], "alt_m": start["alt_m"]},
        "track_end": {"lat": end["lat"], "lng": end["lng"], "alt_m": end["alt_m"]},
        "net_bearing_deg": round(net_bearing, 2),
        "net_distance_km": round(net_dist_km, 3),
        "centroid": {
            "lat": sum(p["lat"] for p in pts) / n,
            "lng": sum(p["lng"] for p in pts) / n,
            "alt_m": sum(p["alt_m"] for p in pts) / n,
        },
    }


def kinematics_candidate_weight(
    candidate: Dict[str, Any],
    features: Dict[str, Any],
    observer: Optional[Coord] = None,
) -> float:
    """
    후보 건물 가중 (0~1+).
    LOS: 궤적에 가시성 좋은 측(드론 중심 기준 관측 반대편 지상) + 고지.
    FPV: 순항 진입 후방/측면 거점.
    """
    if not features.get("usable"):
        return 0.5

    lat = float(candidate.get("lat") or candidate.get("latitude") or 0)
    lng = float(candidate.get("lng") or candidate.get("longitude") or 0)
    centroid = features["centroid"]
    c_lat, c_lng = centroid["lat"], centroid["lng"]

    # bearing from candidate(operator hyp) to drone centroid
    brg_op_to_drone = initial_bearing((lat, lng), (c_lat, c_lng))
    net = features["net_bearing_deg"]
    # alignment: operator looking along flight progress
    align = abs((brg_op_to_drone - net + 180) % 360 - 180)
    align_score = max(0.0, 1.0 - align / 90.0)

    dist_km = distance_between_coords((lat, lng), (c_lat, c_lng))
    # typical visual / FPV control standoff
    if 0.05 <= dist_km <= 2.5:
        dist_score = 1.0
    elif dist_km < 0.05:
        dist_score = 0.3
    else:
        dist_score = max(0.1, 1.0 - (dist_km - 2.5) / 5.0)

    elev = float(
        candidate.get("elevation_m")
        or (candidate.get("elevation") or {}).get("roof_m")
        or candidate.get("ground_elevation_m")
        or 0
    )
    elev_score = min(1.0, elev / 80.0) if elev else 0.4

    mode = features.get("pilot_mode", "unknown")
    if mode == "los":
        w = 0.45 * align_score + 0.25 * dist_score + 0.30 * elev_score
        # LOS prefers closer visual range
        if dist_km <= 1.2:
            w += 0.1
    elif mode == "fpv":
        # FPV: less need for direct visual align; prefer along approach rear
        rear = abs((brg_op_to_drone - (net + 180) % 360 + 180) % 360 - 180)
        rear_score = max(0.0, 1.0 - rear / 120.0)
        w = 0.35 * rear_score + 0.25 * dist_score + 0.20 * elev_score + 0.20 * align_score
    else:
        w = 0.4 * align_score + 0.3 * dist_score + 0.3 * elev_score

    # aggression boosts elevated / road-access sites slightly
    if features.get("aggression_index", 0) > 0.55:
        if (candidate.get("attributes") or {}).get("cell_tower_nearby") or candidate.get(
            "cell_tower_nearby"
        ):
            w += 0.05

    conf = float(features.get("confidence") or 0.5)
    return max(0.05, min(1.5, w)) * (0.5 + 0.5 * conf)


def apply_kinematics_prior(
    candidates: List[Dict[str, Any]],
    track: Optional[List[TrackPoint]],
    observer: Optional[Coord] = None,
) -> Dict[str, Any]:
    """후보 확률에 궤적 prior 곱셈 후 재정규화."""
    if not track:
        return {"features": None, "candidates": candidates, "applied": False}

    features = extract_kinematics_features(track)
    if not features.get("usable"):
        return {"features": features, "candidates": candidates, "applied": False}

    updated = []
    for c in candidates:
        item = dict(c)
        kw = kinematics_candidate_weight(item, features, observer)
        prior_p = float(item.get("probability") or item.get("probability_raw") or 0.01)
        item["kinematics_weight"] = round(kw, 4)
        item["probability"] = prior_p * kw
        reasons = list(item.get("reasons") or [])
        reasons.append(f"track_prior:{features['pilot_mode']}x{kw:.2f}")
        item["reasons"] = reasons
        updated.append(item)

    total = sum(float(c.get("probability") or 0) for c in updated) or 1.0
    for i, c in enumerate(sorted(updated, key=lambda x: x["probability"], reverse=True), 1):
        c["probability"] = round(float(c["probability"]) / total, 5)
        c["probability_pct"] = round(c["probability"] * 100.0, 2)
        c["rank"] = i
        if c.get("map_style"):
            from src.tactics import _prob_color

            col = _prob_color(c["probability"])
            c["color"] = col
            c["map_style"] = {
                **c["map_style"],
                "color": col,
                "fillColor": col,
            }
    updated.sort(key=lambda x: x["probability"], reverse=True)
    return {"features": features, "candidates": updated, "applied": True}
