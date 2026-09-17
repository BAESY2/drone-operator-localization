"""초정밀 후보 축소: 삼각측량·가능도·신뢰반경."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.rf_models import (
    bearing_delta,
    bearing_likelihood,
    constants,
    rssi_distance_likelihood,
)
from src.utils import (
    add_distance_to_coord,
    distance_between_coords,
    initial_bearing,
    normalize_bearing,
)

Coord = Tuple[float, float]
Observer = Dict[str, Any]


def local_en_m(origin: Coord, point: Coord) -> Tuple[float, float]:
    """원점 기준 east/north (m)."""
    lat0 = math.radians(origin[0])
    d_north = (point[0] - origin[0]) * 111_000.0
    d_east = (point[1] - origin[1]) * 111_000.0 * math.cos(lat0)
    return d_east, d_north


def en_to_latlng(origin: Coord, east_m: float, north_m: float) -> Coord:
    lat0 = math.radians(origin[0])
    return (
        origin[0] + north_m / 111_000.0,
        origin[1] + east_m / (111_000.0 * math.cos(lat0)),
    )


def intersect_bearings_fixed(
    obs_a: Coord,
    bearing_a_deg: float,
    obs_b: Coord,
    bearing_b_deg: float,
) -> Optional[Coord]:
    ax, ay = 0.0, 0.0
    bx, by = local_en_m(obs_a, obs_b)
    ra = math.radians(bearing_a_deg)
    rb = math.radians(bearing_b_deg)
    dax, day = math.sin(ra), math.cos(ra)
    dbx, dby = math.sin(rb), math.cos(rb)
    denom = dax * dby - day * dbx
    if abs(denom) < 1e-9:
        return None
    t = ((bx - ax) * dby - (by - ay) * dbx) / denom
    u = ((bx - ax) * day - (by - ay) * dax) / denom
    if t < 50 or u < 50:  # at least 50m along each ray
        return None
    return en_to_latlng(obs_a, t * dax, t * day)


def triangulation_fix(
    primary: Observer,
    others: List[Observer],
) -> Dict[str, Any]:
    """다중 관측자 교점 클러스터."""
    origin = (float(primary["latitude"]), float(primary["longitude"]))
    bearing0 = float(primary["bearing_degrees"])
    points: List[Coord] = []
    for other in others:
        hit = intersect_bearings_fixed(
            origin,
            bearing0,
            (float(other["latitude"]), float(other["longitude"])),
            float(other["bearing_degrees"]),
        )
        if hit:
            points.append(hit)
    if not points:
        return {"fix": None, "points": [], "method": "asa_triangulation"}
    lat = sum(p[0] for p in points) / len(points)
    lng = sum(p[1] for p in points) / len(points)
    spread = 0.0
    if len(points) > 1:
        spread = max(distance_between_coords((lat, lng), p) for p in points) * 1000.0
    return {
        "fix": {"latitude": lat, "longitude": lng},
        "points": [{"latitude": p[0], "longitude": p[1]} for p in points],
        "spread_m": round(spread, 1),
        "method": "asa_triangulation",
        "paper_ref": "dronet2019",
    }


def terrain_los_score(
    building: Dict[str, Any],
    observer: Coord,
    terrain_grid: Optional[Dict[str, Any]],
) -> float:
    """고도 그리드 샘플로 간이 LOS (관측점→후보 사이 장애물)."""
    if not terrain_grid or not terrain_grid.get("points"):
        return 0.7 if building.get("los_clear", True) else 0.25
    b_elev = float(building.get("elevation_m") or building.get("ground_elevation_m") or 0)
    path = []
    for p in terrain_grid["points"]:
        # keep points roughly between observer and building
        d_obs = distance_between_coords(observer, (p["lat"], p["lng"]))
        d_b = distance_between_coords((building["lat"], building["lng"]), (p["lat"], p["lng"]))
        d_total = distance_between_coords(observer, (building["lat"], building["lng"]))
        if d_obs + d_b < d_total * 1.25:
            path.append(float(p["elevation_m"]))
    if not path:
        return 0.7 if building.get("los_clear", True) else 0.3
    max_mid = max(path)
    # clear if building roof above mid-path terrain
    if b_elev >= max_mid - 5:
        return 1.0
    if b_elev >= max_mid - 20:
        return 0.55
    return 0.2


def candidate_precision_score(
    building: Dict[str, Any],
    observer: Coord,
    bearing_deg: float,
    rssi_dbm: Optional[float],
    terrain_grid: Optional[Dict[str, Any]] = None,
    fix: Optional[Coord] = None,
    aoa_sigma: float | None = None,
) -> Tuple[float, Dict[str, float]]:
    """가능도 곱 → 로그합으로 안정화."""
    lat, lng = float(building["lat"]), float(building["lng"])
    brg = initial_bearing(observer, (lat, lng))
    delta = bearing_delta(bearing_deg, brg)
    L_aoa = bearing_likelihood(delta, aoa_sigma)
    dist_km = distance_between_coords(observer, (lat, lng))
    L_rssi = rssi_distance_likelihood(dist_km, rssi_dbm) if rssi_dbm is not None else 0.6
    L_los = terrain_los_score(building, observer, terrain_grid)
    L_fix = 1.0
    if fix is not None:
        d_fix = distance_between_coords(fix, (lat, lng)) * 1000.0
        # DRONET ~11m with 2 stations; soft prior 150m for OSM snap
        L_fix = math.exp(-0.5 * (d_fix / 80.0) ** 2)

    # height prior: rooftop operators
    h = float(building.get("height_m") or 10)
    L_height = min(1.0, 0.4 + h / 40.0)

    score = L_aoa * L_rssi * L_los * L_fix * L_height
    return score, {
        "L_aoa": round(L_aoa, 4),
        "L_rssi": round(L_rssi, 4),
        "L_los": round(L_los, 4),
        "L_fix": round(L_fix, 4),
        "L_height": round(L_height, 4),
        "bearing_error_deg": round(delta, 2),
        "distance_km": round(dist_km, 3),
    }


def confidence_radius_m(
    bearing_error_deg: float,
    distance_km: float,
    n_observers: int,
    agent_agreement: float,
) -> float:
    """
    Lateral uncertainty ≈ d * tan(σ_eff), then agreement shrink.
    Tight caps: single 18–280m, multi 12–90m (was up to 2500m).
    """
    sigma = constants()["aoa_sigma_deg_single"]
    # use residual error, floored by instrument noise * 0.35
    sigma_eff = max(float(bearing_error_deg), sigma * 0.35)
    lateral = distance_km * 1000.0 * math.tan(math.radians(sigma_eff))
    # agreement shrink (0.35..1) → stronger agreement = smaller radius
    agree = max(0.4, min(1.0, float(agent_agreement) or 0.5))
    lateral *= 0.55 + 0.45 * (1.0 - agree)  # high agree → ~0.55×
    if n_observers >= 2:
        lateral *= 0.45
        return float(max(12.0, min(lateral, 90.0)))
    if n_observers >= 3:
        lateral *= 0.35
        return float(max(10.0, min(lateral, 55.0)))
    return float(max(18.0, min(lateral, 280.0)))


def narrow_candidates(
    candidates: List[Dict[str, Any]],
    observer: Coord,
    bearing_deg: float,
    rssi_dbm: Optional[float],
    terrain_grid: Optional[Dict[str, Any]],
    fix: Optional[Coord],
    top_n: int = 28,
    aoa_sigma: float | None = None,
) -> List[Dict[str, Any]]:
    scored = []
    for c in candidates:
        s, detail = candidate_precision_score(
            c, observer, bearing_deg, rssi_dbm, terrain_grid, fix, aoa_sigma
        )
        item = dict(c)
        item["precision_score"] = s
        item["precision"] = detail
        scored.append(item)
    scored.sort(key=lambda x: x["precision_score"], reverse=True)
    if not scored:
        return []
    best = scored[0]["precision_score"] or 1e-12
    # tighter competitive band (was 0.08)
    kept = [c for c in scored if c["precision_score"] >= best * 0.22][:top_n]
    return kept or scored[: min(top_n, len(scored))]
