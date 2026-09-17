"""
Recon compression — cover max probability mass with min flight distance.

Greedy set-cover / angular-sector clustering of top contacts into 3–5
waypoints, then build a small_quad GeoJSON route:
  OBS → high-mass corridors → orbit clusters → RTB

Goal: reduce ISR loss from visiting every candidate separately.
Wire-ready; parent may call after verify_and_prune.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils import add_distance_to_coord, distance_between_coords, initial_bearing

ROOT = Path(__file__).resolve().parent.parent
Coord = Tuple[float, float]


def _load_const() -> Dict[str, Any]:
    path = ROOT / "data" / "recon_constants.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _latlng(c: Dict[str, Any]) -> Coord:
    return (
        float(c.get("latitude") or c.get("lat") or 0),
        float(c.get("longitude") or c.get("lng") or 0),
    )


def _prob(c: Dict[str, Any]) -> float:
    return float(c.get("probability") or 0)


def _sector_index(bearing_deg: float, n_sectors: int) -> int:
    w = 360.0 / max(n_sectors, 1)
    return int(bearing_deg % 360 // w) % n_sectors


def _centroid(members: List[Dict[str, Any]]) -> Coord:
    if not members:
        return (0.0, 0.0)
    wsum = sum(max(_prob(c), 1e-9) for c in members)
    lat = sum(_latlng(c)[0] * max(_prob(c), 1e-9) for c in members) / wsum
    lng = sum(_latlng(c)[1] * max(_prob(c), 1e-9) for c in members) / wsum
    return (lat, lng)


def _cluster_radius_m(members: List[Dict[str, Any]], center: Coord) -> float:
    if len(members) <= 1:
        return 120.0
    dists = [
        distance_between_coords(center, _latlng(c)) * 1000.0 for c in members
    ]
    return max(120.0, min(450.0, sorted(dists)[int(0.75 * (len(dists) - 1))] * 1.15))


def compress_recon_waypoints(
    candidates: List[Dict[str, Any]],
    observer: Coord,
    *,
    min_wp: int = 3,
    max_wp: int = 5,
    n_sectors: int = 8,
    cover_mass: float = 0.82,
    merge_radius_m: float = 280.0,
) -> Dict[str, Any]:
    """
    Compress top contacts into 3–5 recon waypoints.

    Algorithm:
      1) Rank by probability; take prefix until cover_mass (or all).
      2) Bin into angular sectors from OBS.
      3) Greedy: repeatedly pick uncovered sector/cluster with best
         mass / (1 + distance_from_obs_or_last) until min_wp..max_wp.
      4) Merge spatially close centroids (merge_radius_m).

    Returns waypoints with covered candidate ids + mass stats.
    """
    if not candidates:
        return {
            "waypoints": [],
            "covered_mass": 0.0,
            "n_candidates": 0,
            "notes": ["empty_candidates"],
        }

    ranked = sorted(candidates, key=_prob, reverse=True)
    total_mass = sum(_prob(c) for c in ranked) or 1.0
    selected: List[Dict[str, Any]] = []
    cum = 0.0
    for c in ranked:
        selected.append(c)
        cum += _prob(c)
        if cum / total_mass >= cover_mass and len(selected) >= min_wp:
            break

    # Sector buckets
    buckets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_sectors)}
    for c in selected:
        brg = initial_bearing(observer, _latlng(c))
        buckets[_sector_index(brg, n_sectors)].append(c)

    # Seed clusters = non-empty sectors (or spatial seeds if sparse)
    seeds: List[Dict[str, Any]] = []
    for si, members in buckets.items():
        if not members:
            continue
        center = _centroid(members)
        mass = sum(_prob(c) for c in members)
        seeds.append(
            {
                "sector": si,
                "members": members,
                "center": center,
                "mass": mass,
                "dist_km": distance_between_coords(observer, center),
            }
        )
    if not seeds:
        # fallback: top individuals
        for c in selected[:max_wp]:
            center = _latlng(c)
            seeds.append(
                {
                    "sector": _sector_index(initial_bearing(observer, center), n_sectors),
                    "members": [c],
                    "center": center,
                    "mass": _prob(c),
                    "dist_km": distance_between_coords(observer, center),
                }
            )

    # Greedy set cover: maximize mass / (1+dist) among remaining, preferring
    # diversity of sectors and low path growth from last pick.
    uncovered = {id(s): s for s in seeds}
    picked: List[Dict[str, Any]] = []
    last: Optional[Coord] = observer
    used_sectors: set = set()

    while uncovered and len(picked) < max_wp:
        best_id = None
        best_score = -1.0
        for sid, s in uncovered.items():
            dist = distance_between_coords(last or observer, s["center"])
            sector_bonus = 1.25 if s["sector"] not in used_sectors else 1.0
            score = sector_bonus * (s["mass"] / total_mass) / (1.0 + dist)
            # early force diversity until min_wp
            if len(picked) < min_wp and s["sector"] not in used_sectors:
                score *= 1.15
            if score > best_score:
                best_score = score
                best_id = sid
        if best_id is None:
            break
        s = uncovered.pop(best_id)
        picked.append(s)
        used_sectors.add(s["sector"])
        last = s["center"]
        covered = sum(p["mass"] for p in picked)
        if len(picked) >= min_wp and covered / total_mass >= cover_mass:
            break

    # Ensure at least min_wp if mass available
    while len(picked) < min_wp and uncovered:
        # pick nearest remaining heavy cluster
        sid, s = min(
            uncovered.items(),
            key=lambda kv: kv[1]["dist_km"] / (kv[1]["mass"] + 1e-9),
        )
        uncovered.pop(sid)
        picked.append(s)

    # Spatial merge of close centroids
    merged: List[Dict[str, Any]] = []
    for s in sorted(picked, key=lambda x: -x["mass"]):
        absorbed = False
        for m in merged:
            d_m = distance_between_coords(s["center"], m["center"]) * 1000.0
            if d_m <= merge_radius_m:
                m["members"] = m["members"] + s["members"]
                m["mass"] = sum(_prob(c) for c in m["members"])
                m["center"] = _centroid(m["members"])
                m["sectors"] = list(set(m.get("sectors", [m.get("sector")]) + [s["sector"]]))
                absorbed = True
                break
        if not absorbed:
            item = dict(s)
            item["sectors"] = [s["sector"]]
            merged.append(item)

    # Cap at max_wp by mass
    merged = sorted(merged, key=lambda x: -x["mass"])[:max_wp]

    # Order by angular proximity along approach for short flight
    if merged:
        # nearest-neighbor from OBS
        ordered: List[Dict[str, Any]] = []
        remaining = list(merged)
        cur = observer
        while remaining:
            i = min(
                range(len(remaining)),
                key=lambda j: distance_between_coords(cur, remaining[j]["center"]),
            )
            nxt = remaining.pop(i)
            ordered.append(nxt)
            cur = nxt["center"]
        merged = ordered

    waypoints: List[Dict[str, Any]] = []
    covered_ids: List[Any] = []
    for i, m in enumerate(merged, start=1):
        center = m["center"]
        r_m = _cluster_radius_m(m["members"], center)
        ids = [c.get("id") for c in m["members"]]
        covered_ids.extend(ids)
        waypoints.append(
            {
                "index": i,
                "lat": round(center[0], 6),
                "lng": round(center[1], 6),
                "kind": "recon_cluster",
                "orbit_radius_m": round(r_m, 1),
                "mass": round(m["mass"], 6),
                "mass_frac": round(m["mass"] / total_mass, 4),
                "member_ids": ids,
                "member_count": len(m["members"]),
                "sectors": m.get("sectors") or [m.get("sector")],
                "bearing_from_obs_deg": round(initial_bearing(observer, center), 2),
                "dist_from_obs_km": round(
                    distance_between_coords(observer, center), 4
                ),
            }
        )

    covered_mass = sum(w["mass"] for w in waypoints)
    # flight length estimate (OBS → wps → OBS)
    path_km = 0.0
    prev = observer
    for w in waypoints:
        path_km += distance_between_coords(prev, (w["lat"], w["lng"]))
        prev = (w["lat"], w["lng"])
    path_km += distance_between_coords(prev, observer)

    naive_km = 0.0
    prev = observer
    for c in selected:
        naive_km += distance_between_coords(prev, _latlng(c))
        prev = _latlng(c)
    naive_km += distance_between_coords(prev, observer)

    return {
        "waypoints": waypoints,
        "covered_mass": round(covered_mass, 6),
        "covered_mass_frac": round(covered_mass / total_mass, 4),
        "n_candidates_considered": len(selected),
        "n_candidates": len(candidates),
        "covered_ids": covered_ids,
        "flight_distance_km": round(path_km, 4),
        "naive_visit_all_km": round(naive_km, 4),
        "distance_saving_frac": round(
            1.0 - path_km / max(naive_km, 1e-6), 4
        ),
        "notes": [
            f"compressed {len(selected)} contacts → {len(waypoints)} waypoints",
            f"mass_frac={covered_mass / total_mass:.3f}",
            f"saving≈{(1.0 - path_km / max(naive_km, 1e-6)):.2%}",
        ],
    }


def _alt_from_weather(weather: Optional[Dict[str, Any]], base_alt: float, max_alt: float) -> float:
    wx = weather or {}
    vis_km = wx.get("visibility_km")
    if vis_km is None and wx.get("visibility_m") is not None:
        vis_km = float(wx["visibility_m"]) / 1000.0
    wind = float(wx.get("wind_mps") or 0)
    gust = float(wx.get("gust_mps") or wind)
    alt = base_alt
    if vis_km is not None:
        if vis_km < 2.0:
            alt = min(alt, 70.0)
        elif vis_km < 4.0:
            alt = min(alt, 95.0)
        elif vis_km > 12.0:
            alt = min(max_alt, alt + 15.0)
    if gust >= 12:
        alt = min(alt, 85.0)
    elif gust >= 8:
        alt = min(alt, 110.0)
    return max(40.0, min(max_alt, alt))


def _orbit_ring(center: Coord, radius_m: float, samples: int = 8) -> List[List[float]]:
    r_km = radius_m / 1000.0
    return [
        [
            add_distance_to_coord(center, (360.0 * i) / samples, r_km)[1],
            add_distance_to_coord(center, (360.0 * i) / samples, r_km)[0],
        ]
        for i in range(samples + 1)
    ]


def build_compressed_recon_route(
    observer: Coord,
    bearing_deg: float,
    candidates: List[Dict[str, Any]],
    *,
    airframe: str = "small_quad",
    weather: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    recon_profile: str = "compressed_mass",
    min_wp: int = 3,
    max_wp: int = 5,
) -> Dict[str, Any]:
    """
    Auto recon route GeoJSON for small_quad:
      approach from OBS along high-mass corridors → orbit clusters → RTB.
    Altitude from weather/visibility.
    """
    ctx = context or {}
    wx = weather or ctx.get("environment") or ctx.get("weather") or {}
    cnst = _load_const()
    af = cnst["airframe_presets"].get(airframe) or cnst["airframe_presets"]["small_quad"]
    base_alt = float(
        (cnst["profiles"].get("orbit_contacts") or {}).get("default_alt_m") or 100
    )
    cruise = float(af["cruise_mps"])
    max_alt = float(af["max_alt_m"])
    alt = _alt_from_weather(wx, base_alt, max_alt)
    spd = min(cruise, 12.0)

    pack = compress_recon_waypoints(
        candidates,
        observer,
        min_wp=min_wp,
        max_wp=max_wp,
    )
    wps = pack["waypoints"]

    features: List[Dict[str, Any]] = []
    path_coords: List[List[float]] = [[observer[1], observer[0]]]
    mission_wps: List[Dict[str, Any]] = [
        {
            "lat": round(observer[0], 6),
            "lng": round(observer[1], 6),
            "alt_m": int(alt),
            "speed_mps": spd,
            "loiter_s": 0,
            "kind": "home",
            "label": "OBS",
        }
    ]

    # Approach corridor stubs toward high-mass bearings (not every candidate)
    for w in wps:
        brg = float(w["bearing_from_obs_deg"])
        # intermediate corridor point at 40% of cluster distance
        mid_km = float(w["dist_from_obs_km"]) * 0.4
        if mid_km > 0.15:
            mlat, mlng = add_distance_to_coord(observer, brg, mid_km)
            mission_wps.append(
                {
                    "lat": round(mlat, 6),
                    "lng": round(mlng, 6),
                    "alt_m": int(alt),
                    "speed_mps": spd,
                    "loiter_s": 0,
                    "kind": "corridor",
                    "label": f"CORR-{w['index']}",
                    "bearing_deg": brg,
                }
            )
            path_coords.append([mlng, mlat])

        # cluster center + orbit
        clat, clng = float(w["lat"]), float(w["lng"])
        mission_wps.append(
            {
                "lat": clat,
                "lng": clng,
                "alt_m": int(alt),
                "speed_mps": round(spd * 0.65, 1),
                "loiter_s": 40,
                "kind": "orbit_cluster",
                "label": f"ORB-{w['index']}",
                "mass_frac": w["mass_frac"],
                "member_ids": w["member_ids"],
                "orbit_radius_m": w["orbit_radius_m"],
            }
        )
        path_coords.append([clng, clat])
        ring = _orbit_ring((clat, clng), float(w["orbit_radius_m"]), 8)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "orbit",
                    "label": f"ORB-{w['index']}",
                    "color": "#7cff6b",
                    "mass_frac": w["mass_frac"],
                    "member_ids": w["member_ids"],
                },
                "geometry": {"type": "LineString", "coordinates": ring},
            }
        )

    # RTB
    mission_wps.append(
        {
            "lat": round(observer[0], 6),
            "lng": round(observer[1], 6),
            "alt_m": int(alt * 0.9),
            "speed_mps": spd,
            "loiter_s": 0,
            "kind": "rtb",
            "label": "RTB",
        }
    )
    path_coords.append([observer[1], observer[0]])

    features.insert(
        0,
        {
            "type": "Feature",
            "properties": {
                "kind": "primary_path",
                "color": "#00e5ff",
                "profile": recon_profile,
                "airframe": airframe,
                "note": "Compressed mass-cover recon — not per-candidate ISR",
            },
            "geometry": {"type": "LineString", "coordinates": path_coords},
        }
    )

    # High-mass corridor fan wedges as optional reference
    if wps:
        span_km = max(float(w["dist_from_obs_km"]) for w in wps)
        for w in wps[:3]:
            brg = float(w["bearing_from_obs_deg"])
            tip = add_distance_to_coord(observer, brg, span_km)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "kind": "mass_corridor",
                        "color": "#00bcd4",
                        "bearing_deg": brg,
                        "mass_frac": w["mass_frac"],
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [observer[1], observer[0]],
                            [tip[1], tip[0]],
                        ],
                    },
                }
            )

    length_m = 0.0
    for i in range(1, len(mission_wps)):
        a = (mission_wps[i - 1]["lat"], mission_wps[i - 1]["lng"])
        b = (mission_wps[i]["lat"], mission_wps[i]["lng"])
        length_m += distance_between_coords(a, b) * 1000.0

    eta_s = 0.0
    for i in range(1, len(mission_wps)):
        a = (mission_wps[i - 1]["lat"], mission_wps[i - 1]["lng"])
        b = (mission_wps[i]["lat"], mission_wps[i]["lng"])
        dist = distance_between_coords(a, b) * 1000.0
        v = max(float(mission_wps[i]["speed_mps"]), 1.0)
        eta_s += dist / v + float(mission_wps[i].get("loiter_s") or 0)

    return {
        "success": True,
        "profile": recon_profile,
        "airframe": airframe,
        "goal": "reduce_isr_loss_via_mass_cover_min_distance",
        "settings": {
            "alt_m": int(alt),
            "speed_mps": round(spd, 1),
            "alt_source": "weather_visibility_wind",
            "visibility_km": wx.get("visibility_km"),
            "wind_mps": wx.get("wind_mps"),
            "gust_mps": wx.get("gust_mps"),
            "approach_bearing_deg": round(float(bearing_deg) % 360, 2),
        },
        "compression": pack,
        "waypoints": mission_wps,
        "length_m": round(length_m, 1),
        "eta_min": round(eta_s / 60.0, 1),
        "geojson": {"type": "FeatureCollection", "features": features},
        "export": {
            "format": "waypoint_v1_compressed",
            "home": {"lat": observer[0], "lng": observer[1]},
            "points": mission_wps,
        },
    }
