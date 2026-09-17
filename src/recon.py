"""Defensive recon auto-routes from predicted approach / egress / contacts."""

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


def _wp(
    lat: float,
    lng: float,
    alt_m: float,
    kind: str,
    speed_mps: float,
    loiter_s: float = 0,
    label: str = "",
) -> Dict[str, Any]:
    return {
        "lat": round(lat, 6),
        "lng": round(lng, 6),
        "alt_m": int(alt_m),
        "speed_mps": round(speed_mps, 1),
        "loiter_s": int(loiter_s),
        "kind": kind,
        "label": label,
    }


def _sample_line(
    a: Coord,
    bearing_deg: float,
    length_km: float,
    spacing_m: float,
) -> List[Coord]:
    n = max(2, int((length_km * 1000) / max(spacing_m, 50)) + 1)
    pts: List[Coord] = []
    for i in range(n):
        d = (length_km * i) / (n - 1)
        pts.append(add_distance_to_coord(a, bearing_deg, d))
    return pts


def _orbit(center: Coord, radius_m: float, samples: int = 8) -> List[Coord]:
    r_km = radius_m / 1000.0
    return [
        add_distance_to_coord(center, (360.0 * i) / samples, r_km)
        for i in range(samples + 1)
    ]


def build_recon_plan(
    observer: Coord,
    bearing_deg: float,
    *,
    profile: str = "verify_approach",
    airframe: str = "small_quad",
    candidates: Optional[List[Dict[str, Any]]] = None,
    target: Optional[Coord] = None,
    range_km: float = 3.0,
    alt_m: Optional[float] = None,
    speed_mps: Optional[float] = None,
    mission_outcome: str = "unknown",
) -> Dict[str, Any]:
    """
    Deterministic waypoint plan.
    Profiles:
      verify_approach — corridor toward threat axis + loiter contacts
      track_egress — opposite axis (egress) + exit samples
      orbit_contacts — orbits on top candidates
    """
    cnst = _load_const()
    profiles = cnst["profiles"]
    if profile not in profiles:
        profile = "verify_approach"
    p = profiles[profile]
    af = cnst["airframe_presets"].get(airframe) or cnst["airframe_presets"]["small_quad"]

    alt = float(alt_m if alt_m is not None else p["default_alt_m"])
    spd = float(speed_mps if speed_mps is not None else p["default_speed_mps"])
    alt = min(alt, float(af["max_alt_m"]))
    spd = min(spd, float(af["cruise_mps"]) * 1.2)
    spacing = float(p.get("waypoint_spacing_m") or 400)
    loiter = float(p.get("loiter_s") or 30)
    max_range = min(float(range_km), float(af["max_range_km"]))

    waypoints: List[Dict[str, Any]] = []
    path_coords: List[List[float]] = []  # [lng, lat] geojson order
    features: List[Dict[str, Any]] = []

    # home / start
    waypoints.append(_wp(observer[0], observer[1], alt, "home", spd, 0, "HOME"))
    path_coords.append([observer[1], observer[0]])

    cands = candidates or []

    if profile == "verify_approach":
        # fly along observed threat bearing (approach corridor)
        line = _sample_line(observer, bearing_deg, max_range, spacing)
        for i, (lat, lng) in enumerate(line[1:], start=1):
            waypoints.append(
                _wp(lat, lng, alt, "corridor", spd, 0, f"APR-{i}")
            )
            path_coords.append([lng, lat])
        # loiter top contacts
        for i, c in enumerate(cands[:3]):
            lat, lng = float(c["latitude"]), float(c["longitude"])
            waypoints.append(
                _wp(lat, lng, alt, "loiter_contact", spd * 0.6, loiter, f"C{i+1}")
            )
            path_coords.append([lng, lat])
            orb = _orbit((lat, lng), float(p.get("orbit_radius_m") or 150), 6)
            ring = [[o[1], o[0]] for o in orb]
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "kind": "orbit",
                        "color": p["color"],
                        "label": f"C{i+1}",
                    },
                    "geometry": {"type": "LineString", "coordinates": ring},
                }
            )

    elif profile == "track_egress":
        egress_brg = (bearing_deg + 180) % 360
        # urgency: failure → shorter first hop fan
        length = max_range * (0.55 if mission_outcome == "failure" else 0.85)
        fans = [egress_brg, (egress_brg - 25) % 360, (egress_brg + 25) % 360]
        for fi, brg in enumerate(fans):
            line = _sample_line(observer, brg, length * (1.0 - fi * 0.15), spacing)
            branch: List[List[float]] = [[observer[1], observer[0]]]
            for i, (lat, lng) in enumerate(line[1:], start=1):
                label = f"EG-{fi+1}-{i}"
                waypoints.append(
                    _wp(
                        lat,
                        lng,
                        alt,
                        "egress",
                        spd,
                        loiter if i == len(line) - 1 else 0,
                        label,
                    )
                )
                branch.append([lng, lat])
                if fi == 0:
                    path_coords.append([lng, lat])
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "kind": "egress_branch",
                        "color": p["color"],
                        "bearing_deg": round(brg, 1),
                    },
                    "geometry": {"type": "LineString", "coordinates": branch},
                }
            )
        if target:
            waypoints.append(
                _wp(target[0], target[1], alt, "watch_target", spd, loiter, "TGT")
            )

    else:  # orbit_contacts
        if not cands:
            # fallback point along bearing
            lat, lng = add_distance_to_coord(observer, bearing_deg, min(1.2, max_range))
            cands = [{"latitude": lat, "longitude": lng, "rank": 1}]
        r = float(p.get("orbit_radius_m") or 180)
        for i, c in enumerate(cands[:4]):
            lat, lng = float(c["latitude"]), float(c["longitude"])
            waypoints.append(
                _wp(lat, lng, alt, "transit", spd, 0, f"TO-C{i+1}")
            )
            path_coords.append([lng, lat])
            orb = _orbit((lat, lng), r, 8)
            for j, (olat, olng) in enumerate(orb):
                waypoints.append(
                    _wp(
                        olat,
                        olng,
                        alt,
                        "orbit",
                        spd * 0.7,
                        loiter if j == len(orb) - 1 else 0,
                        f"ORB{i+1}-{j}",
                    )
                )
            ring = [[o[1], o[0]] for o in orb]
            features.append(
                {
                    "type": "Feature",
                    "properties": {"kind": "orbit", "color": p["color"]},
                    "geometry": {"type": "LineString", "coordinates": ring},
                }
            )

    # RTB
    waypoints.append(
        _wp(observer[0], observer[1], alt * 0.9, "rtb", spd, 0, "RTB")
    )
    path_coords.append([observer[1], observer[0]])

    length_m = 0.0
    for i in range(1, len(waypoints)):
        a = (waypoints[i - 1]["lat"], waypoints[i - 1]["lng"])
        b = (waypoints[i]["lat"], waypoints[i]["lng"])
        length_m += distance_between_coords(a, b) * 1000.0

    eta_s = 0.0
    for i in range(1, len(waypoints)):
        a = (waypoints[i - 1]["lat"], waypoints[i - 1]["lng"])
        b = (waypoints[i]["lat"], waypoints[i]["lng"])
        dist = distance_between_coords(a, b) * 1000.0
        v = max(float(waypoints[i]["speed_mps"]), 1.0)
        eta_s += dist / v + float(waypoints[i].get("loiter_s") or 0)

    features.insert(
        0,
        {
            "type": "Feature",
            "properties": {
                "kind": "primary_path",
                "color": p["color"],
                "dash": p.get("dash"),
                "profile": profile,
            },
            "geometry": {"type": "LineString", "coordinates": path_coords},
        },
    )

    return {
        "success": True,
        "profile": profile,
        "label": p.get("label"),
        "note": p.get("note"),
        "color": p["color"],
        "dash": p.get("dash"),
        "airframe": airframe,
        "settings": {
            "alt_m": int(alt),
            "speed_mps": round(spd, 1),
            "loiter_s": int(loiter),
            "spacing_m": int(spacing),
            "camera": "nadir_eo",
            "mission_outcome": mission_outcome,
        },
        "waypoints": waypoints,
        "length_m": round(length_m, 1),
        "eta_min": round(eta_s / 60.0, 1),
        "geojson": {"type": "FeatureCollection", "features": features},
        "export": {
            "format": "waypoint_v1",
            "home": {"lat": observer[0], "lng": observer[1]},
            "points": waypoints,
        },
    }
