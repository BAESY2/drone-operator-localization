"""
Realistic multi-hypothesis paths — NOT radial toy rays.

Inbound (drone): multi-leg urban dodge with LARGE lateral offsets + building
clearance reroutes (prefer gaps).

Egress (OP): STRICT OSM road Dijkstra per mode (foot / car / bike).
  - Only consecutive highway geometry edges (no soft node jumps).
  - Water barriers blocked unless OSM bridge tag.
  - No arc / cross-country fallback (those cross rivers & blocks).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.obstacle_paths import (
    avoid_buildings_polyline,
    classify_terrain_class,
    clearance_m_for_building,
)
from src.road_router import (
    EGRESS_MODE_ORDER,
    MODE_COLORS as _MODE_COLORS,
    MODE_DASH as _MODE_DASH,
    MODE_LABELS as _MODE_LABELS,
    MODE_TARGET_KM as _MODE_TARGET_KM,
    route_along_roads,
)
from src.utils import add_distance_to_coord, distance_between_coords, initial_bearing

ROOT = Path(__file__).resolve().parent.parent
Coord = Tuple[float, float]

EGRESS_MODES = EGRESS_MODE_ORDER
_MODE_STROKE = {
    "foot": 3.5,
    "bike": 3.8,
    "motorcycle": 4.2,
    "car": 5.0,
    "armored": 5.5,
}


def _groups() -> Dict[str, Any]:
    return json.loads((ROOT / "data" / "engine_constants.json").read_text(encoding="utf-8"))[
        "path_groups"
    ]


def _egress_cfg() -> Dict[str, Any]:
    path = ROOT / "data" / "egress_constants.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _feat(
    coords_lnglat: List[List[float]],
    *,
    kind: str,
    group: str,
    color: str,
    label: str,
    weight: float,
    note: str,
    stroke: float,
    mode: Optional[str] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
    family: str = "drone",
    dash: Optional[str] = None,
) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "kind": kind,
        "group": group,
        "family": family,  # "drone" | "egress"
        "color": color,
        "label": label,
        "weight": weight,
        "note": note,
        "stroke_weight": stroke,
        "samples": samples or [],
    }
    if dash:
        props["dash"] = dash
    if mode:
        props["mode"] = mode
    if extra:
        props.update(extra)
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": coords_lnglat},
    }


def _lerp(a: Coord, b: Coord, t: float) -> Coord:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _dense_polyline(waypoints: List[Coord], per_seg: int = 8, bulge_m: float = 18.0) -> List[Coord]:
    if len(waypoints) < 2:
        return waypoints
    out: List[Coord] = []
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        for s in range(per_seg):
            t = s / per_seg
            mid = _lerp(a, b, t)
            if 0.05 < t < 0.95 and bulge_m:
                brg = initial_bearing(a, b)
                bulge = bulge_m * math.sin(t * math.pi)
                mid = add_distance_to_coord(mid, (brg + 90) % 360, bulge / 1000.0)
            out.append(mid)
    out.append(waypoints[-1])
    return out


def _cand_coord(c: Dict[str, Any]) -> Optional[Coord]:
    lat = c.get("latitude") if c.get("latitude") is not None else c.get("lat")
    lng = c.get("longitude") if c.get("longitude") is not None else c.get("lng")
    if lat is None or lng is None:
        return None
    return (float(lat), float(lng))


def _buildings_from_sources(
    buildings: Optional[Sequence[Dict[str, Any]]],
    candidates: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for src in (buildings or [], candidates or []):
        for b in src:
            lat = b.get("lat") if b.get("lat") is not None else b.get("latitude")
            lng = b.get("lng") if b.get("lng") is not None else b.get("longitude")
            if lat is None or lng is None:
                continue
            key = (round(float(lat), 5), round(float(lng), 5))
            if key in seen:
                continue
            seen.add(key)
            item = dict(b)
            item["lat"] = float(lat)
            item["lng"] = float(lng)
            if item.get("height_m") is None:
                item["height_m"] = 12.0
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Inbound: multi-leg urban dodge
# ---------------------------------------------------------------------------


def urban_dodge_path(
    origin: Coord,
    dest: Coord,
    *,
    style: str = "g1",
    seed: int = 1,
    buildings: Optional[Sequence[Dict[str, Any]]] = None,
    min_clearance_m: float = 25.0,
) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """
    Multi-waypoint inbound with LARGE visible lateral offsets (cover / canyon dodge).
    Segments are rewritten via building keep-outs so paths prefer gaps.
    """
    dist = distance_between_coords(origin, dest)
    if dist < 0.08:
        return [], []
    brg = initial_bearing(origin, dest)
    blds = list(buildings or [])

    if style == "g1":
        n_legs, amp_frac, phase = 5, 0.14, 0.0
    elif style == "g2":
        n_legs, amp_frac, phase = 6, 0.20, 0.4
    else:
        n_legs, amp_frac, phase = 7, 0.28, 0.9

    # Visible lateral: min ~100 m, up to ~420 m on long runs
    amp_m = max(100.0, min(420.0, dist * 1000.0 * amp_frac))
    waypoints: List[Coord] = [origin]
    for i in range(1, n_legs):
        t = i / n_legs
        along = add_distance_to_coord(origin, brg, dist * t)
        side = 1 if (i + seed) % 2 == 0 else -1
        envelope = math.sin(t * math.pi) ** 0.85
        jitter = 0.35 * math.sin(t * math.pi * 3 + phase * seed)
        off_m = amp_m * side * (envelope + jitter)
        lat, lng = add_distance_to_coord(along, (brg + 90) % 360, off_m / 1000.0)
        waypoints.append((lat, lng))
    waypoints.append(dest)

    if blds:
        waypoints = avoid_buildings_polyline(waypoints, blds, min_clearance_m=min_clearance_m)

    dense = _dense_polyline(waypoints, per_seg=10, bulge_m=12.0)
    if blds:
        # Soft second pass on densified points (skip endpoints)
        dense = avoid_buildings_polyline(dense[::2] + [dense[-1]], blds, min_clearance_m=min_clearance_m)
        dense = _dense_polyline(dense, per_seg=4, bulge_m=6.0)

    samples: List[Dict[str, Any]] = []
    coords: List[List[float]] = []
    agl_base = {"g1": 45, "g2": 70, "g3": 110}.get(style, 50)
    for j, (lat, lng) in enumerate(dense):
        t = j / max(len(dense) - 1, 1)
        agl = agl_base * (0.45 + 0.55 * math.sin(t * math.pi))
        samples.append(
            {
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "agl_m": round(agl, 1),
                "t": round(t, 3),
            }
        )
        coords.append([round(lng, 6), round(lat, 6)])
    return samples, coords


# ---------------------------------------------------------------------------
# Strict OSM road egress (no soft edges / no river jumps / no arcs)
# ---------------------------------------------------------------------------


def road_egress_for_mode(
    origin: Coord,
    axis_bearing: float,
    segments: Sequence[Dict[str, Any]],
    *,
    mode: str = "foot",
    target_km: Optional[float] = None,
    buildings: Optional[Sequence[Dict[str, Any]]] = None,
    traffic: Optional[Dict[str, Any]] = None,
    bearing_offset_deg: float = 0.0,
    min_clearance_m: float = 25.0,
    water: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Strict OSM road Dijkstra only. No soft edges, no river jumps, no arcs.
    `buildings`/`traffic`/`min_clearance_m` kept for API compat — road geom is
    the source of truth (buildings sit beside roads; soft detours caused rivers).
    """
    del buildings, traffic, min_clearance_m
    return route_along_roads(
        origin,
        axis_bearing,
        list(segments),
        mode=mode,
        target_km=target_km,
        water=water,
        bearing_offset_deg=bearing_offset_deg,
        max_snap_m=90.0,
    )


def _fallback_arc_egress(
    origin: Coord,
    axis_bearing: float,
    range_km: float,
    *,
    buildings: Optional[Sequence[Dict[str, Any]]] = None,
    terrain_class: str = "open_plain",
    n: int = 2,
    min_clearance_m: float = 25.0,
) -> List[Dict[str, Any]]:
    """Disabled — arcs cross rivers/blocks. Prefer no route over a fake one."""
    del origin, axis_bearing, range_km, buildings, terrain_class, n, min_clearance_m
    return []


def egress_for_candidate(
    candidate: Dict[str, Any],
    approach_bearing: float,
    roads: Sequence[Dict[str, Any]],
    buildings: Sequence[Dict[str, Any]],
    terrain: Optional[Dict[str, Any]] = None,
    traffic: Optional[Dict[str, Any]] = None,
    *,
    water: Optional[Sequence[Dict[str, Any]]] = None,
    min_clearance_m: float = 25.0,
    modes: Sequence[str] = EGRESS_MODES,
    emit_features: bool = True,
) -> Dict[str, Any]:
    """
    Per-candidate mode-specific egress (foot / bike / motorcycle / car / armored).

    Features are family=egress for UI layer split from drone inbound paths.
    """
    origin = _cand_coord(candidate)
    if not origin:
        return {
            "origin": None,
            "routes": {},
            "feasibility": {"overall": 0.0},
            "features": [],
            "note": "candidate missing coordinates",
        }

    axis = (float(approach_bearing) + 180.0) % 360.0
    raw_blds = _buildings_from_sources(list(buildings), None)
    blds = [
        b
        for b in raw_blds
        if distance_between_coords(origin, (float(b["lat"]), float(b["lng"]))) * 1000.0 > 35.0
    ]
    tclass = classify_terrain_class(terrain, raw_blds, origin)
    rank = candidate.get("rank") or candidate.get("id") or "?"

    routes: Dict[str, Any] = {}
    features: List[Dict[str, Any]] = []
    feas: Dict[str, float] = {}

    mode_offsets = {
        "foot": (-18.0, 12.0),
        "bike": (-28.0, 8.0),
        "motorcycle": (-15.0, 18.0),
        "car": (-12.0, 22.0),
        "armored": (-25.0, 10.0),
    }

    for mode in modes:
        if mode not in EGRESS_MODES:
            continue
        target = _MODE_TARGET_KM[mode]
        best: Optional[Dict[str, Any]] = None
        for off in mode_offsets.get(mode, (0.0,)):
            rt = road_egress_for_mode(
                origin,
                axis,
                roads,
                mode=mode,
                target_km=target,
                buildings=blds,
                traffic=traffic,
                bearing_offset_deg=off,
                min_clearance_m=min_clearance_m,
                water=water,
            )
            if rt and (best is None or rt["feasibility"] > best["feasibility"]):
                best = rt

        routes[mode] = best
        feas[mode] = float((best or {}).get("feasibility") or 0.0)

        if emit_features and best and best.get("coords"):
            label_short = _MODE_LABELS.get(mode, mode.upper())
            features.append(
                _feat(
                    best["coords"],
                    kind=f"egress_{mode}",
                    group=f"eg_{mode}_{rank}",
                    color=_MODE_COLORS[mode],
                    label=f"EGRESS · {label_short}",
                    weight=round(feas[mode], 3),
                    note=(
                        f"{label_short} · {best.get('method')} · "
                        f"{best.get('length_km')} km · η≈{best.get('eta_min')} min · "
                        f"terrain={tclass}"
                    ),
                    stroke=_MODE_STROKE[mode],
                    mode=mode,
                    family="egress",
                    dash=_MODE_DASH.get(mode),
                    extra={
                        "feasibility": feas[mode],
                        "length_km": best.get("length_km"),
                        "method": best.get("method"),
                        "snap_m": best.get("snap_m"),
                        "mode_label": label_short,
                    },
                )
            )

    # Overall: mix of civilian + military mobility options
    overall = 0.0
    if feas:
        overall = (
            0.22 * feas.get("foot", 0.0)
            + 0.18 * feas.get("bike", 0.0)
            + 0.18 * feas.get("motorcycle", 0.0)
            + 0.24 * feas.get("car", 0.0)
            + 0.18 * feas.get("armored", 0.0)
        )

    return {
        "origin": {"lat": origin[0], "lng": origin[1]},
        "axis_deg": round(axis, 2),
        "terrain_class": tclass,
        "routes": routes,
        "feasibility": {**{k: round(v, 3) for k, v in feas.items()}, "overall": round(overall, 3)},
        "features": features,
        "water_barrier_count": len(water or []),
        "clearance_note": (
            "Strict OSM road Dijkstra; water only on bridge edges; "
            "modes: foot/bike/moto/car/mil"
        ),
    }


# ---------------------------------------------------------------------------
# Contact inbound + build_path_groups
# ---------------------------------------------------------------------------


def build_contact_inbound_curves(
    observer: Coord,
    candidates: List[Dict[str, Any]],
    max_n: int = 1,
    *,
    buildings: Optional[Sequence[Dict[str, Any]]] = None,
    min_clearance_m: float = 25.0,
) -> Dict[str, Any]:
    """Primary contact→OBS drone path only (keeps map readable)."""
    colors = ["#ff2d2d", "#ff8c1a", "#f0c14b"]
    styles = ["g1", "g2", "g3"]
    features = []
    blds = _buildings_from_sources(buildings, candidates)
    for i, c in enumerate(candidates[:max_n]):
        origin = _cand_coord(c)
        if not origin:
            continue
        samples, coords = urban_dodge_path(
            origin,
            observer,
            style=styles[i % 3],
            seed=i + 1,
            buildings=blds,
            min_clearance_m=min_clearance_m,
        )
        if not coords:
            continue
        features.append(
            _feat(
                coords,
                kind="drone_contact",
                group=f"c{i+1}",
                color=colors[i % 3],
                label=f"DRONE · CONTACT→OBS #{c.get('rank') or i+1}",
                weight=round(0.5 - 0.08 * i, 3),
                note="Drone flight hypothesis (contact→OBS)",
                stroke=5.5 if i == 0 else 4.0,
                samples=samples,
                family="drone",
                dash="6 4" if i else None,
            )
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "method": "urban_dodge_inbound_v3",
    }


def build_path_groups(
    observer: Coord,
    bearing_deg: float,
    range_km: float,
    *,
    primary: Optional[Dict[str, Any]] = None,
    weather_flags: Optional[List[str]] = None,
    road_segments: Optional[List[Dict[str, Any]]] = None,
    buildings: Optional[List[Dict[str, Any]]] = None,
    terrain: Optional[Dict[str, Any]] = None,
    traffic: Optional[Dict[str, Any]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    water: Optional[Sequence[Dict[str, Any]]] = None,
    min_clearance_m: float = 25.0,
) -> Dict[str, Any]:
    """
    Split path families for clean UI:
      family=drone  → G1/G2/G3 (+ optional primary contact inbound)
      family=egress → primary OP only, modes foot/bike/moto/car/mil
    """
    cfg = _groups()
    eg_cfg = _egress_cfg()
    layer_colors = (eg_cfg.get("layer_colors") or {})
    span = max(0.8, min(range_km, 4.0))
    w1, w2, w3 = cfg["g1"]["weight"], cfg["g2"]["weight"], cfg["g3"]["weight"]
    if weather_flags and "low_visibility" in weather_flags:
        w1, w2, w3 = 0.42, 0.28, 0.30
    tw = w1 + w2 + w3
    w1, w2, w3 = w1 / tw, w2 / tw, w3 / tw

    blds = _buildings_from_sources(buildings, candidates)
    drone_features: List[Dict[str, Any]] = []
    egress_features: List[Dict[str, Any]] = []
    samples: Dict[str, Any] = {}

    dests = []
    for gid, off in (
        ("g1", 0.0),
        ("g2", float(cfg["g2"]["bearing_offset_deg"])),
        ("g3", float(cfg["g3"]["bearing_offset_deg"])),
    ):
        brg = (bearing_deg + off) % 360.0
        r = span * (0.95 if gid == "g1" else (0.85 if gid == "g2" else 1.05))
        dests.append((gid, add_distance_to_coord(observer, brg, r)))

    weights = {"g1": w1, "g2": w2, "g3": w3}
    drone_dash = {"g1": None, "g2": "10 5", "g3": "4 5"}
    for i, (gid, dest) in enumerate(dests):
        g = cfg[gid]
        samp, coords = urban_dodge_path(
            observer,
            dest,
            style=gid,
            seed=i + 2,
            buildings=blds,
            min_clearance_m=min_clearance_m,
        )
        samples[gid] = samp
        drone_features.append(
            _feat(
                coords,
                kind="drone_inbound",
                group=gid,
                color=g["color"],
                label=f"DRONE · {gid.upper()}",
                weight=round(weights[gid], 3),
                note="Drone approach hypothesis · " + g["note"],
                stroke=7 if gid == "g1" else (5.5 if gid == "g2" else 4.0),
                samples=samp,
                family="drone",
                dash=drone_dash.get(gid),
            )
        )

    # Only primary contact→OBS (avoid stacking 3 contact curves on G1–G3)
    if candidates:
        ci = build_contact_inbound_curves(
            observer,
            candidates,
            max_n=1,
            buildings=blds,
            min_clearance_m=min_clearance_m,
        )
        drone_features.extend(ci.get("features") or [])

    egress_origin_cand = primary or (candidates[0] if candidates else None)
    if egress_origin_cand is None:
        egress_origin_cand = {
            "latitude": observer[0],
            "longitude": observer[1],
            "rank": "obs",
        }

    egress_pack = egress_for_candidate(
        egress_origin_cand,
        bearing_deg,
        road_segments or [],
        blds,
        terrain=terrain,
        traffic=traffic,
        water=water,
        min_clearance_m=min_clearance_m,
        emit_features=True,
    )
    egress_features.extend(egress_pack.get("features") or [])

    # Score other top contacts for metrics — no extra map lines
    per_candidate: List[Dict[str, Any]] = [egress_pack]
    for c in (candidates or [])[1:3]:
        pack = egress_for_candidate(
            c,
            bearing_deg,
            road_segments or [],
            blds,
            terrain=terrain,
            traffic=traffic,
            water=water,
            min_clearance_m=min_clearance_m,
            emit_features=False,
        )
        per_candidate.append(pack)

    axis = egress_pack.get("axis_deg", (bearing_deg + 180.0) % 360.0)
    methods = []
    for mode, rt in (egress_pack.get("routes") or {}).items():
        if rt:
            methods.append(f"{mode}:{rt.get('method')}")

    all_features = drone_features + egress_features
    return {
        "type": "FeatureCollection",
        "features": all_features,
        "drone_features": drone_features,
        "egress_features": egress_features,
        "inbound": {
            "g1": {"weight": round(w1, 3), "samples": samples.get("g1"), "color": cfg["g1"]["color"]},
            "g2": {"weight": round(w2, 3), "samples": samples.get("g2"), "color": cfg["g2"]["color"]},
            "g3": {"weight": round(w3, 3), "samples": samples.get("g3"), "color": cfg["g3"]["color"]},
        },
        "egress_axis_deg": axis,
        "egress_method": ",".join(methods) if methods else "none",
        "egress_by_mode": egress_pack.get("routes"),
        "egress_feasibility": egress_pack.get("feasibility"),
        "egress_terrain_class": egress_pack.get("terrain_class"),
        "egress_per_candidate": [
            {
                "feasibility": p.get("feasibility"),
                "terrain_class": p.get("terrain_class"),
                "axis_deg": p.get("axis_deg"),
            }
            for p in per_candidate
        ],
        "clearance_m_policy": {
            "min_m": min_clearance_m,
            "formula": "max(min_clearance_m, 15 + 0.55 * height_m)",
            "example_clearance_fn": "clearance_m_for_building",
        },
        "water_barrier_count": len(water or []),
        "legend": {
            "drone": [
                {"id": "g1", "label": "DRONE G1", "color": cfg["g1"]["color"]},
                {"id": "g2", "label": "DRONE G2", "color": cfg["g2"]["color"]},
                {"id": "g3", "label": "DRONE G3", "color": cfg["g3"]["color"]},
            ],
            "egress": [
                {
                    "id": f"egress_{m}",
                    "label": f"EGRESS {_MODE_LABELS[m]}",
                    "color": layer_colors.get(f"egress_{m}", _MODE_COLORS[m]),
                }
                for m in EGRESS_MODES
            ],
        },
        "method": "drone_inbound_v3+strict_mode_egress",
        "ai_required": False,
    }


# Re-export clearance helper for callers / docs
__all__ = [
    "urban_dodge_path",
    "road_egress_for_mode",
    "egress_for_candidate",
    "build_contact_inbound_curves",
    "build_path_groups",
    "clearance_m_for_building",
    "EGRESS_MODES",
]
