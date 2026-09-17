"""
Building / obstacle clearance for defensive ISR path sketches.

Paths must not cut through building centers. Clearance scales with height
(min ~25 m). Prefer lateral dodge into gaps between structures.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils import add_distance_to_coord, distance_between_coords, initial_bearing

Coord = Tuple[float, float]
Building = Dict[str, Any]

# meters per degree (approx at mid-latitudes; refined per-point below)
_M_PER_DEG_LAT = 111_320.0


def clearance_m_for_building(
    height_m: Optional[float],
    *,
    min_clearance_m: float = 25.0,
) -> float:
    """
    Standoff radius around a building point.

    Scales with height so tall structures get wider keep-out:
      clearance = max(min_clearance_m, 15 + 0.55 * height_m)
    Typical: 10 m → 25 m floor; 40 m → 37 m; 80 m → 59 m.
    """
    h = float(height_m if height_m is not None else 12.0)
    return max(min_clearance_m, 15.0 + 0.55 * max(0.0, h))


def _meters_xy(a: Coord, b: Coord) -> Tuple[float, float]:
    """Local ENU meters of b relative to a."""
    lat0 = math.radians(a[0])
    m_lng = _M_PER_DEG_LAT * math.cos(lat0)
    east = (b[1] - a[1]) * m_lng
    north = (b[0] - a[0]) * _M_PER_DEG_LAT
    return east, north


def point_to_segment_m(p: Coord, a: Coord, b: Coord) -> float:
    """Minimum distance from point p to segment a→b (meters)."""
    ax, ay = 0.0, 0.0
    bx, by = _meters_xy(a, b)
    px, py = _meters_xy(a, p)
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-6:
        return math.hypot(px, py)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
    qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


def segment_hits_building(
    a: Coord,
    b: Coord,
    building: Building,
    *,
    min_clearance_m: float = 25.0,
) -> bool:
    lat = building.get("lat") or building.get("latitude")
    lng = building.get("lng") or building.get("longitude")
    if lat is None or lng is None:
        return False
    bp = (float(lat), float(lng))
    clr = clearance_m_for_building(building.get("height_m"), min_clearance_m=min_clearance_m)
    return point_to_segment_m(bp, a, b) < clr


def offending_buildings(
    a: Coord,
    b: Coord,
    buildings: Sequence[Building],
    *,
    min_clearance_m: float = 25.0,
) -> List[Tuple[Building, float, float]]:
    """
    Buildings whose keep-out intersects segment a→b.
    Returns list of (building, dist_m, clearance_m) sorted by severity (dist/clearance).
    """
    hits: List[Tuple[Building, float, float]] = []
    for bld in buildings:
        lat = bld.get("lat") or bld.get("latitude")
        lng = bld.get("lng") or bld.get("longitude")
        if lat is None or lng is None:
            continue
        bp = (float(lat), float(lng))
        clr = clearance_m_for_building(bld.get("height_m"), min_clearance_m=min_clearance_m)
        d = point_to_segment_m(bp, a, b)
        if d < clr:
            hits.append((bld, d, clr))
    hits.sort(key=lambda x: x[1] / max(x[2], 1.0))
    return hits


def nearest_building(p: Coord, buildings: Sequence[Building]) -> Optional[Tuple[Building, float]]:
    best: Optional[Tuple[Building, float]] = None
    for bld in buildings:
        lat = bld.get("lat") or bld.get("latitude")
        lng = bld.get("lng") or bld.get("longitude")
        if lat is None or lng is None:
            continue
        d_m = distance_between_coords(p, (float(lat), float(lng))) * 1000.0
        if best is None or d_m < best[1]:
            best = (bld, d_m)
    return best


def gap_push_point(
    p: Coord,
    buildings: Sequence[Building],
    *,
    along_bearing: float,
    min_clearance_m: float = 25.0,
    max_push_m: float = 220.0,
) -> Coord:
    """
    Push point laterally away from nearest buildings into a gap.
    Tries both perpendicular sides and keeps the side with larger free space.
    """
    if not buildings:
        return p
    nb = nearest_building(p, buildings)
    if not nb:
        return p
    bld, d_m = nb
    clr = clearance_m_for_building(bld.get("height_m"), min_clearance_m=min_clearance_m)
    if d_m >= clr:
        return p

    # Direction away from building center
    blat = float(bld.get("lat") or bld.get("latitude"))
    blng = float(bld.get("lng") or bld.get("longitude"))
    away = initial_bearing((blat, blng), p)
    need_m = min(max_push_m, (clr - d_m) + 12.0)

    candidates: List[Coord] = []
    for brg in (away, (along_bearing + 90) % 360, (along_bearing - 90) % 360, (away + 180) % 360):
        q = add_distance_to_coord(p, brg, need_m / 1000.0)
        candidates.append(q)

    def free_score(q: Coord) -> float:
        nb2 = nearest_building(q, buildings)
        if not nb2:
            return 1e6
        _, d2 = nb2
        return d2

    return max(candidates, key=free_score)


def reroute_segment(
    a: Coord,
    b: Coord,
    buildings: Sequence[Building],
    *,
    min_clearance_m: float = 25.0,
    max_iters: int = 4,
) -> List[Coord]:
    """
    If a→b clips buildings, insert mid waypoints pushed into gaps.
    Returns waypoint list including a and b (possibly with mids).
    """
    hits = offending_buildings(a, b, buildings, min_clearance_m=min_clearance_m)
    if not hits:
        return [a, b]

    brg = initial_bearing(a, b)
    waypoints: List[Coord] = [a]
    # Split at mid; push mid into gap; recurse once per side if still blocked
    mid = add_distance_to_coord(a, brg, distance_between_coords(a, b) * 0.5)
    mid = gap_push_point(mid, buildings, along_bearing=brg, min_clearance_m=min_clearance_m)

    # Extra lateral if still offending
    for _ in range(max_iters):
        still = offending_buildings(a, mid, buildings, min_clearance_m=min_clearance_m) or offending_buildings(
            mid, b, buildings, min_clearance_m=min_clearance_m
        )
        if not still:
            break
        # Amplify lateral offset alternating sides
        side = 1 if _ % 2 == 0 else -1
        amp = 40.0 + 35.0 * _
        mid = add_distance_to_coord(mid, (brg + 90 * side) % 360, amp / 1000.0)
        mid = gap_push_point(mid, buildings, along_bearing=brg, min_clearance_m=min_clearance_m)

    waypoints.append(mid)
    waypoints.append(b)
    return waypoints


def avoid_buildings_polyline(
    waypoints: List[Coord],
    buildings: Sequence[Building],
    *,
    min_clearance_m: float = 25.0,
) -> List[Coord]:
    """Rewrite a polyline so no segment punches through building keep-outs."""
    if len(waypoints) < 2 or not buildings:
        return waypoints
    out: List[Coord] = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        # Start from last accepted point for continuity
        a = out[-1]
        segs = reroute_segment(a, b, buildings, min_clearance_m=min_clearance_m)
        out.extend(segs[1:])
    # Final nudge of interior vertices
    nudged: List[Coord] = [out[0]]
    for i in range(1, len(out) - 1):
        brg = initial_bearing(out[i - 1], out[i + 1])
        nudged.append(
            gap_push_point(out[i], buildings, along_bearing=brg, min_clearance_m=min_clearance_m)
        )
    nudged.append(out[-1])
    return nudged


def polyline_clearance_ok(
    waypoints: Sequence[Coord],
    buildings: Sequence[Building],
    *,
    min_clearance_m: float = 25.0,
) -> bool:
    if len(waypoints) < 2:
        return True
    for i in range(len(waypoints) - 1):
        if offending_buildings(waypoints[i], waypoints[i + 1], buildings, min_clearance_m=min_clearance_m):
            return False
    return True


def classify_terrain_class(
    terrain: Optional[Dict[str, Any]],
    buildings: Sequence[Building],
    origin: Coord,
    *,
    urban_building_radius_m: float = 450.0,
) -> str:
    """
    Coarse class for egress fallback policy:
      urban | open_plain | rough
    Arc fallback allowed only for open_plain / rough (never urban).
    """
    near_n = 0
    for bld in buildings:
        lat = bld.get("lat") or bld.get("latitude")
        lng = bld.get("lng") or bld.get("longitude")
        if lat is None or lng is None:
            continue
        if distance_between_coords(origin, (float(lat), float(lng))) * 1000.0 <= urban_building_radius_m:
            near_n += 1
    if near_n >= 8:
        return "urban"

    # Explicit class from caller
    if terrain:
        explicit = terrain.get("class") or terrain.get("terrain_class") or terrain.get("label")
        if explicit in ("urban", "open_plain", "rough"):
            return str(explicit)

        pts = terrain.get("points") or []
        if len(pts) >= 4:
            elevs = [float(p.get("elevation_m") or 0) for p in pts]
            mean = sum(elevs) / len(elevs)
            var = sum((e - mean) ** 2 for e in elevs) / len(elevs)
            std = math.sqrt(var)
            if std >= 18.0:
                return "rough"
            if near_n <= 2:
                return "open_plain"
            return "rough" if std >= 8.0 else "open_plain"

    if near_n <= 2:
        return "open_plain"
    if near_n <= 7:
        return "rough"
    return "urban"
