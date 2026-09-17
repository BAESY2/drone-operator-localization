"""
Strict OSM road-only egress router.

Rules:
  - Move ONLY along consecutive points of real highway geometries.
  - NEVER invent soft edges between arbitrary nodes (those cross rivers).
  - NEVER emit arc/straight fallbacks through urban fabric.
  - Origin snaps to nearest road node within max_snap_m; else no route.
  - Optional waterways: reject snap segments that cross water (bridges OK via highway geom).
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

from src.providers.http import get_json
from src.utils import distance_between_coords, initial_bearing

Coord = Tuple[float, float]
NodeKey = Tuple[float, float]

OVERPASS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

# Modes may only traverse these highway classes
MODE_ALLOWED = {
    "foot": {
        "footway", "path", "pedestrian", "steps", "residential", "living_street",
        "track", "service", "unclassified", "tertiary", "cycleway", "bridleway",
    },
    "bike": {
        "cycleway", "residential", "living_street", "tertiary", "secondary",
        "unclassified", "track", "path", "service", "primary",
    },
    "motorcycle": {
        "residential", "living_street", "tertiary", "tertiary_link", "secondary",
        "secondary_link", "primary", "primary_link", "unclassified", "service",
        "trunk", "trunk_link", "track",
    },
    "car": {
        "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
        "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
        "residential", "living_street", "service",
    },
    # Military / armored: prefers rugged & secondary; still OSM-only (no cross-country)
    "armored": {
        "track", "unclassified", "tertiary", "tertiary_link", "secondary",
        "secondary_link", "residential", "service", "path", "primary",
    },
}

MODE_HW_PREF = {
    "foot": {
        "footway": 1.0, "path": 0.95, "pedestrian": 0.95, "residential": 0.8,
        "living_street": 0.8, "track": 0.7, "cycleway": 0.5, "tertiary": 0.45,
    },
    "bike": {
        "cycleway": 1.0, "residential": 0.9, "living_street": 0.85, "tertiary": 0.75,
        "secondary": 0.65, "path": 0.4, "track": 0.5,
    },
    "motorcycle": {
        "secondary": 1.0, "tertiary": 0.95, "primary": 0.9, "residential": 0.75,
        "trunk": 0.7, "unclassified": 0.55, "track": 0.35,
    },
    "car": {
        "primary": 1.0, "secondary": 0.95, "trunk": 0.95, "tertiary": 0.8,
        "residential": 0.55, "unclassified": 0.5, "service": 0.35, "motorway": 1.0,
    },
    "armored": {
        "track": 1.0, "unclassified": 0.9, "tertiary": 0.75, "secondary": 0.7,
        "path": 0.55, "residential": 0.45, "service": 0.4,
    },
}

MODE_TARGET_KM = {
    "foot": 1.2,
    "bike": 2.4,
    "motorcycle": 3.2,
    "car": 3.5,
    "armored": 2.8,
}
MODE_SPEED_KMH = {
    "foot": 4.5,
    "bike": 18.0,
    "motorcycle": 45.0,
    "car": 40.0,
    "armored": 32.0,
}
# Distinct palette: cool/ground tones for egress (drone paths stay warm reds in UI)
MODE_COLORS = {
    "foot": "#6bcf6b",
    "bike": "#3d9bfd",
    "motorcycle": "#2ec4b6",
    "car": "#f0c14b",
    "armored": "#c47a2a",
}
MODE_LABELS = {
    "foot": "FOOT",
    "bike": "BIKE",
    "motorcycle": "MOTO",
    "car": "CAR",
    "armored": "MIL",
}
MODE_DASH = {
    "foot": "4 7",
    "bike": "8 5",
    "motorcycle": "12 5",
    "car": None,
    "armored": "14 6 3 6",
}

EGRESS_MODE_ORDER = ("foot", "bike", "motorcycle", "car", "armored")


def _nk(p: Coord) -> NodeKey:
    return (round(p[0], 5), round(p[1], 5))


def _overpass(query: str, cache_key: str) -> Dict[str, Any]:
    body = urlencode({"data": query}).encode("utf-8")
    last = None
    for ep in OVERPASS:
        try:
            return get_json(
                ep,
                method="POST",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                cache_key=cache_key,
                ttl_sec=6 * 3600,
                timeout=12,
                allow_stale=True,
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(str(last) if last else "overpass failed")


def fetch_roads_and_water(
    center: Coord,
    radius_m: float = 3500,
) -> Dict[str, Any]:
    """Fetch highway ways (with geom) + water barriers for validation."""
    lat, lng = center
    r = int(min(max(radius_m, 900), 8000))
    query = f"""
[out:json][timeout:35];
(
  way["highway"](around:{r},{lat:.5f},{lng:.5f});
  way["waterway"~"^(river|canal|stream|drain)$"](around:{r},{lat:.5f},{lng:.5f});
  relation["waterway"="river"](around:{r},{lat:.5f},{lng:.5f});
  way["natural"="water"](around:{r},{lat:.5f},{lng:.5f});
);
out geom tags;
""".strip()
    try:
        payload = _overpass(query, f"roads_water:{lat:.4f},{lng:.4f}:{r}")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "roads": [], "water": [], "error": str(exc)[:120]}

    roads: List[Dict[str, Any]] = []
    water: List[Dict[str, Any]] = []
    for el in payload.get("elements") or []:
        tags = el.get("tags") or {}
        geom = el.get("geometry") or []
        coords = [
            (float(p["lat"]), float(p["lon"]))
            for p in geom
            if "lat" in p and "lon" in p
        ]
        if len(coords) < 2:
            continue
        if tags.get("highway"):
            roads.append(
                {
                    "id": el.get("id"),
                    "highway": tags.get("highway"),
                    "name": tags.get("name"),
                    "bridge": tags.get("bridge") in ("yes", "true", "viaduct", "aqueduct")
                    or bool(tags.get("bridge")),
                    "tunnel": tags.get("tunnel") in ("yes", "true"),
                    "coords": coords,
                }
            )
        elif tags.get("waterway") or tags.get("natural") == "water":
            water.append(
                {
                    "id": el.get("id"),
                    "kind": tags.get("waterway") or "water",
                    "coords": coords,
                }
            )
    return {"ok": True, "roads": roads, "water": water, "count_roads": len(roads), "count_water": len(water)}


def _orient(a: Coord, b: Coord, c: Coord) -> float:
    return (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])


def _on_seg(a: Coord, b: Coord, c: Coord, eps: float = 1e-12) -> bool:
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    )


def segments_intersect(a: Coord, b: Coord, c: Coord, d: Coord) -> bool:
    """Proper/improper segment intersection (lat/lng as plane — OK at local scale)."""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) < 1e-15 and _on_seg(a, b, c):
        return True
    if abs(o2) < 1e-15 and _on_seg(a, b, d):
        return True
    if abs(o3) < 1e-15 and _on_seg(c, d, a):
        return True
    if abs(o4) < 1e-15 and _on_seg(c, d, b):
        return True
    return False


def crosses_water(a: Coord, b: Coord, water: Sequence[Dict[str, Any]]) -> bool:
    for w in water:
        coords = w.get("coords") or []
        for i in range(len(coords) - 1):
            if segments_intersect(a, b, coords[i], coords[i + 1]):
                return True
    return False


def build_strict_graph(
    roads: Sequence[Dict[str, Any]],
    mode: str,
    *,
    water: Optional[Sequence[Dict[str, Any]]] = None,
    stitch_m: float = 12.0,
) -> Tuple[
    Dict[NodeKey, List[Tuple[NodeKey, float, str, bool]]],
    Dict[NodeKey, Coord],
]:
    """
    Adjacency: only consecutive points on the same OSM way.
    Plus micro-stitch (< stitch_m) for near-miss intersection nodes.
    Soft long jumps are NEVER added (those cross rivers).
    Edge = (neighbor_key, length_km, highway, is_bridge)
    """
    allowed = MODE_ALLOWED.get(mode) or MODE_ALLOWED["foot"]
    adj: Dict[NodeKey, List[Tuple[NodeKey, float, str, bool]]] = {}
    nodes: Dict[NodeKey, Coord] = {}

    for seg in roads:
        hw = str(seg.get("highway") or "")
        base = hw.replace("_link", "")
        if hw not in allowed and base not in allowed:
            continue
        coords = seg.get("coords") or []
        if len(coords) < 2:
            continue
        is_bridge = bool(seg.get("bridge"))
        use_hw = hw if hw in (MODE_HW_PREF.get(mode) or {}) else base
        for i in range(len(coords) - 1):
            a = (float(coords[i][0]), float(coords[i][1]))
            b = (float(coords[i + 1][0]), float(coords[i + 1][1]))
            ka, kb = _nk(a), _nk(b)
            nodes[ka] = a
            nodes[kb] = b
            d = distance_between_coords(a, b)
            if d < 1e-6:
                continue
            if d > 0.35:
                continue
            adj.setdefault(ka, []).append((kb, d, use_hw or hw, is_bridge))
            adj.setdefault(kb, []).append((ka, d, use_hw or hw, is_bridge))

    # Micro-stitch near-miss intersections (≤ stitch_m). Reject water crossings.
    if stitch_m > 0 and len(nodes) >= 2:
        keys = list(nodes.keys())
        # Spatial buckets ~0.00015 deg ≈ 15 m
        buckets: Dict[Tuple[int, int], List[NodeKey]] = {}
        for k in keys:
            lat, lng = nodes[k]
            buckets.setdefault((int(lat / 0.00015), int(lng / 0.00015)), []).append(k)
        for (bi, bj), group in list(buckets.items()):
            cand_keys: List[NodeKey] = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    cand_keys.extend(buckets.get((bi + di, bj + dj), []))
            for i, ka in enumerate(group):
                a = nodes[ka]
                for kb in cand_keys:
                    if ka >= kb:
                        continue
                    b = nodes[kb]
                    d_m = distance_between_coords(a, b) * 1000.0
                    if d_m < 0.4 or d_m > stitch_m:
                        continue
                    if water and crosses_water(a, b, water):
                        continue
                    d_km = d_m / 1000.0
                    adj.setdefault(ka, []).append((kb, d_km, "stitch", False))
                    adj.setdefault(kb, []).append((ka, d_km, "stitch", False))
    return adj, nodes


def nearest_road_node(
    nodes: Dict[NodeKey, Coord],
    p: Coord,
    *,
    max_snap_m: float = 90.0,
    water: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Coord]:
    """Snap to nearest node; reject if snap segment crosses water."""
    best = None
    best_d = 1e9
    for n in nodes.values():
        d = distance_between_coords(p, n) * 1000.0
        if d < best_d:
            best_d = d
            best = n
    if best is None or best_d > max_snap_m:
        return None
    if water and crosses_water(p, best, water):
        ranked = sorted(
            nodes.values(),
            key=lambda n: distance_between_coords(p, n),
        )
        for n in ranked[:40]:
            d = distance_between_coords(p, n) * 1000.0
            if d > max_snap_m:
                break
            if not crosses_water(p, n, water):
                return n
        return None
    return best


def route_along_roads(
    origin: Coord,
    axis_bearing: float,
    roads: Sequence[Dict[str, Any]],
    *,
    mode: str = "foot",
    target_km: Optional[float] = None,
    water: Optional[Sequence[Dict[str, Any]]] = None,
    bearing_offset_deg: float = 0.0,
    max_snap_m: float = 90.0,
) -> Optional[Dict[str, Any]]:
    """
    Dijkstra on strict road graph. Cost = length - alignment bonus.
    Stops when traveled >= target_km or frontier exhausted.
    """
    if not roads:
        return None
    adj, nodes = build_strict_graph(roads, mode, water=water)
    if len(nodes) < 8:
        return None

    start = nearest_road_node(nodes, origin, max_snap_m=max_snap_m, water=water)
    if not start:
        return None

    prefer = (axis_bearing + bearing_offset_deg) % 360.0
    target = float(target_km if target_km is not None else MODE_TARGET_KM.get(mode, 1.5))
    pref = MODE_HW_PREF.get(mode) or {}

    sk = _nk(start)
    dist_cost: Dict[NodeKey, float] = {sk: 0.0}
    prev: Dict[NodeKey, Optional[NodeKey]] = {sk: None}
    pq: List[Tuple[float, float, NodeKey]] = [(0.0, 0.0, sk)]
    best_end = sk
    best_progress = -1.0

    while pq:
        pri, traveled, uk = heapq.heappop(pq)
        if traveled > dist_cost.get(uk, 1e9) + 1e-9:
            continue
        u = nodes[uk]
        brg_u = initial_bearing(origin, u)
        delta_u = abs((brg_u - prefer + 180) % 360 - 180)
        prog = traveled * math.cos(math.radians(min(delta_u, 89)))
        if prog > best_progress and traveled >= 0.15:
            best_progress = prog
            best_end = uk
        if traveled >= target:
            best_end = uk
            break

        for vk, edge_km, hw, is_bridge in adj.get(uk, []):
            v = nodes[vk]
            if water and not is_bridge and crosses_water(u, v, water):
                continue
            nd = traveled + edge_km
            if nd < dist_cost.get(vk, 1e9):
                dist_cost[vk] = nd
                prev[vk] = uk
                brg = initial_bearing(u, v)
                delta = abs((brg - prefer + 180) % 360 - 180)
                align = math.cos(math.radians(min(delta, 90)))
                hw_bonus = 0.08 * float(pref.get(hw, 0.25 if hw == "stitch" else 0.3))
                new_pri = nd - 0.55 * align * edge_km - hw_bonus * edge_km
                if delta > 110:
                    new_pri += 0.4 * edge_km
                heapq.heappush(pq, (new_pri, nd, vk))

    if best_end == sk or dist_cost.get(best_end, 0) < 0.2:
        return None

    chain_keys: List[NodeKey] = []
    cur: Optional[NodeKey] = best_end
    guard = 0
    while cur is not None and guard < 5000:
        chain_keys.append(cur)
        cur = prev.get(cur)
        guard += 1
    chain_keys.reverse()
    if len(chain_keys) < 3:
        return None

    path_pts = [nodes[k] for k in chain_keys]
    full: List[Coord] = []
    snap_d = distance_between_coords(origin, start) * 1000.0
    if snap_d <= max_snap_m and not (water and crosses_water(origin, start, water)):
        full.append(origin)
    full.extend(path_pts)

    coords = [[round(p[1], 6), round(p[0], 6)] for p in full]
    length = dist_cost[best_end] + (snap_d / 1000.0 if full and full[0] == origin else 0)
    speed = MODE_SPEED_KMH.get(mode, 10.0)
    feasibility = max(
        0.1,
        min(
            0.95,
            0.45 * min(1.0, length / target)
            + 0.40 * min(1.0, best_progress / max(target * 0.5, 0.1))
            + 0.15,
        ),
    )
    return {
        "coords": coords,
        "waypoints": full,
        "length_km": round(length, 3),
        "method": "strict_osm_road_dijkstra",
        "mode": mode,
        "eta_min": round((length / max(speed, 0.1)) * 60.0, 1),
        "feasibility": round(feasibility, 3),
        "target_km": target,
        "snap_m": round(snap_d, 1),
        "note": "Road-network only — no cross-country / no river jump",
    }
