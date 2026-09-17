"""
Human-operator behavioral priors (algorithm-only, no AI).

Trained operators do NOT always pick the tallest roof.
They trade RF LOS vs concealment, egress, blending, and counter-detection.

All scores in [0,1]. Final fusion is explicit weighted product/sum — causal, auditable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.utils import distance_between_coords, initial_bearing

Coord = Tuple[float, float]

# Explicit causal weights (must sum ≈ 1)
BEHAVIOR_WEIGHTS = {
    "rf_geometry": 0.22,      # AoA/RSSI residual — physics
    "concealment": 0.18,      # avoid obvious skyline / landmarks
    "egress_access": 0.16,    # road/path escape options
    "blend_cover": 0.12,      # civilian fabric blend (not always military tag)
    "mid_elevation": 0.10,    # mid-band height preference (not max)
    "standoff": 0.10,         # tactical distance band from observer
    "los_usable": 0.08,       # enough LOS for control, not exposed ridge
    "cluster_support": 0.04,  # nearby peer mass (ambiguous hideouts)
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def mid_elevation_score(height_m: Optional[float], ground_m: Optional[float], peers: List[Dict[str, Any]]) -> float:
    """
    Prefer mid-rise (≈8–28m) over skyscraper landmarks or ground-level.
    Relative to local peer distribution — not absolute 'higher is better'.
    """
    h = float(height_m) if height_m is not None else 10.0
    # absolute band: inverted-U around 15m
    if h < 4:
        abs_s = 0.25
    elif h <= 28:
        abs_s = math.exp(-0.5 * ((h - 15.0) / 9.0) ** 2)
    else:
        # very tall = more observable / more likely watched
        abs_s = max(0.15, 0.55 - (h - 28.0) / 80.0)

    grounds = [
        float(p.get("ground_elevation_m") or (p.get("elevation") or {}).get("ground_m") or 0)
        for p in peers
        if (p.get("ground_elevation_m") is not None)
        or ((p.get("elevation") or {}).get("ground_m") is not None)
    ]
    if ground_m is not None and grounds:
        g = float(ground_m)
        med = sorted(grounds)[len(grounds) // 2]
        # slight preference for local high, but penalize being the unique max ridge
        gmax = max(grounds)
        if g >= gmax - 1 and (gmax - med) > 8:
            rel = 0.35  # obvious ridge penalty
        else:
            rel = _clamp01(0.45 + (g - med) / 25.0)
    else:
        rel = 0.5
    return _clamp01(0.65 * abs_s + 0.35 * rel)


def concealment_score(c: Dict[str, Any]) -> float:
    """
    Trained OP favors concealment: courtyards, mid blocks, industrial sheds,
    avoid pure landmark military tags unless corroborated.
    """
    bt = (c.get("building_type") or (c.get("attributes") or {}).get("building_type") or "unknown")
    mil = bool(c.get("military") or (c.get("attributes") or {}).get("military"))
    h = float(c.get("height_m") or 10)
    # type prior
    type_s = {
        "industrial": 0.85,
        "commercial": 0.70,
        "residential": 0.75,
        "unknown": 0.55,
    }.get(str(bt), 0.5)
    # military tag alone is a trap — lower unless mid height
    if mil:
        type_s *= 0.55 if h > 25 else 0.8
    # extremely tall buildings draw ISR attention
    if h > 45:
        type_s *= 0.4
    return _clamp01(type_s)


def egress_access_score(c: Dict[str, Any], approach_bearing: float, observer: Coord) -> float:
    """
    Escape after launch: near road but not ON the approach axis (counter-pursuit).
    Prefer side/rear road access relative to inbound bearing.
    """
    attrs = c.get("attributes") or {}
    road = attrs.get("dist_to_road_m")
    if road is None:
        road = c.get("dist_to_road_m")
    if road is None:
        road_s = 0.45
    else:
        r = float(road)
        # sweet spot 40–350m from road (hide + quick mount)
        if r < 15:
            road_s = 0.35  # too exposed on curb
        elif r <= 350:
            road_s = math.exp(-0.5 * ((r - 120.0) / 140.0) ** 2)
        else:
            road_s = max(0.1, 0.5 - (r - 350) / 1200.0)

    lat = float(c.get("latitude") or c.get("lat"))
    lng = float(c.get("longitude") or c.get("lng"))
    brg = initial_bearing(observer, (lat, lng))
    # angle from approach axis — prefer off-axis hide (30–150°)
    delta = abs((brg - approach_bearing + 180) % 360 - 180)
    off = 1.0 - abs(delta - 90) / 90.0  # peak at 90° off-axis
    off = _clamp01(0.4 + 0.6 * max(off, 0))
    return _clamp01(0.6 * road_s + 0.4 * off)


def standoff_score(dist_km: float, range_km: float) -> float:
    """Control link sweet-band: not too close (exposure), not at max range edge."""
    if range_km <= 0.3:
        return 0.5
    x = dist_km / range_km
    # prefer 0.25–0.7 of control range
    if x < 0.12:
        return 0.3
    if x <= 0.7:
        return math.exp(-0.5 * ((x - 0.42) / 0.22) ** 2)
    return max(0.15, 1.0 - (x - 0.7) / 0.5)


def los_usable_score(c: Dict[str, Any]) -> float:
    """Need workable LOS, but perfect exposed ridge is not required."""
    comps = (c.get("scores") or {}).get("components") or {}
    p_los = comps.get("p_L_los")
    if p_los is None:
        elev = c.get("elevation") or {}
        p_los = 0.85 if elev.get("los_clear", True) else 0.35
    p_los = float(p_los)
    # soft mid: 0.55–0.9 best; 1.0 slightly down-weighted (too open)
    if p_los >= 0.95:
        return 0.75
    if p_los >= 0.55:
        return p_los
    return p_los * 0.8


def rf_geometry_score(c: Dict[str, Any]) -> float:
    comps = (c.get("scores") or {}).get("components") or {}
    aoa = float(comps.get("p_L_aoa") or 0)
    rssi = float(comps.get("p_L_rssi") or 0.5)
    # also use bearing error if present
    be = c.get("bearing_error_deg")
    if be is None:
        be = comps.get("p_bearing_error_deg")
    if be is not None:
        aoa = max(aoa, math.exp(-0.5 * (float(be) / 8.0) ** 2))
    return _clamp01(0.6 * aoa + 0.4 * rssi)


def blend_cover_score(c: Dict[str, Any]) -> float:
    """Blend into urban clutter; cell tower nearby helps C2 but also detection risk."""
    attrs = c.get("attributes") or {}
    bt = attrs.get("building_type") or c.get("building_type") or "unknown"
    tower = bool(attrs.get("cell_tower_nearby"))
    base = {"residential": 0.8, "commercial": 0.7, "industrial": 0.75, "unknown": 0.5}.get(
        str(bt), 0.5
    )
    # tower: mild boost for link, mild penalty for SIGINT exposure
    tower_s = 0.55 if tower else 0.5
    return _clamp01(0.75 * base + 0.25 * tower_s)


def cluster_support_score(c: Dict[str, Any], peers: List[Dict[str, Any]]) -> float:
    """Ambiguous blocks with several mid-score peers — harder to uniquely pin."""
    lat = float(c.get("latitude") or c.get("lat"))
    lng = float(c.get("longitude") or c.get("lng"))
    n = 0
    for p in peers:
        if p.get("id") == c.get("id"):
            continue
        plat = p.get("latitude") or p.get("lat")
        plng = p.get("longitude") or p.get("lng")
        if plat is None or plng is None:
            continue
        if distance_between_coords((lat, lng), (float(plat), float(plng))) * 1000 < 120:
            n += 1
    return _clamp01(0.35 + 0.12 * min(n, 5))


def score_human_operator(
    c: Dict[str, Any],
    observer: Coord,
    approach_bearing: float,
    range_km: float,
    peers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    lat = float(c.get("latitude") or c.get("lat"))
    lng = float(c.get("longitude") or c.get("lng"))
    dist_km = distance_between_coords(observer, (lat, lng))
    parts = {
        "rf_geometry": rf_geometry_score(c),
        "concealment": concealment_score(c),
        "egress_access": egress_access_score(c, approach_bearing, observer),
        "blend_cover": blend_cover_score(c),
        "mid_elevation": mid_elevation_score(
            c.get("height_m"),
            c.get("ground_elevation_m") or (c.get("elevation") or {}).get("ground_m"),
            peers,
        ),
        "standoff": standoff_score(dist_km, range_km),
        "los_usable": los_usable_score(c),
        "cluster_support": cluster_support_score(c, peers),
    }
    total = sum(BEHAVIOR_WEIGHTS[k] * parts[k] for k in BEHAVIOR_WEIGHTS)
    reasons = []
    # top causal drivers
    ranked = sorted(parts.items(), key=lambda kv: BEHAVIOR_WEIGHTS[kv[0]] * kv[1], reverse=True)
    for k, v in ranked[:3]:
        reasons.append(f"human:{k}={v:.2f}")
    if parts["mid_elevation"] < 0.4:
        reasons.append("human:avoid_obvious_highpoint")
    if parts["concealment"] >= 0.7:
        reasons.append("human:concealment_fit")
    if parts["egress_access"] >= 0.65:
        reasons.append("human:egress_ready")
    return {
        "human_score": round(total, 5),
        "human_parts": {k: round(v, 4) for k, v in parts.items()},
        "human_reasons": reasons,
        "weights": BEHAVIOR_WEIGHTS,
    }


def apply_human_behavior_prior(
    pool: List[Dict[str, Any]],
    observer: Coord,
    approach_bearing: float,
    range_km: float,
    mix: float = 0.42,
) -> List[Dict[str, Any]]:
    """
    Reweight probability:
      P' ∝ P_rf^ (1-mix) * H^mix
    mix~0.42 → human behavior strongly shapes ranking without ignoring RF.
    """
    if not pool:
        return []
    scored = []
    for c in pool:
        hb = score_human_operator(c, observer, approach_bearing, range_km, pool)
        item = dict(c)
        item.update(hb)
        p = float(c.get("probability") or c.get("precision_score") or 1e-6)
        h = max(hb["human_score"], 1e-6)
        item["behavior_fusion"] = round((p ** (1.0 - mix)) * (h**mix), 8)
        reasons = list(item.get("reasons") or [])
        for r in hb["human_reasons"]:
            if r not in reasons:
                reasons.append(r)
        item["reasons"] = reasons
        scored.append(item)

    total = sum(max(c["behavior_fusion"], 0.0) for c in scored) or 1.0
    for c in scored:
        c["probability"] = round(c["behavior_fusion"] / total, 6)
        c["probability_pct"] = round(c["probability"] * 100.0, 3)
    scored.sort(key=lambda x: x["probability"], reverse=True)
    for i, c in enumerate(scored, start=1):
        c["rank"] = i
    return scored


def algorithmic_shortlist(
    pool: List[Dict[str, Any]],
    max_keep: int = 18,
    min_sep_m: float = 38.0,
) -> Dict[str, Any]:
    """
    Deterministic shortlist without AI — diversity + score, then fill to max_keep.
    """
    if not pool:
        return {"shortlist": [], "count": 0, "notes": ["empty_pool"]}
    kept: List[Dict[str, Any]] = []
    skipped_close = 0
    for c in pool:
        lat = float(c.get("latitude") or c.get("lat"))
        lng = float(c.get("longitude") or c.get("lng"))
        too_close = False
        for k in kept:
            klat = float(k.get("latitude") or k.get("lat"))
            klng = float(k.get("longitude") or k.get("lng"))
            if distance_between_coords((lat, lng), (klat, klng)) * 1000 < min_sep_m:
                too_close = True
                break
        if too_close:
            skipped_close += 1
            continue
        kept.append(dict(c))
        if len(kept) >= max_keep:
            break
    # Fill remaining slots from pool (allow closer) so urban clusters still show
    if len(kept) < max_keep:
        ids = {c.get("id") for c in kept}
        for c in pool:
            if len(kept) >= max_keep:
                break
            if c.get("id") in ids:
                continue
            kept.append(dict(c))
            ids.add(c.get("id"))
    total = sum(float(c.get("probability") or 0) for c in kept) or 1.0
    for i, c in enumerate(kept, start=1):
        c["rank"] = i
        c["probability"] = round(float(c.get("probability") or 0) / total, 6)
        c["probability_pct"] = round(c["probability"] * 100.0, 3)
        c["shortlist_method"] = "human_ops_diversity_v2"
    return {
        "shortlist": kept,
        "count": len(kept),
        "notes": [
            f"kept={len(kept)}",
            f"skipped_close={skipped_close}",
            f"min_sep_m={min_sep_m}",
            "ai_not_required",
        ],
    }
