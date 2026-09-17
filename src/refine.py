"""
Progressive candidate refinement — algorithm-only (no AI required).

Each pass tightens the pool using measurable residuals:
  AoA gate → RSSI annulus → relative precision → spatial NMS → mass cut → radius shrink
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.precision import confidence_radius_m
from src.rf_models import bearing_delta, constants, rssi_annulus_km
from src.utils import distance_between_coords, initial_bearing

Coord = Tuple[float, float]


def _latlng(c: Dict[str, Any]) -> Coord:
    loc = c.get("location") or {}
    lat = loc.get("latitude", c.get("latitude", c.get("lat")))
    lng = loc.get("longitude", c.get("longitude", c.get("lng")))
    return float(lat), float(lng)


def _bearing_err(c: Dict[str, Any], observer: Coord, bearing_deg: float) -> float:
    comps = (c.get("scores") or {}).get("components") or {}
    if comps.get("p_bearing_error_deg") is not None:
        return float(comps["p_bearing_error_deg"])
    prec = c.get("precision") or {}
    if prec.get("bearing_error_deg") is not None:
        return float(prec["bearing_error_deg"])
    lat, lng = _latlng(c)
    return bearing_delta(bearing_deg, initial_bearing(observer, (lat, lng)))


def progressive_refine(
    pool: List[Dict[str, Any]],
    observer: Coord,
    bearing_deg: float,
    *,
    rssi_dbm: Optional[float] = None,
    n_observers: int = 1,
    max_keep: int = 5,
) -> Dict[str, Any]:
    """
    Returns refined candidates + pass log + tightened departure radius.
    Designed to run AFTER aggregate/agents/review for extra squeeze.
    """
    cnst = constants()
    sigma = float(cnst.get("aoa_sigma_deg_single", 10.0))
    if n_observers >= 2:
        sigma = float(cnst.get("aoa_sigma_deg_multi", 6.0))

    notes: List[str] = []
    rejected: List[Dict[str, Any]] = []
    cur = [dict(c) for c in pool]
    notes.append(f"start n={len(cur)}")

    # Pass 1 — hard AoA: keep within 1.5σ (tight) unless pool tiny
    aoa_lim = sigma * (1.25 if n_observers >= 2 else 1.5)
    nxt: List[Dict[str, Any]] = []
    for c in cur:
        err = _bearing_err(c, observer, bearing_deg)
        c["_bearing_err_deg"] = round(err, 3)
        if err <= aoa_lim or len(cur) <= 3:
            nxt.append(c)
        else:
            rejected.append({"id": c.get("id"), "pass": 1, "reason": f"aoa>{aoa_lim:.1f}° ({err:.2f}°)"})
    if nxt:
        cur = nxt
    notes.append(f"pass1_aoa≤{aoa_lim:.1f}° → {len(cur)}")

    # Pass 2 — RSSI annulus (if RSSI present)
    if rssi_dbm is not None and cur:
        d_min, d_nom, d_max = rssi_annulus_km(rssi_dbm)
        # tighten annulus: use 0.85 of outer / 1.15 of inner
        d_min_t = d_min * 1.1
        d_max_t = d_max * 0.9
        nxt = []
        for c in cur:
            lat, lng = _latlng(c)
            d = distance_between_coords(observer, (lat, lng))
            c["_dist_km"] = round(d, 4)
            if d_min_t <= d <= d_max_t or len(cur) <= 3:
                nxt.append(c)
            else:
                rejected.append(
                    {
                        "id": c.get("id"),
                        "pass": 2,
                        "reason": f"rssi_annulus {d:.3f}km not in [{d_min_t:.3f},{d_max_t:.3f}]",
                    }
                )
        if nxt:
            cur = nxt
        notes.append(f"pass2_rssi [{d_min_t:.2f},{d_max_t:.2f}]km → {len(cur)} (nom {d_nom:.2f})")

    # Pass 3 — relative precision / probability peak
    if cur:
        best = max(
            float(c.get("precision_score") or c.get("probability") or 0) for c in cur
        ) or 1e-12
        floor = best * (0.35 if n_observers >= 2 else 0.28)
        nxt = []
        for c in cur:
            s = float(c.get("precision_score") or c.get("probability") or 0)
            if s >= floor or len(cur) <= max_keep:
                nxt.append(c)
            else:
                rejected.append({"id": c.get("id"), "pass": 3, "reason": f"below_peak*{floor/best:.2f}"})
        cur = nxt or cur[:max_keep]
        notes.append(f"pass3_peak_floor={floor:.4g} → {len(cur)}")

    # Pass 4 — spatial NMS (urban: allow denser cluster reps)
    nms_m = 32.0 if n_observers >= 2 else 42.0
    cur = sorted(cur, key=lambda x: float(x.get("probability") or 0), reverse=True)
    kept: List[Dict[str, Any]] = []
    for c in cur:
        lat, lng = _latlng(c)
        clash = False
        for k in kept:
            klat, klng = _latlng(k)
            if distance_between_coords((lat, lng), (klat, klng)) * 1000.0 < nms_m:
                k["cluster_size"] = int(k.get("cluster_size") or 1) + 1
                clash = True
                rejected.append({"id": c.get("id"), "pass": 4, "reason": f"nms<{nms_m:.0f}m"})
                break
        if not clash:
            item = dict(c)
            item["cluster_size"] = int(item.get("cluster_size") or 1)
            kept.append(item)
    cur = kept
    notes.append(f"pass4_nms {nms_m:.0f}m → {len(cur)}")

    # Pass 5 — keep wide pool: mass OR min_count floor (don't collapse to 2–3)
    cur = sorted(cur, key=lambda x: float(x.get("probability") or 0), reverse=True)
    mass_target = 0.92
    min_keep = min(max_keep, max(12, max_keep // 2))
    mass_kept: List[Dict[str, Any]] = []
    cum = 0.0
    for c in cur:
        mass_kept.append(c)
        cum += float(c.get("probability") or 0)
        if len(mass_kept) >= max_keep:
            break
        if cum >= mass_target and len(mass_kept) >= min_keep:
            break
    # ensure floor
    if len(mass_kept) < min_keep:
        mass_kept = cur[: min(min_keep, len(cur))]
        cum = sum(float(c.get("probability") or 0) for c in mass_kept)
    for c in cur[len(mass_kept) :]:
        rejected.append({"id": c.get("id"), "pass": 5, "reason": "mass_tail"})
    cur = mass_kept
    notes.append(f"pass5_mass≥{mass_target} min={min_keep} cum={cum:.3f} → {len(cur)}")

    # Pass 6 — recompute tight confidence radii + renormalize
    final: List[Dict[str, Any]] = []
    for c in cur[:max_keep]:
        lat, lng = _latlng(c)
        dist_km = distance_between_coords(observer, (lat, lng))
        err = float(c.get("_bearing_err_deg") or _bearing_err(c, observer, bearing_deg))
        agree = float(c.get("agent_agreement") or 0.55)
        radius = confidence_radius_m(err, dist_km, n_observers, agree)
        item = dict(c)
        item["confidence_radius_m"] = round(radius, 2)
        item["bearing_error_deg"] = round(err, 3)
        item["distance_m"] = round(dist_km * 1000.0, 2)
        item["distance_km"] = round(dist_km, 4)
        # precision badge
        if radius <= 60 and err <= sigma:
            item["fix_quality"] = "tight"
        elif radius <= 150:
            item["fix_quality"] = "moderate"
        else:
            item["fix_quality"] = "loose"
        final.append(item)

    total = sum(float(c.get("probability") or 0) for c in final) or 1.0
    for i, c in enumerate(final, start=1):
        c["rank"] = i
        c["probability"] = round(float(c.get("probability") or 0) / total, 6)
        c["probability_pct"] = round(c["probability"] * 100.0, 3)

    # Cluster departure radius from top-1 residual geometry
    if final:
        top = final[0]
        dep_r = float(top["confidence_radius_m"])
        # if top-2 within 120m, shrink to half-distance + 15m
        if len(final) >= 2:
            d12 = distance_between_coords(_latlng(final[0]), _latlng(final[1])) * 1000.0
            if d12 < 120:
                dep_r = min(dep_r, max(20.0, d12 * 0.55 + 12.0))
        departure = {
            "latitude": _latlng(top)[0],
            "longitude": _latlng(top)[1],
            "radius_m": round(dep_r, 2),
            "secondary_radius_m": round(min(dep_r * 1.55, dep_r + 80.0), 2),
            "bearing_error_deg": top.get("bearing_error_deg"),
            "fix_quality": top.get("fix_quality"),
        }
    else:
        departure = None

    return {
        "candidates": final,
        "before_count": len(pool),
        "after_count": len(final),
        "rejected_count": len(rejected),
        "rejected": rejected[:50],
        "notes": notes,
        "departure_tight": departure,
        "aoa_limit_deg": round(aoa_lim, 2),
    }
