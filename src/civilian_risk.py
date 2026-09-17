"""
Civilian density RISK for DEFENSIVE use only.

Purpose: flag high-density areas so operators AVOID collateral,
prioritize warning/evacuation, and choose non-kinetic verification (recon).
Does NOT compute strike solutions, munitions, or aim points.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.utils import distance_between_coords

Coord = Tuple[float, float]

# Heuristic population proxies (not census) — residential weight higher
TYPE_WEIGHT = {
    "residential": 1.0,
    "commercial": 0.55,
    "industrial": 0.15,
    "unknown": 0.35,
}


def _density_score(candidate: Dict[str, Any], peers: List[Dict[str, Any]]) -> float:
    """Local peer count within 250m weighted by building type (0..1)."""
    lat = float(candidate.get("latitude") or candidate.get("lat") or 0)
    lng = float(candidate.get("longitude") or candidate.get("lng") or 0)
    if not lat and not lng:
        return 0.3
    origin = (lat, lng)
    acc = 0.0
    for p in peers:
        plat = p.get("latitude") or p.get("lat")
        plng = p.get("longitude") or p.get("lng")
        if plat is None or plng is None:
            continue
        d_m = distance_between_coords(origin, (float(plat), float(plng))) * 1000.0
        if d_m > 250:
            continue
        bt = str(p.get("building_type") or "unknown")
        w = TYPE_WEIGHT.get(bt, 0.35)
        # closer → more weight
        acc += w * max(0.0, 1.0 - d_m / 250.0)
    # self type
    bt0 = str(candidate.get("building_type") or "unknown")
    acc += TYPE_WEIGHT.get(bt0, 0.35)
    return max(0.0, min(1.0, acc / 4.0))


def assess_civilian_risk(
    candidates: List[Dict[str, Any]],
    protected_asset: Optional[Coord] = None,
) -> Dict[str, Any]:
    """
    Attach risk labels. High risk → recon-only / evacuate advice.
    """
    styled: List[Dict[str, Any]] = []
    for c in candidates:
        item = dict(c)
        dens = _density_score(c, candidates)
        # distance to protected asset increases protective urgency, not strike value
        asset_factor = 0.0
        if protected_asset:
            lat = float(c.get("latitude") or 0)
            lng = float(c.get("longitude") or 0)
            d_km = distance_between_coords((lat, lng), protected_asset)
            asset_factor = max(0.0, 1.0 - min(d_km, 5.0) / 5.0) * 0.2

        risk = max(0.0, min(1.0, dens * 0.85 + asset_factor))
        if risk >= 0.7:
            band = "high"
            action = "avoid_kinetic_near_civilians; warn_evacuate; recon_only"
            color = "#ff4d4d"
        elif risk >= 0.4:
            band = "medium"
            action = "prefer_standoff_recon; limit_exposure"
            color = "#f0c14b"
        else:
            band = "low"
            action = "standard_defensive_recon_ok"
            color = "#6bcf6b"

        item["civilian_risk"] = {
            "score": round(risk, 3),
            "density_proxy": round(dens, 3),
            "band": band,
            "defensive_action": action,
            "color": color,
        }
        # Down-rank for any automated "engagement" — we only expose as caution weight
        item["defensive_priority"] = round(
            float(c.get("probability") or 0) * (1.0 - 0.35 * risk), 4
        )
        styled.append(item)

    styled.sort(key=lambda x: x.get("defensive_priority", 0), reverse=True)
    high_n = sum(1 for x in styled if x["civilian_risk"]["band"] == "high")
    return {
        "candidates": styled,
        "summary": {
            "high_risk_count": high_n,
            "policy": "DEFENSIVE_ONLY — minimize harm by avoiding dense areas; no strike planning",
            "recommendation": (
                "Prioritize warning/evacuation near high-risk bands; "
                "verify with ISR recon before any defensive interdiction decision by humans."
            ),
        },
    }


def recommend_recon_mode(
    candidates: List[Dict[str, Any]],
    range_km: float,
) -> Dict[str, Any]:
    """
    Choose drone vs human scout for VERIFICATION only (not strike).
    """
    if not candidates:
        return {
            "mode": "drone",
            "reason": "No contacts — wide-area ISR sweep preferred",
            "profile": "verify_approach",
        }
    high = sum(
        1
        for c in candidates
        if (c.get("civilian_risk") or {}).get("band") == "high"
    )
    top = candidates[0]
    # Human preferred when high civilian density (quieter, street-level ID) and short range
    if high >= 2 or (
        (top.get("civilian_risk") or {}).get("band") == "high" and range_km <= 2.5
    ):
        return {
            "mode": "human",
            "reason": "High civilian-density proxy near contacts — ground verification safer for bystanders",
            "profile": "orbit_contacts",
            "nav_style": "foot_approach_cover",
        }
    if range_km > 2.0:
        return {
            "mode": "drone",
            "reason": "Standoff distance favors aerial ISR verification",
            "profile": "verify_approach",
            "nav_style": "aerial_corridor",
        }
    return {
        "mode": "drone",
        "reason": "Default aerial recon for candidate confirmation",
        "profile": "orbit_contacts",
        "nav_style": "aerial_orbit",
    }
