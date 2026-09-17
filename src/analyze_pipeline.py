"""
DEFENSIVE AI analysis pipeline (no strike / no munitions).

detect → localize → narrow → egress sketch → civilian risk →
recon mode (drone|human) → auto recon route / foot nav cues
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.predict_api import predict
from src.recon import build_recon_plan
from src.ai_assist import enrich_recon_plan, chat_json
from src.utils import add_distance_to_coord
from src.civilian_risk import recommend_recon_mode

Coord = Tuple[float, float]


def _egress_sketch(
    origin: Coord,
    approach_bearing: float,
    outcome: str = "unknown",
) -> Dict[str, Any]:
    axis = (approach_bearing + 180) % 360
    budgets = {"success": [15, 30, 60], "failure": [5, 15, 30], "unknown": [10, 20, 45]}
    mins = budgets.get(outcome, budgets["unknown"])
    # rough foot/car reach rings (km)
    rings = []
    for t in mins:
        foot_km = 4.5 * (t / 60.0)
        car_km = 40.0 * (t / 60.0) * 0.7
        rings.append(
            {
                "t_min": t,
                "foot_km": round(foot_km, 2),
                "car_km": round(car_km, 2),
                "axis_bearing_deg": round(axis, 1),
                "sample_point": {
                    "lat": add_distance_to_coord(origin, axis, foot_km)[0],
                    "lng": add_distance_to_coord(origin, axis, foot_km)[1],
                },
            }
        )
    return {
        "egress_axis_deg": round(axis, 1),
        "fan_deg": [round((axis + d) % 360, 1) for d in (-45, -20, 0, 20, 45)],
        "isochrones": rings,
        "note": "Defensive track vectors only — not pursuit-to-kill guidance",
    }


def run_defensive_analysis(
    latitude: float,
    longitude: float,
    bearing_degrees: float,
    *,
    signal_strength_dbm: float = -65,
    asset_latitude: Optional[float] = None,
    asset_longitude: Optional[float] = None,
    mission_outcome: str = "unknown",
    lang: str = "en",
    use_ai: bool = False,
    ai_api_key: Optional[str] = None,
    ai_base_url: Optional[str] = None,
    ai_model: Optional[str] = None,
) -> Dict[str, Any]:
    pred = predict(
        latitude=latitude,
        longitude=longitude,
        bearing_degrees=bearing_degrees,
        signal_strength_dbm=signal_strength_dbm,
        asset_latitude=asset_latitude,
        asset_longitude=asset_longitude,
        use_ai=use_ai,
        ai_api_key=ai_api_key,
        ai_base_url=ai_base_url,
        ai_model=ai_model,
        lang=lang,
    )
    if not pred.get("success"):
        return pred

    # Engine already ran civilian/OSINT/paths — reuse, then recon plan
    cands = pred.get("top_10_candidates") or []
    narrowed = cands[:5]
    origin = (latitude, longitude)
    if narrowed:
        origin = (
            float(narrowed[0]["latitude"]),
            float(narrowed[0]["longitude"]),
        )

    egress = pred.get("path_hypotheses") or _egress_sketch(
        origin, bearing_degrees, mission_outcome
    )
    if "egress_axis_deg" not in egress:
        sketch = _egress_sketch(origin, bearing_degrees, mission_outcome)
        egress = {**sketch, "path_hypotheses": pred.get("path_hypotheses")}

    range_km = float(
        (pred.get("drone_info") or {}).get("effective_range_km")
        or (pred.get("drone_info") or {}).get("estimated_range_km")
        or 3
    )
    recon_rec = pred.get("recon_recommendation") or recommend_recon_mode(
        narrowed, range_km
    )

    asset = None
    if asset_latitude is not None and asset_longitude is not None:
        asset = (float(asset_latitude), float(asset_longitude))

    risk_summary = pred.get("civilian_risk") or {"high_risk_count": 0}
    if not isinstance(risk_summary, dict):
        risk_summary = {"high_risk_count": 0}

    plan = build_recon_plan(
        (latitude, longitude),
        bearing_degrees,
        profile=recon_rec["profile"],
        airframe="small_quad" if recon_rec["mode"] == "drone" else "small_quad",
        candidates=narrowed,
        target=asset,
        range_km=min(range_km, 4.0),
        mission_outcome=mission_outcome,
    )

    if use_ai:
        plan = enrich_recon_plan(
            plan,
            lang=lang,
            api_key=ai_api_key,
            base_url=ai_base_url,
            model=ai_model,
            force=True,
        )

    human_nav = None
    if recon_rec["mode"] == "human" and narrowed:
        # Simple foot nav cues between observer → top 3 contacts (verification path)
        legs = []
        prev = (latitude, longitude)
        for i, c in enumerate(narrowed[:3]):
            nxt = (float(c["latitude"]), float(c["longitude"]))
            from src.utils import initial_bearing, distance_between_coords

            legs.append(
                {
                    "from": {"lat": prev[0], "lng": prev[1]},
                    "to": {"lat": nxt[0], "lng": nxt[1]},
                    "bearing_deg": round(initial_bearing(prev, nxt), 1),
                    "distance_m": round(distance_between_coords(prev, nxt) * 1000, 1),
                    "label": f"VERIFY-{i+1}",
                    "civilian_risk": (c.get("civilian_risk") or {}).get("band"),
                }
            )
            prev = nxt
        human_nav = {
            "mode": "foot_verification",
            "legs": legs,
            "caution": "Stay covert; do not engage; high-risk bands = maximize standoff",
        }

    ai_brief = None
    if use_ai:
        ai_brief = chat_json(
            system=(
                "DEFENSIVE analysis only. No weapons, no strike, no aim points. "
                "Help minimize civilian harm via warning, evacuation, recon verification. "
                "Return JSON: brief (string), avoid_zones (array of strings), "
                "next_steps (array of strings). Language: " + lang
            ),
            user={
                "contacts": len(narrowed),
                "high_risk": risk_summary.get("high_risk_count", 0),
                "recon_mode": recon_rec["mode"],
                "egress_axis": egress.get("egress_axis_deg"),
                "comms": pred.get("comms"),
                "environment_flags": (pred.get("environment") or {}).get("ops_flags"),
            },
            api_key=ai_api_key,
            base_url=ai_base_url,
            model=ai_model,
        )

    return {
        "success": True,
        "pipeline": [
            "defensive_engine_v2",
            "env_comms",
            "path_g1_g2_g3",
            "civilian_osint",
            "recon_mode",
            "auto_route",
        ],
        "policy": "DEFENSIVE_ONLY_NO_STRIKE",
        "engine": pred.get("engine"),
        "predict_meta": pred.get("metadata"),
        "place": pred.get("place"),
        "narrowed_candidates": narrowed,
        "civilian_risk": risk_summary,
        "environment": pred.get("environment"),
        "comms": pred.get("comms"),
        "osint": pred.get("osint"),
        "path_hypotheses": pred.get("path_hypotheses"),
        "egress": egress,
        "recon_recommendation": recon_rec,
        "recon_plan": plan,
        "human_nav": human_nav,
        "ai_brief": ai_brief,
        "tactical_layers": pred.get("tactical_layers"),
    }
