"""
Defensive Causal Engine v2 — algorithm-first core.

Orchestrates:
  env (weather/vis) → comms range → localize → human_ops → tight radius
  → path groups G1/G2/G3 → egress → civilian risk → OSM CCTV/traffic
  → optional AI rerank (never invents coords)

AI is assist-only and fail-open.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import API_FORMULA_POOL, API_MAX_CANDIDATES, API_SHORTLIST, VERSION
from src.accuracy_model import accuracy_pack_for_candidate
from src.agent_conference import run_agent_conference
from src.candidate_pool import ai_narrow_candidates, clean_candidate_elevations
from src.candidate_verify import verify_and_prune
from src.civilian_risk import assess_civilian_risk, recommend_recon_mode
from src.comms_visibility import assess_comms
from src.drone_detection import resolve_range_km
from src.env_factors import assess_environment
from src.kinematics import apply_kinematics_prior
from src.localize import localize_drone_operator
from src.osint_sensors import attach_osint_to_candidates, fetch_open_cctv, fetch_traffic_proxy
from src.path_hypotheses import build_path_groups, egress_for_candidate
from src.road_router import fetch_roads_and_water
from src.recon_compress import build_compressed_recon_route
from src.tactics import build_tactical_layers
from src.terrain_context import assess_terrain_context, attach_terrain_to_candidates

ROOT = Path(__file__).resolve().parent.parent
Coord = Tuple[float, float]


def _engine_cfg() -> Dict[str, Any]:
    return json.loads((ROOT / "data" / "engine_constants.json").read_text(encoding="utf-8"))


def tighten_radius_m(
    base_conf_m: float,
    *,
    bearing_err_deg: Optional[float],
    n_observers: int,
    weather_atten_db: float,
    human_score: Optional[float],
) -> Dict[str, Any]:
    cfg = _engine_cfg()["tight_radius"]
    r = float(base_conf_m or cfg["base_m"])
    # AoA residual shrinks radius
    if bearing_err_deg is not None:
        r = min(r, cfg["base_m"] + float(bearing_err_deg) * cfg["aoa_sigma_scale"])
    if n_observers >= 2:
        r *= float(cfg["multi_observer_factor"])
    # weather inflates slightly (uncertainty)
    inflate = 1.0 + min(0.35, weather_atten_db / 20.0)
    inflate = min(inflate, float(cfg["weather_inflate_cap"]))
    r *= inflate
    # strong human fit → slight shrink (behaviorally consistent hide)
    if human_score is not None and human_score >= 0.65:
        r *= 0.92
    r = max(float(cfg["min_m"]), min(float(cfg["max_m"]), r))
    return {
        "radius_m": round(r, 2),
        "inflate_weather": round(inflate, 3),
        "method": "tight_causal_v2",
    }


def run_defensive_engine(
    latitude: float,
    longitude: float,
    bearing_degrees: float,
    *,
    drone_type: str = "unknown",
    signal_strength_dbm: float = -65,
    additional_observers: Optional[List[Dict[str, Any]]] = None,
    asset_latitude: Optional[float] = None,
    asset_longitude: Optional[float] = None,
    flight_track: Optional[List[Dict[str, Any]]] = None,
    use_ai: bool = False,
    ai_api_key: Optional[str] = None,
    ai_base_url: Optional[str] = None,
    ai_model: Optional[str] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    started = time.perf_counter()
    cfg = _engine_cfg()
    observer: Coord = (latitude, longitude)

    if bearing_degrees is None or not (0 <= float(bearing_degrees) <= 359):
        return {
            "success": False,
            "error": "INVALID_BEARING",
            "message": "bearing_degrees must be 0-359",
            "engine": cfg["engine_id"],
        }

    range_km, estimated_type, signal_conf = resolve_range_km(
        drone_type=drone_type,
        signal_strength_dbm=signal_strength_dbm,
    )

    # 1) Environment + 2) visibility/comms (parallel causal inputs)
    env = assess_environment(latitude, longitude, range_km, freq_ghz=2.4)
    wx_atten = float((env.get("rf_attenuation") or {}).get("atten_db") or 0)
    vis_km = (env.get("optical") or {}).get("visibility_km")
    comms = assess_comms(
        range_km,
        urban=True,
        weather_atten_db=wx_atten,
        visibility_km=vis_km,
        signal_dbm=signal_strength_dbm,
    )
    # Soft cap: keep drone-type range, blend with link-budget effective search
    effective_range = float(comms.get("effective_search_km") or range_km)
    effective_range = max(0.8, min(range_km, effective_range))

    n_obs = 1 + (len(additional_observers) if additional_observers else 0)
    tolerance = 18
    if n_obs >= 2:
        tolerance = max(8, 18 // n_obs)

    # 3) Core localize — ask for wide shortlist (not min with 8)
    loc = localize_drone_operator(
        center_lat=latitude,
        center_lng=longitude,
        bearing_deg=bearing_degrees,
        drone_range_km=effective_range,
        num_results=max(API_SHORTLIST, API_FORMULA_POOL),
        bearing_tolerance_deg=tolerance,
        signal_strength_dbm=signal_strength_dbm,
        additional_observers=additional_observers,
        flight_track=flight_track,
    )

    formula_pool = loc.get("formula_pool") or loc.get("top_10_candidates") or []
    candidates = loc.get("top_10_candidates") or []
    # Keep display shortlist sized but retain full formula pool in payload
    if len(candidates) > API_SHORTLIST and not (use_ai or ai_api_key):
        # Already algorithmically shortlisted; trim only if oversized
        candidates = candidates[:API_SHORTLIST]
    primary = loc.get("primary_target")
    if candidates:
        primary = candidates[0]

    # Optional AI — assist only (rerank within formula pool)
    ai_meta = {"applied": False, "reason": "disabled"}
    if use_ai or ai_api_key:
        try:
            ai_pack = ai_narrow_candidates(
                formula_pool,
                keep=API_SHORTLIST,
                lang=lang,
                api_key=ai_api_key,
                base_url=ai_base_url,
                model=ai_model,
            )
            if ai_pack.get("candidates"):
                candidates = [clean_candidate_elevations(c) for c in ai_pack["candidates"]]
                primary = candidates[0] if candidates else primary
            ai_meta = {
                "applied": bool(ai_pack.get("applied")),
                "reason": ai_pack.get("reason") or ai_pack.get("method"),
                "role": "optional_rerank",
            }
        except Exception as exc:  # noqa: BLE001
            ai_meta = {"applied": False, "reason": f"fail_open:{str(exc)[:80]}"}
            candidates = [clean_candidate_elevations(c) for c in candidates]
    else:
        candidates = [clean_candidate_elevations(c) for c in candidates]

    kin_pack = apply_kinematics_prior(
        candidates, flight_track, observer=observer
    )
    if kin_pack.get("applied"):
        candidates = kin_pack["candidates"]
        primary = candidates[0] if candidates else primary

    # 4) Tight radius + accuracy pack on each candidate
    tightened = []
    for c in candidates:
        tr = tighten_radius_m(
            float(c.get("confidence_radius_m") or 80),
            bearing_err_deg=c.get("bearing_error_deg"),
            n_observers=n_obs,
            weather_atten_db=wx_atten,
            human_score=c.get("human_score"),
        )
        item = dict(c)
        acc = accuracy_pack_for_candidate(
            item, rssi_dbm=signal_strength_dbm, n_observers=n_obs
        )
        # Blend model radius with tightened radius (take tighter when confident)
        blended = min(float(tr["radius_m"]), float(acc["model_radius_m"]))
        if acc["confidence"] >= 0.55:
            blended = min(blended, float(tr["radius_m"]) * 0.92)
        item["confidence_radius_m"] = round(blended, 1)
        item["radius_tight"] = tr
        item["accuracy"] = acc
        tightened.append(item)
    candidates = tightened
    if candidates:
        primary = candidates[0]

    asset = None
    if asset_latitude is not None and asset_longitude is not None:
        asset = (float(asset_latitude), float(asset_longitude))

    risk_pack = assess_civilian_risk(candidates, asset)
    candidates = risk_pack["candidates"]
    if candidates:
        primary = candidates[0]

    focus = primary or {"latitude": latitude, "longitude": longitude}
    flat = float(focus.get("latitude") or latitude)
    flng = float(focus.get("longitude") or longitude)

    try:
        cctv = fetch_open_cctv((flat, flng), cfg["osint"]["cctv_radius_m"])
    except Exception as exc:  # noqa: BLE001
        cctv = {"ok": False, "cameras": [], "count": 0, "error": str(exc)[:100]}
    try:
        traffic = fetch_traffic_proxy((flat, flng), cfg["osint"]["traffic_radius_m"])
    except Exception as exc:  # noqa: BLE001
        traffic = {"ok": False, "score": 0.3, "band": "unknown", "error": str(exc)[:100]}

    candidates = attach_osint_to_candidates(candidates, cctv, traffic)

    # Terrain / forest / plain / concealment (bunker·tunnel OSINT — defensive only)
    try:
        terrain_ctx = assess_terrain_context(flat, flng, radius_m=900, concealment_radius_m=1000)
    except Exception as exc:  # noqa: BLE001
        terrain_ctx = {"ok": False, "error": str(exc)[:120], "band": {"band": "mixed"}}
    candidates = attach_terrain_to_candidates(candidates, terrain_ctx)
    if candidates:
        primary = candidates[0]

    # Parallel algorithmic conference → path style / egress priority / prune / recon
    conference = run_agent_conference(
        {
            "candidates": candidates,
            "traffic": traffic,
            "cctv": cctv,
            "environment": env,
            "terrain": terrain_ctx,
            "observer": {"latitude": latitude, "longitude": longitude},
            "bearing_deg": bearing_degrees,
        }
    )
    consensus = conference.get("consensus") or {}

    roads: List[Dict[str, Any]] = []
    water: List[Dict[str, Any]] = []
    try:
        rw = fetch_roads_and_water((flat, flng), radius_m=3500)
        roads = list(rw.get("roads") or [])
        water = list(rw.get("water") or [])
    except Exception:  # noqa: BLE001
        roads, water = [], []

    # Building keep-outs from formula pool + shortlist (centers + height)
    building_pts = []
    seen_b = set()
    for c in list(formula_pool) + list(candidates):
        la = c.get("latitude") or c.get("lat")
        ln = c.get("longitude") or c.get("lng")
        if la is None or ln is None:
            continue
        key = (round(float(la), 5), round(float(ln), 5))
        if key in seen_b:
            continue
        seen_b.add(key)
        building_pts.append(
            {
                "lat": float(la),
                "lng": float(ln),
                "height_m": float(c.get("height_m") or 12.0),
            }
        )

    # Per-candidate egress scores (map lines come from path_groups primary only)
    enriched = []
    for c in candidates[:12]:
        try:
            eg = egress_for_candidate(
                c,
                bearing_degrees,
                roads,
                building_pts,
                terrain=terrain_ctx,
                traffic=traffic,
                water=water,
                emit_features=False,
            )
        except Exception:  # noqa: BLE001
            eg = {"feasibility": {}, "features": [], "routes": {}}
        item = dict(c)
        feas = eg.get("feasibility") or {}
        item["egress_modes"] = {
            "foot": float(feas.get("foot") or 0),
            "bike": float(feas.get("bike") or 0),
            "motorcycle": float(feas.get("motorcycle") or 0),
            "car": float(feas.get("car") or 0),
            "armored": float(feas.get("armored") or 0),
            "egress_foot": float(feas.get("foot") or 0),
            "egress_bike": float(feas.get("bike") or 0),
            "egress_motorcycle": float(feas.get("motorcycle") or 0),
            "egress_car": float(feas.get("car") or 0),
            "egress_armored": float(feas.get("armored") or 0),
            "egress_mixed": round(float(feas.get("overall") or 0), 3),
            "overall": float(feas.get("overall") or 0),
        }
        item["egress_pack"] = {
            "terrain_class": eg.get("terrain_class"),
            "feasibility": feas,
            "axis_deg": eg.get("axis_deg"),
            "method": "strict_osm_road_dijkstra",
            "water_barrier_count": len(water),
        }
        item["egress_foot"] = item["egress_modes"]["egress_foot"]
        item["egress_car"] = item["egress_modes"]["egress_car"]
        item["egress_bike"] = item["egress_modes"]["egress_bike"]
        item["egress_motorcycle"] = item["egress_modes"]["egress_motorcycle"]
        item["egress_armored"] = item["egress_modes"]["egress_armored"]
        enriched.append(item)
    candidates = enriched or candidates

    paths = build_path_groups(
        observer,
        bearing_degrees,
        effective_range,
        primary=primary,
        weather_flags=env.get("ops_flags"),
        road_segments=roads,
        buildings=building_pts,
        terrain=terrain_ctx,
        traffic=traffic,
        candidates=candidates[:5],
        water=water,
    )

    # Detailed unique verify + causal prune (conference rules)
    verify_ctx = {
        "observer": observer,
        "bearing_deg": bearing_degrees,
        "signal_dbm": signal_strength_dbm,
        "environment": env,
        "traffic": traffic,
        "cctv": cctv,
        "terrain": terrain_ctx,
        "prune_rules_active": consensus.get("prune_rules_active") or {},
    }
    verify_pack = verify_and_prune(candidates, verify_ctx)
    candidates = verify_pack.get("kept") or candidates
    if candidates:
        primary = candidates[0]

    recon_rec = recommend_recon_mode(candidates, effective_range)
    recon_profile = consensus.get("recon_profile") or recon_rec.get("profile") or "orbit_contacts"

    recon_auto = build_compressed_recon_route(
        observer,
        bearing_degrees,
        candidates,
        airframe="small_quad",
        weather=env.get("weather") or env,
        context={"environment": env, "terrain": terrain_ctx},
        recon_profile=str(recon_profile),
        min_wp=3,
        max_wp=5,
    )

    layers = build_tactical_layers(
        observer=observer,
        bearing_deg=bearing_degrees,
        drone_range_km=effective_range,
        signal_dbm=signal_strength_dbm,
        candidates=candidates,
        primary=primary,
        asset=asset,
    )
    candidates = layers["candidates_styled"]
    if candidates:
        primary = candidates[0]

    dep = dict(loc.get("departure_tight") or {})
    if primary:
        dep.setdefault("latitude", primary.get("latitude"))
        dep.setdefault("longitude", primary.get("longitude"))
        dep["radius_m"] = primary.get("confidence_radius_m") or dep.get("radius_m")
        dep["secondary_radius_m"] = round(float(dep["radius_m"]) * 1.45, 2)

    elapsed = int((time.perf_counter() - started) * 1000)
    conf_label = (
        "high" if signal_conf >= 0.7 else ("medium" if signal_conf >= 0.4 else "low")
    )
    band = (terrain_ctx.get("band") or {}).get("band") if isinstance(terrain_ctx.get("band"), dict) else terrain_ctx.get("band")

    return {
        "success": True,
        "version": VERSION,
        "engine": {
            "id": "defensive_causal_v3",
            "ai_role": cfg["ai_role"],
            "ai": ai_meta,
            "security": cfg["security"],
            "causal_graph": cfg.get("causal_graph"),
            "papers_basis": cfg.get("papers_basis"),
            "conference": {
                "consensus": consensus,
                "audit_log": (conference.get("audit_log") or [])[:8],
            },
        },
        "primary_target": primary,
        "top_10_candidates": candidates,
        "formula_pool": formula_pool,
        "formula_pool_count": loc.get("formula_pool_count", len(formula_pool)),
        "algorithm_shortlist": loc.get("algorithm_shortlist"),
        "human_ops": loc.get("human_ops"),
        "environment": env,
        "comms": comms,
        "terrain": terrain_ctx,
        "path_hypotheses": paths,
        "verify": {
            "kept": len(verify_pack.get("kept") or []),
            "pruned": len(verify_pack.get("pruned") or []),
            "notes": verify_pack.get("notes") or [],
            "pruned_sample": (verify_pack.get("pruned") or [])[:8],
        },
        "civilian_risk": risk_pack["summary"],
        "osint": {
            "cctv": cctv,
            "traffic": traffic,
            "concealment": terrain_ctx.get("concealment_osint"),
            "landcover": terrain_ctx.get("landcover"),
            "policy": "Public OSM only — no private feeds; concealment is hide-awareness not targeting",
        },
        "recon_recommendation": {
            **recon_rec,
            "profile": recon_profile,
            "egress_mode_priority": consensus.get("egress_mode_priority"),
            "path_style": consensus.get("path_style"),
        },
        "recon_auto": recon_auto,
        "drone_info": {
            "estimated_type": estimated_type,
            "estimated_range_km": range_km,
            "effective_range_km": effective_range,
            "signal_confidence": conf_label,
        },
        "tactical_layers": {
            **layers,
            "path_groups": paths,
            "recon_auto": recon_auto.get("geojson") if isinstance(recon_auto, dict) else None,
        },
        "departure_tight": dep,
        "refine": loc.get("refine"),
        "search_sector": loc.get("search_sector"),
        "terrain_grid": loc.get("terrain_grid"),
        "imagery": loc.get("imagery"),
        "bbox": loc.get("bbox"),
        "attribution": (loc.get("attribution") or [])
        + [
            "Weather: Open-Meteo",
            "CCTV/traffic/landcover/concealment tags: OpenStreetMap",
        ],
        "agents": loc.get("agents"),
        "review": loc.get("review"),
        "triangulation": loc.get("triangulation"),
        "kinematics": loc.get("kinematics") or kin_pack.get("features"),
        "observer": {"latitude": latitude, "longitude": longitude},
        "asset": (
            {"latitude": asset[0], "longitude": asset[1]} if asset else None
        ),
        "metadata": {
            "processing_time_ms": elapsed,
            "pipeline": [
                "env_weather_wind_daynight",
                "comms_link_budget",
                "osm_localize_human_ops",
                "algorithmic_shortlist",
                "optional_ai_failopen",
                "tight_radius_causal",
                "civilian_density",
                "osint_cctv_traffic",
                "terrain_landcover_concealment",
                "agent_conference",
                "per_candidate_egress_foot_bike_moto_car_mil",
                "obstacle_aware_path_groups",
                "verify_and_prune",
                "recon_compress_auto",
                "tactical_layers",
            ],
            "ai_required": False,
            "osm_building_count": loc.get("osm_building_count", 0),
            "formula_pool_count": loc.get("formula_pool_count", len(formula_pool)),
            "final_count": len(candidates),
            "verify_pruned": len(verify_pack.get("pruned") or []),
            "terrain_band": band,
            "bearing_tolerance_deg": tolerance,
            "live_data": not bool((loc.get("sources") or {}).get("osm_degraded")),
            "osm_degraded": bool((loc.get("sources") or {}).get("osm_degraded")),
            "osm_error": (loc.get("sources") or {}).get("osm_error"),
            "accuracy_note": (
                f"Engine v3: pool→{len(candidates)} "
                f"terrain={band} ctrl={comms['control_range_km']}km "
                f"recon_wp={len((recon_auto or {}).get('waypoints') or [])} "
                f"AI={ai_meta.get('applied')}"
                + (
                    " · OSM_DEGRADED"
                    if (loc.get("sources") or {}).get("osm_degraded")
                    else ""
                )
            ),
        },
    }
