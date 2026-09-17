"""
Candidate verification — attach unique causal reasons, then prune.

Wire-ready: call verify_and_prune(ranked_candidates, context) after scoring /
human_ops / OSINT attach. Parent may integrate into defensive_engine later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.utils import distance_between_coords, initial_bearing

Coord = Tuple[float, float]

# Soft thresholds (causal, auditable)
BEARING_HARD_FAIL_DEG = 14.0
BEARING_WARN_DEG = 8.0
RSSI_FIT_HARD_FAIL = 0.18
SKYLINE_CCTV_DOWNRANK = 0.55  # multiply probability
CONCEAL_BOOST = 1.28
DENSE_TRAFFIC_BANDS = {"heavy", "dense", "high"}
FOOT_EGRESS_KEYS = ("egress_foot", "foot", "path", "trail", "pedestrian")
CAR_EGRESS_KEYS = ("egress_car", "car", "vehicle", "road")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _latlng(c: Dict[str, Any]) -> Coord:
    return (
        float(c.get("latitude") or c.get("lat") or 0),
        float(c.get("longitude") or c.get("lng") or 0),
    )


def _attrs(c: Dict[str, Any]) -> Dict[str, Any]:
    return dict(c.get("attributes") or {})


def _bearing_residual_deg(c: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[float]:
    if c.get("bearing_error_deg") is not None:
        return float(c["bearing_error_deg"])
    comps = (c.get("scores") or {}).get("components") or {}
    if comps.get("p_bearing_error_deg") is not None:
        return float(comps["p_bearing_error_deg"])
    prec = c.get("precision") or {}
    if prec.get("bearing_error_deg") is not None:
        return float(prec["bearing_error_deg"])
    obs = ctx.get("observer")
    bearing = ctx.get("bearing_deg")
    if obs is None or bearing is None:
        return None
    lat, lng = _latlng(c)
    if lat == 0 and lng == 0:
        return None
    brg = initial_bearing(tuple(obs), (lat, lng))
    d = abs((brg - float(bearing) + 180) % 360 - 180)
    return round(d, 3)


def _rssi_fit(c: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    comps = (c.get("scores") or {}).get("components") or {}
    if comps.get("p_L_rssi") is not None:
        return _clamp01(float(comps["p_L_rssi"]))
    detail = c.get("rf_detail") or c.get("detail") or {}
    if detail.get("L_rssi") is not None:
        return _clamp01(float(detail["L_rssi"]))
    # distance vs expected annulus if parent left annulus on context
    ann = ctx.get("rssi_annulus_km")
    obs = ctx.get("observer")
    if ann and obs:
        d = distance_between_coords(tuple(obs), _latlng(c))
        d0, d1 = float(ann[0]), float(ann[1])
        if d0 <= d <= d1:
            mid = 0.5 * (d0 + d1)
            span = max(d1 - d0, 1e-3)
            return _clamp01(1.0 - abs(d - mid) / span)
        return 0.12
    return 0.5


def _human_parts(c: Dict[str, Any]) -> Dict[str, float]:
    parts = c.get("human_parts") or {}
    if parts:
        return {k: float(v) for k, v in parts.items()}
    return {}


def _terrain_band(c: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    for key in ("terrain_band", "elev_band", "elevation_band"):
        if c.get(key):
            return str(c[key])
    elev = c.get("elevation") or {}
    g = elev.get("ground_m")
    if g is None:
        g = c.get("ground_elevation_m")
    grid = ctx.get("terrain_grid") or {}
    samples = grid.get("samples") or grid.get("elevations") or []
    vals = []
    for s in samples:
        if isinstance(s, dict) and s.get("elevation_m") is not None:
            vals.append(float(s["elevation_m"]))
        elif isinstance(s, (int, float)):
            vals.append(float(s))
    if g is None:
        return "unknown"
    g = float(g)
    if not vals:
        h = float(c.get("height_m") or elev.get("roof_m") or 10)
        if h > 40:
            return "skyline"
        if h < 6:
            return "low"
        return "mid"
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1.0)
    rel = (g - lo) / span
    if rel >= 0.88 and (hi - float(sorted(vals)[len(vals) // 2])) > 6:
        return "skyline"
    if rel <= 0.25:
        return "low"
    if rel >= 0.65:
        return "high"
    return "mid"


def _nearby_flags(c: Dict[str, Any]) -> Dict[str, bool]:
    attrs = _attrs(c)
    tags = c.get("nearby") or attrs.get("nearby") or {}
    mil = str(attrs.get("military") or c.get("military") or "").lower()

    def _flag(*keys: str) -> bool:
        for k in keys:
            if c.get(k) or attrs.get(k) or tags.get(k):
                return True
        return False

    forest = _flag("forest_nearby", "near_forest", "woodland", "landuse_forest")
    bunker = _flag("bunker_nearby", "near_bunker", "bunker") or "bunker" in mil
    tunnel = _flag("tunnel_nearby", "near_tunnel", "tunnel", "subway_entrance")
    # OSM natural/landuse hints on attributes
    landuse = str(attrs.get("landuse") or attrs.get("natural") or "").lower()
    if landuse in ("forest", "wood", "scrub"):
        forest = True
    if landuse in ("military",) and "bunker" in str(attrs).lower():
        bunker = True
    return {"forest": forest, "bunker": bunker, "tunnel": tunnel}


def _traffic_band(c: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    tc = c.get("traffic_context") or {}
    band = tc.get("band") or (ctx.get("traffic") or {}).get("band")
    return str(band or "unknown").lower()


def _cctv_count(c: Dict[str, Any], ctx: Dict[str, Any]) -> int:
    near = c.get("nearby_cctv") or {}
    if near.get("count") is not None:
        return int(near["count"])
    cams = (ctx.get("cctv") or {}).get("cameras") or []
    if not cams:
        return 0
    lat, lng = _latlng(c)
    n = 0
    for cam in cams:
        d_m = distance_between_coords((lat, lng), (cam["lat"], cam["lng"])) * 1000
        if d_m <= 350:
            n += 1
    return n


def _egress_modes(c: Dict[str, Any]) -> Dict[str, float]:
    """
    Per-mode egress feasibility in [0,1].
    Accepts explicit maps or derives from road/path distance.
    """
    modes = c.get("egress_modes") or c.get("egress_feasibility") or {}
    if modes:
        return {str(k): _clamp01(float(v)) for k, v in modes.items()}
    attrs = _attrs(c)
    road = attrs.get("dist_to_road_m")
    path = attrs.get("dist_to_path_m") or attrs.get("dist_to_footway_m")
    car = 0.55
    foot = 0.55
    if road is not None:
        car = _clamp01(1.0 - float(road) / 900.0)
    if path is not None:
        foot = _clamp01(1.0 - float(path) / 600.0)
    elif road is not None and float(road) < 250:
        foot = 0.45  # roadside but no dedicated foot tag
    return {
        "egress_car": round(car, 3),
        "egress_foot": round(foot, 3),
        "egress_mixed": round(0.5 * car + 0.5 * foot, 3),
    }


def _weather_wind_alignment(c: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    env = ctx.get("environment") or ctx.get("weather") or {}
    wx = env.get("weather") if isinstance(env.get("weather"), dict) else env
    wind_mps = float(wx.get("wind_mps") or env.get("wind_mps") or 0)
    gust = float(wx.get("gust_mps") or env.get("gust_mps") or wind_mps)
    wind_dir = (
        wx.get("wind_direction_deg")
        or wx.get("wind_dir_deg")
        or env.get("wind_direction_deg")
        or env.get("wind_dir_deg")
    )
    # conference / causal pack may nest under env["wind"]
    wind_pack = env.get("wind") if isinstance(env.get("wind"), dict) else {}
    if wind_dir is None:
        wind_dir = wind_pack.get("wind_direction_deg")
    obs = ctx.get("observer")
    bearing = ctx.get("bearing_deg")
    align = 0.5
    note = "wind_neutral"
    if wind_dir is not None and obs is not None:
        lat, lng = _latlng(c)
        brg = initial_bearing(tuple(obs), (lat, lng)) if bearing is None else float(bearing)
        # Crosswind vs head/tail on approach axis
        delta = abs((float(wind_dir) - brg + 180) % 360 - 180)
        # Prefer slight crosswind for small_quad ISR stability (not pure headwind)
        align = _clamp01(1.0 - abs(delta - 90) / 90.0)
        note = "crosswind_favored" if 60 <= delta <= 120 else "axial_wind"
    severity = _clamp01((gust - 8.0) / 12.0) if gust else 0.0
    score = _clamp01(align * (1.0 - 0.45 * severity))
    return {
        "score": round(score, 3),
        "wind_mps": wind_mps,
        "gust_mps": gust,
        "align": round(align, 3),
        "note": note,
    }


def _skyline_penalty(c: Dict[str, Any], terrain_band: str, human: Dict[str, float]) -> float:
    """1.0 = no penalty; lower = downrank."""
    mid = human.get("mid_elevation")
    h = float(c.get("height_m") or (c.get("elevation") or {}).get("roof_m") or 10)
    pen = 1.0
    if terrain_band == "skyline":
        pen *= 0.62
    if mid is not None and mid < 0.38:
        pen *= 0.75
    if h > 45:
        pen *= 0.7
    return round(_clamp01(pen), 3)


def _reason(code: str, text: str, severity: str = "info") -> Dict[str, str]:
    return {"code": code, "text": text, "severity": severity}


def build_unique_reasons(
    c: Dict[str, Any],
    ctx: Dict[str, Any],
    factors: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Detailed unique reasons — machine codes + English text.
    Values are interpolated so candidates do not share the same 4 phrases.
    """
    cid = str(c.get("id") or c.get("rank") or "?")
    lat, lng = _latlng(c)
    be = factors["bearing_residual_deg"]
    rssi = factors["rssi_fit"]
    band = factors["terrain_band"]
    flags = factors["nearby"]
    traffic = factors["traffic_band"]
    cctv_n = factors["cctv_count"]
    egress = factors["egress_modes"]
    wind = factors["weather_wind"]
    sky_pen = factors["skyline_penalty"]
    human = factors["human_parts"]
    reasons: List[Dict[str, str]] = []

    if be is not None:
        reasons.append(
            _reason(
                f"BEARING_RESIDUAL_{be:.1f}DEG",
                (
                    f"Candidate {cid} at ({lat:.5f},{lng:.5f}): AoA residual "
                    f"{be:.1f}° from observed bearing"
                    + (
                        f" {ctx.get('bearing_deg')}°"
                        if ctx.get("bearing_deg") is not None
                        else ""
                    )
                    + "."
                ),
                "fail" if be > BEARING_HARD_FAIL_DEG else ("warn" if be > BEARING_WARN_DEG else "info"),
            )
        )
    reasons.append(
        _reason(
            f"RSSI_FIT_{rssi:.2f}",
            f"RSSI path-loss fit score {rssi:.2f} "
            f"(annulus/context-aware residual for id={cid}).",
            "fail" if rssi < RSSI_FIT_HARD_FAIL else "info",
        )
    )

    if human:
        top = sorted(human.items(), key=lambda kv: kv[1], reverse=True)[:3]
        for k, v in top:
            reasons.append(
                _reason(
                    f"HUMAN_OPS_{k.upper()}_{v:.2f}",
                    f"Human-ops part '{k}'={v:.2f} drives OP-site prior for {cid}.",
                )
            )
    else:
        reasons.append(
            _reason(
                "HUMAN_OPS_UNAVAILABLE",
                f"No human_parts on {cid}; verification used RF/OSINT factors only.",
                "warn",
            )
        )

    reasons.append(
        _reason(
            f"TERRAIN_BAND_{band.upper()}",
            f"Local terrain band classified as '{band}' "
            f"(mid preferred; skyline attracts ISR).",
            "warn" if band == "skyline" else "info",
        )
    )

    if flags["forest"] or flags["bunker"] or flags["tunnel"]:
        bits = [k for k, v in flags.items() if v]
        reasons.append(
            _reason(
                "NEARBY_" + "_".join(b.upper() for b in bits),
                (
                    f"Concealment geometry near {cid}: "
                    + ", ".join(bits)
                    + " — boosts OP-hide hypothesis when forest+bunker co-occur."
                ),
            )
        )
    else:
        reasons.append(
            _reason(
                "NEARBY_OPEN_FABRIC",
                f"No forest/bunker/tunnel tags near {cid}; concealment prior neutral.",
            )
        )

    reasons.append(
        _reason(
            f"TRAFFIC_BAND_{traffic.upper()}",
            f"Traffic density band '{traffic}' affects car egress feasibility for {cid}.",
            "warn" if traffic in DENSE_TRAFFIC_BANDS else "info",
        )
    )
    reasons.append(
        _reason(
            f"CCTV_COUNT_{cctv_n}",
            f"{cctv_n} OSM surveillance tag(s) within ~350m of {cid}.",
            "warn" if cctv_n >= 3 else "info",
        )
    )

    car = float(egress.get("egress_car") or egress.get("car") or 0)
    foot = float(egress.get("egress_foot") or egress.get("foot") or 0)
    reasons.append(
        _reason(
            f"EGRESS_CAR_{car:.2f}_FOOT_{foot:.2f}",
            (
                f"Per-mode egress feasibility car={car:.2f}, foot={foot:.2f} "
                f"for {cid}."
            ),
            "fail"
            if (traffic in DENSE_TRAFFIC_BANDS and car >= 0.55 and foot < 0.35)
            else "info",
        )
    )

    reasons.append(
        _reason(
            f"WEATHER_WIND_{wind['note'].upper()}_{wind['score']:.2f}",
            (
                f"Wind/gust alignment score {wind['score']:.2f} "
                f"(wind={wind['wind_mps']:.1f} m/s, gust={wind['gust_mps']:.1f}, "
                f"{wind['note']})."
            ),
            "warn" if wind["gust_mps"] >= 12 else "info",
        )
    )

    if sky_pen < 0.95:
        reasons.append(
            _reason(
                f"SKYLINE_MID_ELEV_PENALTY_{sky_pen:.2f}",
                (
                    f"Mid-elevation vs skyline penalty factor {sky_pen:.2f} "
                    f"applied to {cid} (band={band})."
                ),
                "warn",
            )
        )

    if band == "skyline" and cctv_n >= 2:
        reasons.append(
            _reason(
                "RULE_SKYLINE_HIGH_CCTV_DOWNRANK",
                (
                    f"Causal rule: skyline + high CCTV ({cctv_n}) → downrank "
                    f"{cid} (attribution / observation risk)."
                ),
                "warn",
            )
        )
    if flags["forest"] and flags["bunker"]:
        reasons.append(
            _reason(
                "RULE_FOREST_BUNKER_CONCEAL_BOOST",
                (
                    f"Causal rule: forest+bunker nearby → concealment boost "
                    f"for OP-hide hypothesis on {cid}."
                ),
            )
        )

    return reasons


def extract_factors(c: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    human = _human_parts(c)
    band = _terrain_band(c, ctx)
    return {
        "bearing_residual_deg": _bearing_residual_deg(c, ctx),
        "rssi_fit": _rssi_fit(c, ctx),
        "human_parts": human,
        "terrain_band": band,
        "nearby": _nearby_flags(c),
        "traffic_band": _traffic_band(c, ctx),
        "cctv_count": _cctv_count(c, ctx),
        "egress_modes": _egress_modes(c),
        "weather_wind": _weather_wind_alignment(c, ctx),
        "skyline_penalty": _skyline_penalty(c, band, human),
    }


def attach_verification_reasons(
    candidates: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Annotate each candidate with unique verify_reasons + factor snapshot."""
    ctx = context or {}
    out: List[Dict[str, Any]] = []
    for c in candidates:
        item = dict(c)
        factors = extract_factors(item, ctx)
        reasons = build_unique_reasons(item, ctx, factors)
        item["verify_factors"] = factors
        item["verify_reasons"] = reasons
        # also flatten codes for quick UI
        item["verify_reason_codes"] = [r["code"] for r in reasons]
        out.append(item)
    return out


def _active_rules(ctx: Dict[str, Any]) -> Dict[str, bool]:
    defaults = {
        "bearing_hard_fail": True,
        "rssi_hard_fail": True,
        "no_car_egress_dense_traffic_without_foot": True,
        "skyline_high_cctv_downrank": True,
        "forest_bunker_conceal_boost": True,
        "skyline_mid_elev_penalty": True,
        "extreme_gust_prune": False,
    }
    override = ctx.get("prune_rules_active") or {}
    merged = dict(defaults)
    merged.update({k: bool(v) for k, v in override.items()})
    return merged


def _apply_causal_rules(
    c: Dict[str, Any],
    factors: Dict[str, Any],
    rules: Dict[str, bool],
) -> Tuple[str, List[str], float]:
    """
    Returns (decision, rule_hits, probability_multiplier).
    decision: keep | prune | downrank
    """
    hits: List[str] = []
    mult = 1.0
    be = factors["bearing_residual_deg"]
    rssi = factors["rssi_fit"]
    traffic = factors["traffic_band"]
    egress = factors["egress_modes"]
    flags = factors["nearby"]
    band = factors["terrain_band"]
    cctv_n = factors["cctv_count"]
    wind = factors["weather_wind"]
    sky_pen = factors["skyline_penalty"]

    car = float(egress.get("egress_car") or egress.get("car") or 0)
    foot = float(egress.get("egress_foot") or egress.get("foot") or 0)

    # Hard fail: bearing
    if rules.get("bearing_hard_fail") and be is not None and be > BEARING_HARD_FAIL_DEG:
        hits.append("FAIL:bearing_residual_exceeds_threshold")
        return "prune", hits, 0.0

    # Hard fail: RSSI
    if rules.get("rssi_hard_fail") and rssi < RSSI_FIT_HARD_FAIL:
        hits.append("FAIL:rssi_fit_below_threshold")
        return "prune", hits, 0.0

    # Fail: car-only egress in dense traffic without alt foot path
    if rules.get("no_car_egress_dense_traffic_without_foot"):
        if traffic in DENSE_TRAFFIC_BANDS and car >= 0.55 and foot < 0.35:
            hits.append("FAIL:no_egress_car_in_dense_traffic_without_alt_foot_path")
            return "prune", hits, 0.0

    if rules.get("extreme_gust_prune") and wind.get("gust_mps", 0) >= 18:
        hits.append("FAIL:extreme_gust_unsafe_for_small_quad_recon")
        return "prune", hits, 0.0

    # Soft: skyline + high CCTV → downrank
    if rules.get("skyline_high_cctv_downrank") and band == "skyline" and cctv_n >= 2:
        hits.append("DOWNRANK:skyline_plus_high_cctv")
        mult *= SKYLINE_CCTV_DOWNRANK

    if rules.get("skyline_mid_elev_penalty") and sky_pen < 1.0:
        hits.append(f"DOWNRANK:skyline_mid_elev_penalty_{sky_pen:.2f}")
        mult *= sky_pen

    # Boost: forest + bunker → concealment for OP hide
    if rules.get("forest_bunker_conceal_boost") and flags.get("forest") and flags.get("bunker"):
        hits.append("BOOST:forest_bunker_concealment_op_hide")
        mult *= CONCEAL_BOOST

    # Mild tunnel boost for hide hypothesis
    if flags.get("tunnel") and flags.get("forest"):
        hits.append("BOOST:forest_tunnel_concealment")
        mult *= 1.12

    if mult < 0.999:
        return "downrank", hits, mult
    if mult > 1.001:
        return "keep", hits, mult
    return "keep", hits, mult


def verify_and_prune(
    candidates: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach detailed unique reasons, apply explicit pass/fail causal rules,
    renormalize probability on kept set.

    Returns:
      {
        kept: [...],
        pruned: [{candidate_id, decision, rule_hits, reasons, ...}],
        notes: [str, ...],
      }
    """
    ctx = context or {}
    rules = _active_rules(ctx)
    annotated = attach_verification_reasons(candidates, ctx)
    notes: List[str] = [
        f"verify_and_prune n_in={len(annotated)}",
        "rules=" + ",".join(k for k, v in rules.items() if v),
    ]

    kept: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []

    for c in annotated:
        factors = c["verify_factors"]
        decision, hits, mult = _apply_causal_rules(c, factors, rules)
        item = dict(c)
        item["verify_decision"] = decision
        item["verify_rule_hits"] = hits
        item["verify_prob_multiplier"] = round(mult, 4)
        prior = float(item.get("probability") or item.get("precision_score") or 1e-6)
        item["probability_pre_verify"] = prior
        item["probability"] = max(prior * mult, 0.0)

        if decision == "prune":
            pruned.append(
                {
                    "id": item.get("id"),
                    "decision": "prune",
                    "rule_hits": hits,
                    "verify_reasons": item.get("verify_reasons"),
                    "probability_pre_verify": prior,
                }
            )
        else:
            kept.append(item)
            if decision == "downrank":
                notes.append(f"downrank id={item.get('id')} hits={hits}")

    if not kept and annotated:
        # fail-open: restore top by prior mass so pipeline never empties
        restore = sorted(
            annotated,
            key=lambda x: float(x.get("probability") or 0),
            reverse=True,
        )[: max(3, min(5, len(annotated)))]
        for c in restore:
            item = dict(c)
            item["verify_decision"] = "keep_failopen"
            item["verify_rule_hits"] = ["FAILOPEN:restored_after_empty_prune"]
            item["verify_prob_multiplier"] = 1.0
            kept.append(item)
        pruned = [p for p in pruned if p.get("id") not in {k.get("id") for k in kept}]
        notes.append("failopen: prune emptied pool; restored top prior mass")

    # Renormalize
    total = sum(float(c.get("probability") or 0) for c in kept) or 1.0
    for i, c in enumerate(kept, start=1):
        c["rank"] = i
        c["probability"] = round(float(c.get("probability") or 0) / total, 6)
        c["probability_pct"] = round(c["probability"] * 100.0, 3)

    kept.sort(key=lambda x: x.get("probability", 0), reverse=True)
    for i, c in enumerate(kept, start=1):
        c["rank"] = i

    notes.append(f"kept={len(kept)} pruned={len(pruned)}")
    return {
        "kept": kept,
        "pruned": pruned,
        "notes": notes,
        "rules_active": rules,
    }
