"""Build formula-scored wide candidate pools; optional AI shortlist (never invents coords)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.ai_assist import chat_json


def sanitize_elevation_m(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < -50 or v > 9000:
        return None
    return round(v, 2)


def clean_candidate_elevations(c: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(c)
    elev = dict(item.get("elevation") or {})
    ground = sanitize_elevation_m(
        elev.get("ground_m")
        if elev.get("ground_m") is not None
        else item.get("ground_elevation_m")
    )
    roof = sanitize_elevation_m(
        elev.get("roof_m") if elev.get("roof_m") is not None else item.get("elevation_m")
    )
    height = sanitize_elevation_m(
        elev.get("height_m") if elev.get("height_m") is not None else item.get("height_m")
    )
    if ground is None and roof is not None and height is not None and roof > height:
        ground = round(roof - height, 2)
    if roof is None and ground is not None and height is not None:
        roof = round(ground + height, 2)
    elev["ground_m"] = ground
    elev["roof_m"] = roof
    elev["height_m"] = height
    item["elevation"] = elev
    item["ground_elevation_m"] = ground
    item["elevation_m"] = roof
    item["height_m"] = height
    return item


def ai_narrow_candidates(
    candidates: List[Dict[str, Any]],
    *,
    keep: int = 8,
    lang: str = "en",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rank/filter an ALREADY scored formula pool. AI may only pick ids from the list.
    """
    if not candidates:
        return {"applied": False, "reason": "empty", "candidates": [], "dropped_ids": []}

    # Always clean elevations first
    pool = [clean_candidate_elevations(c) for c in candidates]

    if not api_key:
        # algorithmic shortlist: keep top by probability / precision
        short = pool[:keep]
        return {
            "applied": False,
            "reason": "no_ai_key_algorithm_top",
            "candidates": short,
            "dropped_ids": [c.get("id") for c in pool[keep:]],
            "method": "formula_topk",
        }

    summary = []
    for c in pool[:40]:
        summary.append(
            {
                "id": c.get("id"),
                "rank": c.get("rank"),
                "p": c.get("probability"),
                "brg_err": c.get("bearing_error_deg"),
                "dist_m": c.get("distance_m") or (c.get("location") or {}).get("distance_m"),
                "r_m": c.get("confidence_radius_m"),
                "elev_g": (c.get("elevation") or {}).get("ground_m"),
                "h_m": c.get("height_m"),
                "type": c.get("building_type"),
                "reasons": (c.get("reasons") or [])[:4],
                "agree": c.get("agent_agreement"),
            }
        )

    parsed = chat_json(
        system=(
            "You filter DEFENSIVE localization candidates. "
            "You MUST only select ids from the provided list. Never invent coordinates. "
            "Prefer low bearing_error, tight radius, coherent RSSI distance, LOS, rooftop height. "
            "Drop duplicates that are spatially redundant. "
            f"Return JSON: keep_ids (array, max {keep}), drop_ids (array), rationale (string). "
            f"Language for rationale: {lang}."
        ),
        user={"candidates": summary, "keep_max": keep},
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    if not parsed or not isinstance(parsed.get("keep_ids"), list):
        return {
            "applied": False,
            "reason": "ai_failed_fallback_topk",
            "candidates": pool[:keep],
            "dropped_ids": [c.get("id") for c in pool[keep:]],
            "method": "formula_topk",
        }

    id_set = {c.get("id") for c in pool}
    keep_ids = [i for i in parsed["keep_ids"] if i in id_set][:keep]
    if not keep_ids:
        keep_ids = [c.get("id") for c in pool[:keep]]

    by_id = {c.get("id"): c for c in pool}
    short = [by_id[i] for i in keep_ids if i in by_id]
    # fill if AI returned too few
    for c in pool:
        if len(short) >= keep:
            break
        if c.get("id") not in {x.get("id") for x in short}:
            short.append(c)

    total = sum(float(c.get("probability") or 0) for c in short) or 1.0
    for i, c in enumerate(short, start=1):
        c["rank"] = i
        c["probability"] = round(float(c.get("probability") or 0) / total, 6)
        c["probability_pct"] = round(c["probability"] * 100.0, 3)
        c["ai_kept"] = True

    return {
        "applied": True,
        "reason": "ai_narrow",
        "method": "formula_pool+ai_filter",
        "rationale": str(parsed.get("rationale") or "")[:600],
        "candidates": short,
        "dropped_ids": [c.get("id") for c in pool if c.get("id") not in keep_ids],
    }
