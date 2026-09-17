"""재검토(review) — 집계·경쟁 이후 후보지군을 최종 축소."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.rf_models import constants
from src.utils import distance_between_coords

Coord = Tuple[float, float]


def _cluster_nearby(pool: List[Dict[str, Any]], radius_m: float = 45.0) -> List[Dict[str, Any]]:
    """근접 후보는 최고 확률 하나만 남긴다 (중복 건물 footprint)."""
    kept: List[Dict[str, Any]] = []
    for c in sorted(pool, key=lambda x: x.get("probability", 0), reverse=True):
        too_close = False
        for k in kept:
            d = distance_between_coords(
                (c["lat"], c["lng"]), (k["lat"], k["lng"])
            ) * 1000.0
            if d < radius_m:
                too_close = True
                # merge reasons
                k["reasons"] = list(dict.fromkeys(k.get("reasons", []) + c.get("reasons", [])))
                k["cluster_size"] = k.get("cluster_size", 1) + 1
                break
        if not too_close:
            item = dict(c)
            item["cluster_size"] = 1
            kept.append(item)
    return kept


def rereview_candidates(
    pool: List[Dict[str, Any]],
    observer: Coord,
    max_keep: int = 8,
    min_agreement: float | None = None,
) -> Dict[str, Any]:
    """
    집계 완료 후 재검토:
    1) 에이전트 합의 미달 제거
    2) 확률 질량 하위 꼬리 제거
    3) 공간 클러스터링
    4) 상위 max_keep + 확률 재정규화
    """
    if not pool:
        return {
            "before_count": 0,
            "after_count": 0,
            "rejected": [],
            "final": [],
            "review_notes": ["empty pool"],
        }

    cnst = constants()
    min_agree = (
        min_agreement
        if min_agreement is not None
        else (cnst.get("min_agreement_agents", 2) / max(len(pool[0].get("agent_votes") or {1: 1}), 1))
    )
    # if agreement stored as fraction 0-1
    if pool and pool[0].get("agent_agreement") is not None:
        min_agree_frac = cnst.get("min_agreement_agents", 2) / 5.0  # 5 agents
    else:
        min_agree_frac = 0.0

    rejected: List[Dict[str, Any]] = []
    survivors: List[Dict[str, Any]] = []
    notes: List[str] = []

    # Step A: agreement gate
    for c in pool:
        agree = float(c.get("agent_agreement") or 0)
        if agree + 1e-9 < min_agree_frac and len(pool) > max_keep:
            rejected.append(
                {
                    "id": c["id"],
                    "reason": f"agent_agreement {agree:.2f} < {min_agree_frac:.2f}",
                }
            )
        else:
            survivors.append(c)
    if not survivors:
        survivors = pool[: max(3, max_keep)]
        notes.append("agreement gate emptied pool; restored top prior")
    else:
        notes.append(f"agreement gate kept {len(survivors)}/{len(pool)}")

    # Step B: probability mass — keep wide (don't collapse early)
    survivors = sorted(survivors, key=lambda x: x.get("probability", 0), reverse=True)
    cum = 0.0
    mass_kept: List[Dict[str, Any]] = []
    min_mass = min(max_keep, max(16, max_keep // 2))
    for c in survivors:
        mass_kept.append(c)
        cum += float(c.get("probability") or 0)
        if len(mass_kept) >= max_keep * 2:
            break
        if cum >= 0.92 and len(mass_kept) >= min_mass:
            break
    if len(mass_kept) < min_mass:
        mass_kept = survivors[: min(min_mass, len(survivors))]
        cum = sum(float(c.get("probability") or 0) for c in mass_kept)
    for c in survivors[len(mass_kept) :]:
        rejected.append({"id": c["id"], "reason": "low_probability_tail"})
    notes.append(f"probability mass cut → {len(mass_kept)} (cum≈{cum:.2f})")

    # Step C: spatial cluster (tighter merge radius → more survivors)
    clustered = _cluster_nearby(mass_kept, radius_m=32.0)
    notes.append(f"spatial cluster {len(mass_kept)} → {len(clustered)}")

    # Step D: hard AoA / precision veto on bottom
    if clustered:
        best_p = float(clustered[0].get("precision_score") or clustered[0].get("probability") or 1)
        refined = []
        for c in clustered:
            p = float(c.get("precision_score") or 0)
            be = (
                c.get("scores", {})
                .get("components", {})
                .get("p_bearing_error_deg")
            )
            if be is not None and be > 12 and len(clustered) > 2:
                rejected.append({"id": c["id"], "reason": f"bearing_error {be}°"})
                continue
            if p < best_p * 0.22 and len(clustered) > max_keep:
                rejected.append({"id": c["id"], "reason": "precision_far_below_peak"})
                continue
            refined.append(c)
        clustered = refined or clustered[:max_keep]

    final = clustered[:max_keep]
    # renormalize probabilities
    total = sum(float(c.get("probability") or 0) for c in final) or 1.0
    for i, c in enumerate(final, start=1):
        c["rank"] = i
        c["probability"] = round(float(c.get("probability") or 0) / total, 5)
        c["probability_pct"] = round(c["probability"] * 100.0, 2)
        # human summary line
        loc = c.get("location") or {}
        elev = c.get("elevation") or {}
        c["summary"] = (
            f"#{i} {c['probability_pct']}% | "
            f"{loc.get('latitude')}, {loc.get('longitude')} | "
            f"고도 {elev.get('roof_m') or elev.get('ground_m')}m | "
            f"{'; '.join((c.get('reasons') or [])[:3])}"
        )

    return {
        "before_count": len(pool),
        "after_count": len(final),
        "rejected_count": len(rejected),
        "rejected": rejected[:40],
        "final": final,
        "review_notes": notes,
    }
