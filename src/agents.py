"""병렬 경쟁 에이전트 — 서로 다른 가설로 후보를 채점한다."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

from src.rf_models import bearing_delta, bearing_likelihood, constants, rssi_distance_likelihood
from src.utils import distance_between_coords, initial_bearing

Coord = Tuple[float, float]
AgentFn = Callable[[List[Dict[str, Any]], Dict[str, Any]], List[Tuple[str, float]]]


def _aoa_agent(pool: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Tuple[str, float]]:
    """DRONET AoA 가우시안 — 방위각 일치 최우선."""
    obs = ctx["observer"]
    bearing = ctx["bearing_deg"]
    sigma = constants()["aoa_sigma_deg_single"]
    out = []
    for c in pool:
        brg = initial_bearing(obs, (c["lat"], c["lng"]))
        L = bearing_likelihood(bearing_delta(bearing, brg), sigma)
        out.append((c["id"], L))
    return out


def _rssi_agent(pool: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Sensors 2023 RSSI 경로손실 환대."""
    obs = ctx["observer"]
    rssi = ctx.get("rssi_dbm")
    out = []
    for c in pool:
        d = distance_between_coords(obs, (c["lat"], c["lng"]))
        L = rssi_distance_likelihood(d, rssi) if rssi is not None else 0.5
        out.append((c["id"], L))
    return out


def _elevation_los_agent(
    pool: List[Dict[str, Any]], ctx: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """
    Mid-elevation + usable LOS — NOT 'highest roof wins'.
    Trained OPs avoid skyline landmarks.
    """
    from src.human_ops import los_usable_score, mid_elevation_score

    out = []
    for c in pool:
        mid = mid_elevation_score(
            c.get("height_m"),
            c.get("ground_elevation_m") or (c.get("elevation") or {}).get("ground_m"),
            pool,
        )
        los = los_usable_score(c)
        out.append((c["id"], 0.55 * mid + 0.45 * los))
    return out


def _human_behavior_agent(
    pool: List[Dict[str, Any]], ctx: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """Full human-operator causal prior."""
    from src.human_ops import score_human_operator

    obs = ctx["observer"]
    bearing = ctx["bearing_deg"]
    range_km = float(ctx.get("range_km") or 5.0)
    out = []
    for c in pool:
        hb = score_human_operator(c, obs, bearing, range_km, pool)
        out.append((c["id"], hb["human_score"]))
    return out


def _urban_access_agent(
    pool: List[Dict[str, Any]], ctx: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """주거/도로/통신탑 — 도심 조종 거점 가설."""
    out = []
    for c in pool:
        attrs = c.get("attributes") or {}
        bt = attrs.get("building_type") or c.get("building_type") or "unknown"
        type_s = {"residential": 1.0, "commercial": 0.65, "industrial": 0.45}.get(
            bt, 0.3
        )
        road = attrs.get("dist_to_road_m")
        road_s = 1.0
        if road is not None:
            road_s = max(0.0, 1.0 - float(road) / 1200.0)
        tower = 1.0 if attrs.get("cell_tower_nearby") else 0.35
        mil = 1.0 if attrs.get("military") else 0.0
        out.append((c["id"], 0.4 * type_s + 0.3 * road_s + 0.2 * tower + 0.1 * mil))
    return out


def _triangulation_agent(
    pool: List[Dict[str, Any]], ctx: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """삼각측량 fix 근접도 (관측자 2+일 때만 강함)."""
    fix = ctx.get("fix")
    out = []
    for c in pool:
        if not fix:
            out.append((c["id"], 0.4))
            continue
        d_m = distance_between_coords(fix, (c["lat"], c["lng"])) * 1000.0
        out.append((c["id"], math.exp(-0.5 * (d_m / 120.0) ** 2)))
    return out


def _kinematics_agent(
    pool: List[Dict[str, Any]], ctx: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """BGU 궤적 prior — flight_track이 있으면 가중, 없으면 중립."""
    from src.kinematics import kinematics_candidate_weight

    features = ctx.get("kinematics_features")
    out = []
    if not features or not features.get("usable"):
        return [(c["id"], 0.45) for c in pool]
    for c in pool:
        out.append((c["id"], kinematics_candidate_weight(c, features, ctx.get("observer"))))
    return out


AGENTS: Dict[str, AgentFn] = {
    "aoa_gaussian": _aoa_agent,
    "rssi_annulus": _rssi_agent,
    "elevation_los": _elevation_los_agent,
    "human_behavior": _human_behavior_agent,
    "urban_access": _urban_access_agent,
    "triangulation_snap": _triangulation_agent,
    "kinematics_path": _kinematics_agent,
}


def run_competing_agents(
    pool: List[Dict[str, Any]],
    observer: Coord,
    bearing_deg: float,
    rssi_dbm: float | None,
    fix: Coord | None = None,
    kinematics_features: Dict[str, Any] | None = None,
    range_km: float | None = None,
) -> Dict[str, Any]:
    """에이전트 병렬 실행 → 후보별 투표·합의."""
    ctx = {
        "observer": observer,
        "bearing_deg": bearing_deg,
        "rssi_dbm": rssi_dbm,
        "fix": fix,
        "kinematics_features": kinematics_features,
        "range_km": range_km or 5.0,
    }
    votes: Dict[str, Dict[str, float]] = {c["id"]: {} for c in pool}
    rankings: Dict[str, List[str]] = {}

    with ThreadPoolExecutor(max_workers=len(AGENTS)) as ex:
        futures = {ex.submit(fn, pool, ctx): name for name, fn in AGENTS.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            scored = fut.result()
            scored.sort(key=lambda x: x[1], reverse=True)
            rankings[name] = [cid for cid, _ in scored]
            # softmax-ish normalize
            vals = [max(v, 1e-12) for _, v in scored]
            s = sum(vals) or 1.0
            for (cid, v), nv in zip(scored, vals):
                votes[cid][name] = round(nv / s, 5)

    # Borda + mass agreement
    borda: Dict[str, float] = {c["id"]: 0.0 for c in pool}
    n = len(pool)
    for ranking in rankings.values():
        for rank, cid in enumerate(ranking):
            borda[cid] += (n - rank) / max(n, 1)

    # agreement = how many agents put candidate in their top 30%
    top_k = max(1, int(math.ceil(n * 0.3)))
    agreement: Dict[str, int] = {c["id"]: 0 for c in pool}
    for ranking in rankings.values():
        for cid in ranking[:top_k]:
            agreement[cid] += 1

    temp = constants()["ensemble_temperature"]
    fused = []
    for c in pool:
        cid = c["id"]
        agent_mass = sum(votes[cid].values()) / max(len(AGENTS), 1)
        borda_n = borda[cid] / max(len(AGENTS) * 1.0, 1.0)
        agree_n = agreement[cid] / max(len(AGENTS), 1)
        # blend with prior precision probability
        prior = float(c.get("probability_raw") or c.get("probability") or 0)
        energy = (
            0.35 * agent_mass
            + 0.25 * (borda_n / max(n, 1))
            + 0.25 * agree_n
            + 0.15 * prior
        )
        fused.append((cid, energy, agree_n, votes[cid]))

    # temperature softmax
    max_e = max(e for _, e, _, _ in fused) if fused else 0
    weights = [math.exp((e - max_e) / max(temp, 0.05)) for _, e, _, _ in fused]
    z = sum(weights) or 1.0
    by_id = {cid: (w / z, agree, v) for (cid, _, agree, v), w in zip(fused, weights)}

    enriched = []
    for c in pool:
        cid = c["id"]
        prob, agree, agent_votes = by_id[cid]
        item = dict(c)
        item["probability"] = round(prob, 5)
        item["agent_votes"] = agent_votes
        item["agent_agreement"] = round(agree, 3)
        item["ensemble_score"] = round(prob, 5)
        enriched.append(item)
    enriched.sort(key=lambda x: x["probability"], reverse=True)

    winner = max(rankings.items(), key=lambda kv: _top1_mass(kv[1], by_id))[0] if rankings else None
    return {
        "pool": enriched,
        "rankings": rankings,
        "winning_agent": winner,
        "agent_names": list(AGENTS.keys()),
    }


def _top1_mass(ranking: List[str], by_id: Dict) -> float:
    if not ranking:
        return 0.0
    return float(by_id.get(ranking[0], (0.0, 0, {}))[0])
