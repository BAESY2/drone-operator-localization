"""
Algorithmic agent conference — parallel votes, no LLM required.

Named agents vote on:
  - path_style
  - egress_mode_priority
  - prune_rules_active
  - recon_profile

Returns consensus + audit log. AI enrichment optional later.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

AgentFn = Callable[[Dict[str, Any]], Dict[str, Any]]

PATH_STYLES = ("direct_corridor", "urban_dodge_g123", "standoff_arc")
EGRESS_PRIORITIES = ("foot_first", "car_first", "mixed")
RECON_PROFILES = ("compressed_mass", "orbit_contacts", "verify_approach", "track_egress")


def _traffic_band(ctx: Dict[str, Any]) -> str:
    t = ctx.get("traffic") or {}
    if isinstance(t, dict):
        return str(t.get("band") or "unknown").lower()
    return "unknown"


def _wind_gust(ctx: Dict[str, Any]) -> float:
    env = ctx.get("environment") or ctx.get("weather") or {}
    return float(env.get("gust_mps") or env.get("wind_mps") or 0)


def _cctv_density(ctx: Dict[str, Any]) -> int:
    cctv = ctx.get("cctv") or {}
    return int(cctv.get("count") or len(cctv.get("cameras") or []))


def _mean_prob_mass(ctx: Dict[str, Any], n: int = 5) -> float:
    cands = ctx.get("candidates") or []
    if not cands:
        return 0.0
    top = sorted(
        cands, key=lambda c: float(c.get("probability") or 0), reverse=True
    )[:n]
    return sum(float(c.get("probability") or 0) for c in top)


def _agent_rf_geometry(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Favors tight AoA corridors and compressed recon when mass is peaked."""
    mass = _mean_prob_mass(ctx, 3)
    path = "direct_corridor" if mass >= 0.45 else "standoff_arc"
    recon = "compressed_mass" if mass >= 0.35 else "verify_approach"
    return {
        "agent": "rf_geometry",
        "votes": {
            "path_style": path,
            "egress_mode_priority": "mixed",
            "recon_profile": recon,
            "prune_rules_active": {
                "bearing_hard_fail": True,
                "rssi_hard_fail": True,
                "no_car_egress_dense_traffic_without_foot": True,
                "skyline_high_cctv_downrank": True,
                "forest_bunker_conceal_boost": True,
                "skyline_mid_elev_penalty": True,
                "extreme_gust_prune": False,
            },
        },
        "rationale": f"top3_mass={mass:.3f} → path={path} recon={recon}",
    }


def _agent_human_ops(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Concealment / mid-elev → dodge paths + foot egress + conceal boosts."""
    cands = ctx.get("candidates") or []
    conceal = 0.0
    n = 0
    for c in cands[:8]:
        parts = c.get("human_parts") or {}
        if parts.get("concealment") is not None:
            conceal += float(parts["concealment"])
            n += 1
    avg = conceal / n if n else 0.5
    return {
        "agent": "human_ops",
        "votes": {
            "path_style": "urban_dodge_g123" if avg >= 0.55 else "standoff_arc",
            "egress_mode_priority": "foot_first",
            "recon_profile": "compressed_mass",
            "prune_rules_active": {
                "bearing_hard_fail": True,
                "rssi_hard_fail": False,
                "no_car_egress_dense_traffic_without_foot": True,
                "skyline_high_cctv_downrank": True,
                "forest_bunker_conceal_boost": True,
                "skyline_mid_elev_penalty": True,
                "extreme_gust_prune": False,
            },
        },
        "rationale": f"avg_concealment={avg:.2f} → foot_first + dodge",
    }


def _agent_traffic_osint(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Dense traffic / CCTV → foot egress, strict prune, compressed recon."""
    band = _traffic_band(ctx)
    cams = _cctv_density(ctx)
    dense = band in ("heavy", "dense", "high") or cams >= 5
    return {
        "agent": "traffic_osint",
        "votes": {
            "path_style": "urban_dodge_g123" if dense else "direct_corridor",
            "egress_mode_priority": "foot_first" if dense else "car_first",
            "recon_profile": "compressed_mass" if dense else "orbit_contacts",
            "prune_rules_active": {
                "bearing_hard_fail": True,
                "rssi_hard_fail": True,
                "no_car_egress_dense_traffic_without_foot": True,
                "skyline_high_cctv_downrank": cams >= 2,
                "forest_bunker_conceal_boost": True,
                "skyline_mid_elev_penalty": True,
                "extreme_gust_prune": False,
            },
        },
        "rationale": f"traffic={band} cctv={cams} dense={dense}",
    }


def _agent_weather_flight(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Wind/vis → shorter recon profile + optional gust prune."""
    gust = _wind_gust(ctx)
    env = ctx.get("environment") or ctx.get("weather") or {}
    vis = env.get("visibility_km")
    if vis is None and env.get("visibility_m") is not None:
        vis = float(env["visibility_m"]) / 1000.0
    harsh = gust >= 10 or (vis is not None and float(vis) < 3.0)
    return {
        "agent": "weather_flight",
        "votes": {
            "path_style": "standoff_arc" if harsh else "direct_corridor",
            "egress_mode_priority": "mixed",
            "recon_profile": "compressed_mass" if harsh else "verify_approach",
            "prune_rules_active": {
                "bearing_hard_fail": True,
                "rssi_hard_fail": True,
                "no_car_egress_dense_traffic_without_foot": True,
                "skyline_high_cctv_downrank": True,
                "forest_bunker_conceal_boost": True,
                "skyline_mid_elev_penalty": True,
                "extreme_gust_prune": gust >= 14,
            },
        },
        "rationale": f"gust={gust:.1f} vis_km={vis} harsh={harsh}",
    }


def _agent_isr_economy(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Minimize flight distance: always compress recon; mixed egress."""
    n = len(ctx.get("candidates") or [])
    return {
        "agent": "isr_economy",
        "votes": {
            "path_style": "direct_corridor" if n <= 6 else "urban_dodge_g123",
            "egress_mode_priority": "mixed",
            "recon_profile": "compressed_mass",
            "prune_rules_active": {
                "bearing_hard_fail": True,
                "rssi_hard_fail": True,
                "no_car_egress_dense_traffic_without_foot": True,
                "skyline_high_cctv_downrank": True,
                "forest_bunker_conceal_boost": False,
                "skyline_mid_elev_penalty": True,
                "extreme_gust_prune": False,
            },
        },
        "rationale": f"n_candidates={n} → compress always (ISR loss reduction)",
    }


CONFERENCE_AGENTS: Dict[str, AgentFn] = {
    "rf_geometry": _agent_rf_geometry,
    "human_ops": _agent_human_ops,
    "traffic_osint": _agent_traffic_osint,
    "weather_flight": _agent_weather_flight,
    "isr_economy": _agent_isr_economy,
}


def _majority_str(votes: List[str], options: Tuple[str, ...]) -> Tuple[str, Dict[str, int]]:
    counts = Counter(votes)
    # stable: prefer first option on ties by max count then option order
    best = max(options, key=lambda o: (counts.get(o, 0), -options.index(o)))
    return best, {o: counts.get(o, 0) for o in options}


def _majority_rules(rule_votes: List[Dict[str, bool]]) -> Dict[str, bool]:
    keys = set()
    for rv in rule_votes:
        keys.update(rv.keys())
    out: Dict[str, bool] = {}
    for k in sorted(keys):
        trues = sum(1 for rv in rule_votes if rv.get(k))
        out[k] = trues > len(rule_votes) / 2.0
    return out


def run_agent_conference(
    context: Optional[Dict[str, Any]] = None,
    *,
    agents: Optional[Dict[str, AgentFn]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run named algorithmic agents in parallel; majority-vote conference fields.

    context may include: candidates, traffic, cctv, environment/weather, observer.

    Returns:
      {
        consensus: {path_style, egress_mode_priority, prune_rules_active, recon_profile},
        audit_log: [...],
        ballots: [...],
        agent_names: [...],
      }
    """
    ctx = context or {}
    roster = agents or CONFERENCE_AGENTS
    workers = max_workers or min(8, max(1, len(roster)))
    ballots: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, ctx): name for name, fn in roster.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                ballot = fut.result()
                ballots.append(ballot)
                audit.append(
                    {
                        "agent": name,
                        "status": "ok",
                        "votes": ballot.get("votes"),
                        "rationale": ballot.get("rationale"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                audit.append(
                    {
                        "agent": name,
                        "status": "error",
                        "error": str(exc)[:160],
                    }
                )

    # Sort audit by agent name for stable logs
    audit.sort(key=lambda a: str(a.get("agent") or ""))
    ballots.sort(key=lambda b: str(b.get("agent") or ""))

    if not ballots:
        consensus = {
            "path_style": "direct_corridor",
            "egress_mode_priority": "mixed",
            "recon_profile": "compressed_mass",
            "prune_rules_active": {
                "bearing_hard_fail": True,
                "rssi_hard_fail": True,
                "no_car_egress_dense_traffic_without_foot": True,
                "skyline_high_cctv_downrank": True,
                "forest_bunker_conceal_boost": True,
                "skyline_mid_elev_penalty": True,
                "extreme_gust_prune": False,
            },
        }
        audit.append({"agent": "system", "status": "fallback", "reason": "no_ballots"})
        return {
            "consensus": consensus,
            "audit_log": audit,
            "ballots": [],
            "agent_names": list(roster.keys()),
            "llm_used": False,
        }

    path_style, path_tally = _majority_str(
        [b["votes"]["path_style"] for b in ballots], PATH_STYLES
    )
    egress, egress_tally = _majority_str(
        [b["votes"]["egress_mode_priority"] for b in ballots], EGRESS_PRIORITIES
    )
    recon, recon_tally = _majority_str(
        [b["votes"]["recon_profile"] for b in ballots], RECON_PROFILES
    )
    rules = _majority_rules(
        [b["votes"].get("prune_rules_active") or {} for b in ballots]
    )

    consensus = {
        "path_style": path_style,
        "egress_mode_priority": egress,
        "recon_profile": recon,
        "prune_rules_active": rules,
    }
    audit.append(
        {
            "agent": "conference_chair",
            "status": "consensus",
            "tallies": {
                "path_style": path_tally,
                "egress_mode_priority": egress_tally,
                "recon_profile": recon_tally,
            },
            "consensus": consensus,
        }
    )

    return {
        "consensus": consensus,
        "audit_log": audit,
        "ballots": ballots,
        "agent_names": list(roster.keys()),
        "llm_used": False,
        "note": "Algorithmic conference only — AI optional later",
    }
