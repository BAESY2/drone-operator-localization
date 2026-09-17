"""Validate DEFENSIVE localization/recon against public approach scenarios (no strike)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analyze_pipeline import run_defensive_analysis
from src.recon import build_recon_plan


def main() -> None:
    path = ROOT / "data" / "scenarios" / "defensive_public_reports.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for sc in data["scenarios"]:
        obs = sc["observer"]
        analysis = run_defensive_analysis(
            obs["lat"],
            obs["lng"],
            obs["bearing_deg"],
            use_ai=False,
        )
        ok = bool(analysis.get("success"))
        egress = (analysis.get("egress") or {}).get("egress_axis_deg")
        expect = sc.get("expect") or {}
        if "egress_axis_near_deg" in expect and egress is not None:
            diff = abs((egress - expect["egress_axis_near_deg"] + 180) % 360 - 180)
            ok = ok and diff <= 25
        plan = build_recon_plan(
            (obs["lat"], obs["lng"]),
            obs["bearing_deg"],
            profile=(analysis.get("recon_recommendation") or {}).get("profile")
            or "verify_approach",
            candidates=analysis.get("narrowed_candidates") or [],
        )
        results.append(
            {
                "id": sc["id"],
                "ok": ok,
                "contacts": len(analysis.get("narrowed_candidates") or []),
                "recon_mode": (analysis.get("recon_recommendation") or {}).get("mode"),
                "civ_high": (analysis.get("civilian_risk") or {}).get("high_risk_count"),
                "egress_axis_deg": egress,
                "recon_wp": len(plan.get("waypoints") or []),
                "policy": analysis.get("policy"),
            }
        )
        print(
            f"{sc['id']}: ok={ok} mode={results[-1]['recon_mode']} "
            f"contacts={results[-1]['contacts']} egress={egress}"
        )

    out = ROOT / "data" / "test_results" / "defensive_scenario_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
