"""
Control-link / video communication range + optical horizon helpers.

Log-distance path loss (Sensors RSSI literature) + weather atten from env_factors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent


def _comms() -> Dict[str, Any]:
    return json.loads((ROOT / "data" / "engine_constants.json").read_text(encoding="utf-8"))[
        "comms"
    ]


def path_loss_db(distance_km: float, n: float, pl0_db: float, d0_m: float = 1.0) -> float:
    d_m = max(distance_km * 1000.0, d0_m)
    return pl0_db + 10.0 * n * math.log10(d_m / d0_m)


def max_range_km_from_budget(
    link_budget_db: float,
    n: float,
    pl0_db: float,
    weather_atten_db: float = 0.0,
) -> float:
    """Invert log-distance: PL + weather <= budget → max distance."""
    usable = link_budget_db - weather_atten_db
    if usable <= pl0_db:
        return 0.05
    # PL = pl0 + 10 n log10(d/d0) ≤ usable
    ratio = 10 ** ((usable - pl0_db) / (10.0 * n))
    return max(0.05, min(ratio / 1000.0, 25.0))


def assess_comms(
    range_hint_km: float,
    *,
    urban: bool = True,
    weather_atten_db: float = 0.0,
    freq_ghz: float = 2.4,
    visibility_km: Optional[float] = None,
    signal_dbm: Optional[float] = None,
) -> Dict[str, Any]:
    c = _comms()
    tx = float(c["tx_dbm_controller"])
    sens = float(c["rx_sensitivity_dbm"])
    g = float(c["antenna_gain_db"])
    fade = float(c["fade_margin_db"])
    n = float(c["urban_n"] if urban else c["open_n"])
    pl0 = float(c["pl0_db"])
    # Link budget for control (command) — slightly more robust than HD video
    budget_ctrl = tx + 2 * g - sens - fade
    budget_video = budget_ctrl - 8.0  # HD video needs higher SNR
    ctrl_model_km = max_range_km_from_budget(budget_ctrl, n, pl0, weather_atten_db)
    vid_model_km = max_range_km_from_budget(budget_video, n, pl0, weather_atten_db)

    # Empirical floor: field C2 often exceeds harsh urban log-distance;
    # RSSI (if present) expands annulus-based practical range.
    empirical = 1.8 if urban else 4.0
    if signal_dbm is not None:
        # rough: -50→~3km, -65→~1.5km, -80→~0.6km
        empirical = max(0.5, min(8.0, 10 ** ((-40 - float(signal_dbm)) / (10 * n)) / 1000.0))
    ctrl_km = max(ctrl_model_km, empirical * 0.85)
    vid_km = max(vid_model_km, empirical * 0.55)
    # weather shrinks practical
    if weather_atten_db >= 2:
        ctrl_km *= max(0.55, 1.0 - weather_atten_db / 40.0)
        vid_km *= max(0.45, 1.0 - weather_atten_db / 35.0)

    # Optical horizon rough (observer height 2m + drone AGL 50m)
    h1, h2 = 2.0, 50.0
    geo_horizon_km = 3.57 * (math.sqrt(h1) + math.sqrt(h2))
    opt_km = min(visibility_km or geo_horizon_km, geo_horizon_km)

    # Search span: prefer drone-type hint, soft-limited by control
    effective = min(max(range_hint_km, 0.8), max(ctrl_km * 1.25, range_hint_km * 0.75))
    pl_at_hint = path_loss_db(range_hint_km, n, pl0) + weather_atten_db

    return {
        "freq_ghz": freq_ghz,
        "path_loss_exponent_n": n,
        "link_budget_control_db": round(budget_ctrl, 2),
        "link_budget_video_db": round(budget_video, 2),
        "control_range_km": round(ctrl_km, 3),
        "control_range_model_km": round(ctrl_model_km, 3),
        "video_range_km": round(vid_km, 3),
        "optical_horizon_km": round(geo_horizon_km, 3),
        "optical_usable_km": round(opt_km, 3),
        "path_loss_at_hint_db": round(pl_at_hint, 2),
        "weather_atten_db": round(weather_atten_db, 3),
        "effective_search_km": round(effective, 3),
        "causal": {
            "control_ok_at_hint": pl_at_hint <= budget_ctrl or range_hint_km <= ctrl_km,
            "video_ok_at_hint": range_hint_km <= vid_km,
            "note": (
                "Operator must stay within control_range_km for C2 link; "
                "video drops earlier; optical limited by visibility × horizon."
            ),
        },
    }
