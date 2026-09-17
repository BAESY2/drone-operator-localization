"""
Environment factors — weather, optical visibility, RF weather attenuation,
wind direction / day-night for defensive approach & egress analysis.

Data: Open-Meteo (no key). Formulas: ITU-R P.838-style rain atten approx,
Kruse fog visibility, causal link to RF path loss (not AI).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from src.providers.http import get_json

ROOT = Path(__file__).resolve().parent.parent
Coord = Tuple[float, float]


def _engine() -> Dict[str, Any]:
    path = ROOT / "data" / "engine_constants.json"
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_weather(lat: float, lng: float) -> Dict[str, Any]:
    """Live weather snapshot from Open-Meteo (incl. wind_direction_10m)."""
    try:
        data = get_json(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": round(lat, 4),
                "longitude": round(lng, 4),
                "current": (
                    "temperature_2m,relative_humidity_2m,precipitation,"
                    "weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,"
                    "wind_gusts_10m,visibility"
                ),
                "timezone": "auto",
            },
            cache_key=f"wx:{round(lat, 3)}:{round(lng, 3)}",
            ttl_sec=15 * 60,
            timeout=12,
        )
        cur = data.get("current") or {}
        vis_m = cur.get("visibility")
        wind_dir = cur.get("wind_direction_10m")
        tz_name = data.get("timezone") or "UTC"
        return {
            "ok": True,
            "source": "open-meteo",
            "timezone": tz_name,
            "utc_offset_seconds": data.get("utc_offset_seconds"),
            "observation_time": cur.get("time"),
            "temperature_c": cur.get("temperature_2m"),
            "humidity_pct": cur.get("relative_humidity_2m"),
            "precip_mm": float(cur.get("precipitation") or 0),
            "weather_code": cur.get("weather_code"),
            "cloud_cover_pct": cur.get("cloud_cover"),
            "wind_mps": float(cur.get("wind_speed_10m") or 0) / 3.6
            if cur.get("wind_speed_10m") is not None
            else 0.0,
            "wind_direction_deg": float(wind_dir) if wind_dir is not None else None,
            "gust_mps": float(cur.get("wind_gusts_10m") or 0) / 3.6
            if cur.get("wind_gusts_10m") is not None
            else 0.0,
            "visibility_m": float(vis_m) if vis_m is not None else None,
            "visibility_km": round(float(vis_m) / 1000.0, 3) if vis_m else None,
        }
    except Exception as exc:  # noqa: BLE001
        eng = _engine()
        return {
            "ok": False,
            "source": "fallback_defaults",
            "error": str(exc)[:120],
            "timezone": "UTC",
            "utc_offset_seconds": 0,
            "observation_time": None,
            "temperature_c": None,
            "humidity_pct": None,
            "precip_mm": 0.0,
            "weather_code": None,
            "cloud_cover_pct": 40,
            "wind_mps": 3.0,
            "wind_direction_deg": None,
            "gust_mps": 5.0,
            "visibility_m": eng["visibility"]["optical_default_km"] * 1000,
            "visibility_km": eng["visibility"]["optical_default_km"],
        }


def local_day_night(
    weather: Dict[str, Any],
    lat: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Local hour and day/night factor from Open-Meteo timezone (auto).
    day_night_factor: 1.0 = full day EO, ~0.25 = deep night (RF/thermal preferred).
    """
    tz_name = str(weather.get("timezone") or "UTC")
    obs = weather.get("observation_time")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
        tz_name = "UTC"

    if obs:
        try:
            # Open-Meteo current.time is local wall time without offset when timezone=auto
            local_dt = datetime.fromisoformat(str(obs)).replace(tzinfo=tz)
        except ValueError:
            local_dt = datetime.now(tz)
    else:
        local_dt = datetime.now(tz)

    hour = local_dt.hour + local_dt.minute / 60.0
    # Simple civil day window; refine with latitude season if provided
    sunrise = 6.0
    sunset = 18.5
    if lat is not None:
        # crude seasonal stretch: higher |lat| → longer summer day (month from local)
        month = local_dt.month
        # northern-hemisphere bias; flip for south
        season = math.cos(2 * math.pi * (month - 6) / 12.0)
        if lat < 0:
            season = -season
        stretch = 1.2 * abs(float(lat)) / 90.0 * season
        sunrise = max(4.0, 6.0 - stretch)
        sunset = min(21.0, 18.5 + stretch)

    if sunrise <= hour <= sunset:
        # peak noon ≈ 1.0, edges taper
        mid = 0.5 * (sunrise + sunset)
        half = max(0.5, 0.5 * (sunset - sunrise))
        day_night_factor = max(0.55, 1.0 - 0.45 * abs(hour - mid) / half)
        phase = "day"
    elif sunrise - 1.5 <= hour < sunrise:
        day_night_factor = 0.25 + 0.35 * (hour - (sunrise - 1.5)) / 1.5
        phase = "dawn"
    elif sunset < hour <= sunset + 1.5:
        day_night_factor = 0.6 - 0.35 * (hour - sunset) / 1.5
        phase = "dusk"
    else:
        day_night_factor = 0.25
        phase = "night"

    day_night_factor = max(0.2, min(1.0, day_night_factor))
    return {
        "timezone": tz_name,
        "local_iso": local_dt.isoformat(),
        "local_hour": round(hour, 2),
        "phase": phase,
        "day_night_factor": round(day_night_factor, 3),
        "sunrise_hour_approx": round(sunrise, 2),
        "sunset_hour_approx": round(sunset, 2),
        "note": (
            "Day favors EO/visual confirm; night lowers optical factor — "
            "prefer RF/thermal/ISR for defensive review."
        ),
    }


def wind_causal_notes(weather: Dict[str, Any]) -> Dict[str, Any]:
    """
    Defensive-only causal links: wind → drone approach bias & egress smoke/dust.
    """
    wind_mps = float(weather.get("wind_mps") or 0)
    gust = float(weather.get("gust_mps") or wind_mps)
    wdir = weather.get("wind_direction_deg")
    notes = []
    approach_bias_deg = None
    smoke_dust_factor = 0.0

    if wdir is not None:
        # Meteorological wind FROM direction; small UAS often approach with
        # headwind component for energy/control — defensive hypothesis only.
        approach_bias_deg = round((float(wdir) + 180.0) % 360.0, 1)
        notes.append(
            f"Wind from {float(wdir):.0f}° → weak approach-bias hypothesis "
            f"into wind (~{approach_bias_deg}°) for energy/control (defensive, not targeting)."
        )
    else:
        notes.append("Wind direction unavailable — no approach-bias vector.")

    if wind_mps >= 4.0 or gust >= 8.0:
        # Higher wind lifts dust/smoke downwind of roads/fields during egress
        smoke_dust_factor = min(
            1.0,
            0.15 * max(0.0, wind_mps - 3.0) + 0.08 * max(0.0, gust - 6.0),
        )
        downwind = round(float(wdir), 1) if wdir is not None else None
        notes.append(
            "Egress smoke/dust plume drifts downwind"
            + (f" (~toward {downwind}°)" if downwind is not None else "")
            + " — obscures EO on downwind sector, may reveal movement upwind of observers."
        )
    else:
        notes.append("Light wind — limited smoke/dust egress signature.")

    if gust >= 12:
        notes.append("High gusts — small-UAS control margin reduced; RF link jitter risk.")

    return {
        "wind_direction_deg": float(wdir) if wdir is not None else None,
        "wind_mps": round(wind_mps, 2),
        "gust_mps": round(gust, 2),
        "approach_bias_deg": approach_bias_deg,
        "smoke_dust_factor": round(smoke_dust_factor, 3),
        "downwind_deg": float(wdir) if wdir is not None else None,
        "notes": notes,
        "policy": "Defensive analysis only — approach bias / plume awareness, not strike tasking.",
    }


def optical_visibility_km(weather: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer station visibility; else humidity/precip heuristic (Kruse-inspired).
    """
    eng = _engine()["visibility"]
    if weather.get("visibility_km") is not None:
        v = float(weather["visibility_km"])
        method = "station"
    else:
        # crude: heavy precip / high humidity → reduced vis
        precip = float(weather.get("precip_mm") or 0)
        hum = float(weather.get("humidity_pct") or 60)
        v = eng["optical_default_km"]
        if precip >= 2:
            v *= 0.35
        elif precip >= 0.5:
            v *= 0.55
        if hum >= 92:
            v *= 0.5
        elif hum >= 85:
            v *= 0.75
        method = "heuristic_kruse_proxy"
    v = max(0.05, min(v, 50.0))
    ops_ok = v >= float(eng["min_ops_vis_km"])
    return {
        "visibility_km": round(v, 3),
        "method": method,
        "ops_visual_ok": ops_ok,
        "note": (
            "Visual ID feasible"
            if ops_ok
            else "Low visibility — prefer RF/thermal/ISR over EO"
        ),
    }


def rf_weather_attenuation_db(
    range_km: float,
    weather: Dict[str, Any],
    freq_ghz: float = 2.4,
) -> Dict[str, Any]:
    """
    Extra path loss from rain/fog/gusts (dB).
    Rain rates mapped from precip_mm (approx instantaneous).
    """
    eng = _engine()["weather_rf"]
    precip = float(weather.get("precip_mm") or 0)
    # treat precip_mm as rough rain-rate proxy mm/h for short horizon
    rain_rate = precip * 6.0  # amplify snapshot → effective rate proxy
    key = (
        "rain_atten_db_per_km_5g8"
        if freq_ghz >= 5.0
        else "rain_atten_db_per_km_2g4"
    )
    rain_db = float(eng[key]) * rain_rate * max(range_km, 0.1)
    vis_km = float(
        (optical_visibility_km(weather).get("visibility_km") or 10)
    )
    fog_db = float(eng["fog_extra_db_vis_lt_1km"]) if vis_km < 1.0 else 0.0
    gust = float(weather.get("gust_mps") or weather.get("wind_mps") or 0)
    wind_db = float(eng["wind_gust_link_penalty_db_per_mps"]) * max(0.0, gust - 8.0)
    total = rain_db + fog_db + wind_db
    return {
        "atten_db": round(total, 3),
        "rain_db": round(rain_db, 3),
        "fog_db": round(fog_db, 3),
        "wind_db": round(wind_db, 3),
        "range_km": round(range_km, 3),
        "freq_ghz": freq_ghz,
    }


def assess_environment(
    lat: float,
    lng: float,
    range_km: float,
    freq_ghz: float = 2.4,
) -> Dict[str, Any]:
    weather = fetch_weather(lat, lng)
    vis = optical_visibility_km(weather)
    rf = rf_weather_attenuation_db(range_km, weather, freq_ghz)
    circadian = local_day_night(weather, lat=lat)
    wind = wind_causal_notes(weather)

    # Blend optical ops with day/night (night reduces EO confidence)
    vis_adj = dict(vis)
    vis_adj["day_night_factor"] = circadian["day_night_factor"]
    vis_adj["effective_eo_factor"] = round(
        float(vis["visibility_km"]) / 10.0 * float(circadian["day_night_factor"]),
        3,
    )
    if circadian["phase"] in ("night", "dusk") and vis.get("ops_visual_ok"):
        vis_adj["note"] = (
            f"{vis['note']} · local {circadian['phase']} "
            f"(hour={circadian['local_hour']}) — EO degraded vs day"
        )

    factors = []
    if not vis["ops_visual_ok"]:
        factors.append("low_visibility")
    if rf["atten_db"] >= 3:
        factors.append("rf_weather_loss")
    if float(weather.get("gust_mps") or 0) >= 12:
        factors.append("high_gusts")
    if circadian["phase"] in ("night", "dusk"):
        factors.append(f"local_{circadian['phase']}")
    if wind.get("smoke_dust_factor", 0) >= 0.35:
        factors.append("egress_smoke_dust")
    if wind.get("approach_bias_deg") is not None and float(weather.get("wind_mps") or 0) >= 4:
        factors.append("wind_approach_bias")

    causal_parts = [
        "Weather modifies RF link budget and EO feasibility; "
        "tightens or inflates confidence radius via engine fusion.",
        (
            f"Local {circadian['phase']} @ {circadian['local_hour']:.1f}h "
            f"({circadian['timezone']}) → day_night_factor="
            f"{circadian['day_night_factor']}."
        ),
    ]
    causal_parts.extend(wind.get("notes") or [])

    return {
        "weather": weather,
        "optical": vis_adj,
        "rf_attenuation": rf,
        "day_night": circadian,
        "wind": wind,
        "ops_flags": factors,
        "causal_note": " ".join(causal_parts),
        "causal_notes": causal_parts,
    }
