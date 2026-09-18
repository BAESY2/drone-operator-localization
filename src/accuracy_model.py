"""
Accuracy / confidence model — defensive localization only.

Formulas are derived from public papers (see data/research_constants.json):
  - DroNet 2019: mean controller bearing error ≈ 9.9°
  - Sensors 2023: RSSI ratio / log-distance path loss
  - Sensors 2025: switched-beam AoA ≈ 5° mean error

These do NOT invent coordinates — they scale radii / scores only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

# Free-space-ish urban path-loss for control links (2.4 GHz-ish)
PATH_LOSS_EXP = 2.4
REF_DBM_AT_1KM = -55.0  # rough RX at 1 km for typical hobby C2


def bearing_fit_score(bearing_error_deg: Optional[float], tol_deg: float = 18.0) -> float:
    """1.0 at perfect AoA, ~0 at |err| >= tol. Gaussian-like."""
    if bearing_error_deg is None:
        return 0.45
    e = abs(float(bearing_error_deg))
    sigma = max(3.0, tol_deg / 2.5)
    return float(math.exp(-0.5 * (e / sigma) ** 2))


def rssi_range_consistency(
    rssi_dbm: Optional[float],
    dist_km: Optional[float],
) -> float:
    """
    Log-distance: rssi ≈ REF - 10·n·log10(d_km)
    Score = 1 - clipped |Δ|/20 dB
    """
    if rssi_dbm is None or dist_km is None or dist_km <= 0.05:
        return 0.4
    expected = REF_DBM_AT_1KM - 10.0 * PATH_LOSS_EXP * math.log10(max(dist_km, 0.05))
    delta = abs(float(rssi_dbm) - expected)
    return float(max(0.0, min(1.0, 1.0 - delta / 22.0)))


def mid_elevation_preference(height_m: Optional[float]) -> float:
    """Human OP: prefer mid roofs (~8–35 m), not tallest."""
    if height_m is None:
        return 0.5
    h = float(height_m)
    if 8.0 <= h <= 35.0:
        return 1.0
    if h < 8.0:
        return max(0.25, h / 8.0)
    # Soft penalty above 35 m (no tallest-roof bias)
    return max(0.2, 1.0 - (h - 35.0) / 80.0)


def expected_radius_m(
    *,
    bearing_err_deg: Optional[float],
    dist_m: Optional[float],
    n_observers: int = 1,
    aoa_sigma_deg: float = 9.9,
) -> float:
    """
    Geometric: σ_cross ≈ d · tan(σ_θ)
    Multi-obs shrink ~ 1/√n
    """
    d = float(dist_m or 800.0)
    sigma = float(aoa_sigma_deg)
    if bearing_err_deg is not None:
        sigma = max(sigma, abs(float(bearing_err_deg)) * 0.85)
    cross = d * math.tan(math.radians(min(sigma, 45.0)))
    n = max(1, int(n_observers))
    return float(max(25.0, min(2000.0, cross / math.sqrt(n))))


def composite_confidence(
    parts: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Weighted mean of 0..1 parts → confidence label inputs."""
    if not parts:
        return 0.35, {}
    wsum = sum(parts.values())
    if wsum <= 0:
        return 0.35, parts
    # Already weighted fragments — normalize by count
    avg = sum(parts.values()) / len(parts)
    return float(max(0.05, min(0.98, avg))), {k: round(v, 3) for k, v in parts.items()}


def accuracy_pack_for_candidate(
    c: Dict[str, Any],
    *,
    rssi_dbm: Optional[float] = None,
    n_observers: int = 1,
) -> Dict[str, Any]:
    brg_err = c.get("bearing_error_deg")
    dist_m = c.get("distance_m")
    if dist_m is None and c.get("location"):
        dist_m = (c.get("location") or {}).get("distance_m")
    dist_km = (float(dist_m) / 1000.0) if dist_m is not None else None
    height = c.get("height_m")
    parts = {
        "bearing_fit": bearing_fit_score(brg_err),
        "rssi_range": rssi_range_consistency(rssi_dbm, dist_km),
        "mid_elev": mid_elevation_preference(height),
        "human": float(c.get("human_score") or 0.45),
    }
    conf, detail = composite_confidence(parts)
    rad = expected_radius_m(
        bearing_err_deg=brg_err,
        dist_m=float(dist_m) if dist_m is not None else None,
        n_observers=n_observers,
    )
    return {
        "confidence": round(conf, 3),
        "parts": detail,
        "model_radius_m": round(rad, 1),
        "formula": "σ_cross=d·tan(σ_θ)/√n · RSSI log-distance · mid-elev prior",
    }
