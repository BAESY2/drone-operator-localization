"""논문 기반 RF·AoA 수치 모델."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_research() -> Dict:
    path = ROOT / "data" / "research_constants.json"
    return json.loads(path.read_text(encoding="utf-8"))


def constants() -> Dict:
    return load_research()["engine_constants"]


def bearing_likelihood(delta_deg: float, sigma_deg: float | None = None) -> float:
    """
    AoA 오차 가우시안 가능도.
    DRONET ~9.9°, switched-beam ~5° → 기본 σ=10°.
    """
    sigma = sigma_deg if sigma_deg is not None else constants()["aoa_sigma_deg_single"]
    z = delta_deg / max(sigma, 0.5)
    return math.exp(-0.5 * z * z)


def bearing_delta(observed_deg: float, candidate_bearing_deg: float) -> float:
    return abs((candidate_bearing_deg - observed_deg + 180) % 360 - 180)


def log_distance_path_loss_db(distance_m: float, n: float | None = None) -> float:
    """PL(d) = PL(d0) + 10 n log10(d/d0). WLAN F / 도심 기본 n=2.8."""
    c = constants()
    n = n if n is not None else c["path_loss_exponent_urban"]
    d0 = c["ref_distance_m"]
    pl0 = c["pl0_db_2g4"]
    d = max(distance_m, d0)
    return pl0 + 10.0 * n * math.log10(d / d0)


def rssi_to_distance_m(
    rssi_dbm: float,
    tx_dbm: float | None = None,
    n: float | None = None,
) -> float:
    """Prx = Ptx - PL(d) → d."""
    c = constants()
    tx = tx_dbm if tx_dbm is not None else c["assumed_controller_tx_dbm"]
    n = n if n is not None else c["path_loss_exponent_urban"]
    pl0 = c["pl0_db_2g4"]
    d0 = c["ref_distance_m"]
    pl = tx - rssi_dbm
    exponent = (pl - pl0) / (10.0 * n)
    return d0 * (10.0**exponent)


def rssi_annulus_km(
    rssi_dbm: float,
    tx_dbm: float | None = None,
) -> Tuple[float, float, float]:
    """
    섀도잉 σ=6dB (WLAN F)로 거리 환대(annulus) 생성.
    Returns: (d_min_km, d_nominal_km, d_max_km)
    """
    c = constants()
    sigma = c["shadowing_sigma_db"]
    factor = c["rssi_annulus_sigma_factor"]
    d_nom = rssi_to_distance_m(rssi_dbm, tx_dbm=tx_dbm)
    # ΔPL = ±factor*σ → d' = d * 10^(±ΔPL/(10n))
    n = c["path_loss_exponent_urban"]
    scale = 10 ** ((factor * sigma) / (10.0 * n))
    d_min = d_nom / scale
    d_max = d_nom * scale
    return d_min / 1000.0, d_nom / 1000.0, d_max / 1000.0


def rssi_distance_likelihood(distance_km: float, rssi_dbm: float) -> float:
    """관측 RSSI 환대에 대한 거리 가능도."""
    d_min, d_nom, d_max = rssi_annulus_km(rssi_dbm)
    if distance_km < d_min or distance_km > d_max:
        # soft reject
        excess = min(abs(distance_km - d_nom) / max(d_nom, 0.05), 4.0)
        return math.exp(-0.5 * excess * excess) * 0.15
    # within annulus — peak at nominal
    width = max((d_max - d_min) / 2.0, 0.05)
    z = (distance_km - d_nom) / width
    return math.exp(-0.5 * z * z)
