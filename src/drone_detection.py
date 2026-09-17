"""드론 타입 추론."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DRONE_SPECS


def estimate_drone_type(
    signal_strength_dbm: Optional[float] = None,
    bearing_error_deg: Optional[float] = None,
) -> List[Dict]:
    """
    신호강도로부터 드론 타입 후보 추정.

    Returns:
        [{"name": key, "range_km": ..., "confidence": ...}, ...]
    """
    _ = bearing_error_deg  # reserved

    if signal_strength_dbm is None:
        keys = list(DRONE_SPECS.keys())
        conf = round(1.0 / max(len(keys), 1), 3)
        return [
            {
                "name": k,
                "range_km": DRONE_SPECS[k].get("range_km", 10),
                "confidence": conf,
            }
            for k in keys
        ]

    if signal_strength_dbm > -50:
        keys = ["DJI_Mini_3", "DJI_Mini_3_Pro", "DJI_Mini_4_Pro"]
        confidences = [0.5, 0.3, 0.2]
    elif signal_strength_dbm > -60:
        keys = ["DJI_Air_3", "DJI_Air_3S", "DJI_Mini_3_Pro"]
        confidences = [0.45, 0.35, 0.2]
    elif signal_strength_dbm > -70:
        keys = ["DJI_Air_3", "DJI_Air_3S", "DJI_Air_2S"]
        confidences = [0.5, 0.3, 0.2]
    elif signal_strength_dbm > -75:
        keys = ["DJI_Mavic_3", "DJI_Mavic_3_Pro"]
        confidences = [0.6, 0.4]
    else:
        keys = ["DJI_Mavic_3", "Auterion_Skynode", "Lancet"]
        confidences = [0.4, 0.35, 0.25]

    results = []
    for key, conf in zip(keys, confidences):
        if key not in DRONE_SPECS:
            continue
        results.append(
            {
                "name": key,
                "range_km": DRONE_SPECS[key].get("range_km", 10),
                "confidence": conf,
            }
        )
    return results or [
        {"name": "DJI_Air_3", "range_km": 10, "confidence": 0.5}
    ]


def get_drone_range_km(drone_type: str) -> float:
    """드론 타입으로부터 조종 반경 조회."""
    if drone_type in DRONE_SPECS:
        return float(DRONE_SPECS[drone_type].get("range_km", 10))
    # 표시명으로도 검색
    for key, spec in DRONE_SPECS.items():
        if spec.get("name") == drone_type:
            return float(spec.get("range_km", 10))
    return 10.0


def resolve_range_km(
    drone_type: str = "unknown",
    signal_strength_dbm: Optional[float] = None,
) -> tuple[float, str, float]:
    """
    사용할 반경/추정타입/신뢰도 반환.
    Returns: (range_km, estimated_type, confidence)
    """
    if drone_type and drone_type != "unknown":
        return get_drone_range_km(drone_type), drone_type, 0.9

    estimated = estimate_drone_type(signal_strength_dbm)
    top = estimated[0]
    return float(top["range_km"]), top["name"], float(top["confidence"])
