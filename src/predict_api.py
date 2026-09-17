"""CLI/API 진입점 — 실시간 OSM·고도·위성 예측."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import API_MAX_CANDIDATES, VERSION
from src.drone_detection import resolve_range_km
from src.localize import localize_drone_operator


def build_tactical_output(
    primary: Optional[Dict[str, Any]],
    bearing_degrees: float,
    drone_range_km: float,
) -> Dict[str, Any]:
    center_lat = primary["latitude"] if primary else None
    center_lng = primary["longitude"] if primary else None
    return {
        "strike_zone": {
            "center_lat": center_lat,
            "center_lng": center_lng,
            "primary_radius_m": 500,
            "secondary_radius_m": 1500,
        },
        "danger_zone": {
            "radius_m": int(drone_range_km * 1000) + 500,
            "description": "Estimated drone control range + buffer",
        },
        "escape_vector": {
            "bearing_degrees": int((bearing_degrees + 180) % 360),
            "recommended_distance_m": int(max(drone_range_km, 3) * 1000),
            "reason": "Opposite direction of observed drone approach",
        },
    }


def predict(
    latitude: float,
    longitude: float,
    bearing_degrees: float,
    drone_type: str = "unknown",
    signal_strength_dbm: float = -65,
    additional_observers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()

    if bearing_degrees is None:
        return {
            "success": False,
            "error": "MISSING_BEARING",
            "message": "Bearing angle (bearing_degrees) is required",
        }
    if not (0 <= float(bearing_degrees) <= 359):
        return {
            "success": False,
            "error": "INVALID_BEARING",
            "message": "bearing_degrees must be 0-359",
        }

    range_km, estimated_type, signal_conf = resolve_range_km(
        drone_type=drone_type,
        signal_strength_dbm=signal_strength_dbm,
    )
    tolerance = 45
    triangulation = "single_observer"
    if additional_observers:
        triangulation = "multi_observer"
        tolerance = max(15, 45 // (1 + len(additional_observers)))

    result = localize_drone_operator(
        center_lat=latitude,
        center_lng=longitude,
        bearing_deg=bearing_degrees,
        drone_range_km=range_km,
        num_results=API_MAX_CANDIDATES,
        bearing_tolerance_deg=tolerance,
    )

    primary = result.get("primary_target")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    conf_label = (
        "high" if signal_conf >= 0.7 else ("medium" if signal_conf >= 0.4 else "low")
    )
    osm_count = result.get("osm_building_count") or 0

    return {
        "success": True,
        "version": VERSION,
        "primary_target": primary,
        "top_10_candidates": result.get("top_10_candidates") or [],
        "drone_info": {
            "estimated_type": estimated_type,
            "estimated_range_km": range_km,
            "signal_confidence": conf_label,
        },
        "tactical_output": build_tactical_output(primary, bearing_degrees, range_km),
        "search_sector": result.get("search_sector"),
        "terrain_grid": result.get("terrain_grid"),
        "imagery": result.get("imagery"),
        "bbox": result.get("bbox"),
        "attribution": result.get("attribution") or [],
        "metadata": {
            "processing_time_ms": elapsed_ms,
            "triangulation": triangulation,
            "filtered_count": result.get("filtered_count", 0),
            "osm_building_count": osm_count,
            "query_range_km": result.get("query_range_km"),
            "sources": result.get("sources"),
            "live_data": True,
            "accuracy_note": (
                "OSM buildings + SRTM/Open-Meteo elevation. "
                "Empty candidates means no mapped buildings in the search fan."
                if osm_count == 0
                else "Live OSM footprints; single-observer bearing still has km-scale uncertainty."
            ),
        },
    }


def main() -> None:
    if len(sys.argv) < 4:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": "MISSING_ARGS",
                    "message": "Usage: python src/predict_api.py <lat> <lng> <bearing> [drone_type] [signal_dbm]",
                }
            )
        )
        sys.exit(1)

    response = predict(
        latitude=float(sys.argv[1]),
        longitude=float(sys.argv[2]),
        bearing_degrees=float(sys.argv[3]),
        drone_type=sys.argv[4] if len(sys.argv) > 4 else "unknown",
        signal_strength_dbm=float(sys.argv[5]) if len(sys.argv) > 5 else -65.0,
    )
    payload = json.dumps(response, ensure_ascii=True)
    sys.stdout.buffer.write(payload.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.exit(0 if response.get("success") else 1)


if __name__ == "__main__":
    main()
