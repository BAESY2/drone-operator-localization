"""CLI/API — defensive causal engine (algorithm-first; AI optional)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defensive_engine import run_defensive_engine


def predict(
    latitude: float,
    longitude: float,
    bearing_degrees: float,
    drone_type: str = "unknown",
    signal_strength_dbm: float = -65,
    additional_observers: Optional[List[Dict[str, Any]]] = None,
    asset_latitude: Optional[float] = None,
    asset_longitude: Optional[float] = None,
    flight_track: Optional[List[Dict[str, Any]]] = None,
    use_ai: bool = False,
    ai_api_key: Optional[str] = None,
    ai_base_url: Optional[str] = None,
    ai_model: Optional[str] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    return run_defensive_engine(
        latitude,
        longitude,
        bearing_degrees,
        drone_type=drone_type,
        signal_strength_dbm=signal_strength_dbm,
        additional_observers=additional_observers,
        asset_latitude=asset_latitude,
        asset_longitude=asset_longitude,
        flight_track=flight_track,
        use_ai=use_ai,
        ai_api_key=ai_api_key,
        ai_base_url=ai_base_url,
        ai_model=ai_model,
        lang=lang,
    )


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
