"""FastAPI 서버 — 실시간 예측 + 위성 지도 UI."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import VERSION
from src.predict_api import predict

WEB_DIR = ROOT / "web"

app = FastAPI(
    title="Drone Operator Localization",
    version=VERSION,
    description="Live OSM buildings + elevation + satellite tiles. Defensive use only.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    latitude: float
    longitude: float
    bearing_degrees: float = Field(..., ge=0, le=359)
    drone_type: str = "unknown"
    signal_strength_dbm: float = -65
    additional_observers: list | None = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "drone-operator-localization",
        "version": VERSION,
        "live_data": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/predict")
def api_predict(body: PredictRequest):
    try:
        result = predict(
            latitude=body.latitude,
            longitude=body.longitude,
            bearing_degrees=body.bearing_degrees,
            drone_type=body.drone_type,
            signal_strength_dbm=body.signal_strength_dbm,
            additional_observers=body.additional_observers,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/")
def index():
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html missing")
    return FileResponse(page)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
