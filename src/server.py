"""FastAPI — global predict, geocode, measure, map UI."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    ALLOWED_ORIGINS,
    ENABLE_OPENAPI,
    RATE_LIMIT_PER_MIN,
    THEATER_ENFORCE,
    VERSION,
)
from src.measure import measure_path, measure_segment
from src.predict_api import predict
from src.providers.elevation import fetch_elevations
from src.providers.geocode import reverse_geocode, search_places
from src.recon import build_recon_plan
from src.ai_assist import enrich_recon_plan
from src.analyze_pipeline import run_defensive_analysis
from src.security import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    assert_theater,
    safe_error_detail,
)

WEB_DIR = ROOT / "web"

app = FastAPI(
    title="DOL C2 Console",
    version=VERSION,
    description="Defensive drone-operator localization — OSM, elevation, measure, egress.",
    docs_url="/docs" if ENABLE_OPENAPI else None,
    redoc_url="/redoc" if ENABLE_OPENAPI else None,
    openapi_url="/openapi.json" if ENABLE_OPENAPI else None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.add_middleware(RateLimitMiddleware, limit_per_min=RATE_LIMIT_PER_MIN)
app.add_middleware(SecurityHeadersMiddleware)


class ObserverIn(BaseModel):
    latitude: float
    longitude: float
    bearing_degrees: float = Field(..., ge=0, le=359)


class TrackPointIn(BaseModel):
    t: float = 0
    lat: Optional[float] = None
    latitude: Optional[float] = None
    lng: Optional[float] = None
    longitude: Optional[float] = None
    alt_m: Optional[float] = None
    altitude_m: Optional[float] = None
    yaw_deg: Optional[float] = None


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    bearing_degrees: float = Field(..., ge=0, le=359)
    drone_type: str = "unknown"
    signal_strength_dbm: float = -65
    additional_observers: list[ObserverIn] | None = None
    asset_latitude: Optional[float] = Field(None, ge=-90, le=90)
    asset_longitude: Optional[float] = Field(None, ge=-180, le=180)
    flight_track: list[TrackPointIn] | None = None
    use_ai: bool = False
    lang: str = "en"
    ai_api_key: Optional[str] = None
    ai_base_url: Optional[str] = None
    ai_model: Optional[str] = None


class PointIn(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class MeasureRequest(BaseModel):
    from_point: PointIn
    to_point: PointIn
    include_elevation: bool = True


class MeasurePathRequest(BaseModel):
    points: List[PointIn] = Field(..., min_length=2)
    include_elevation: bool = True


class ElevRequest(BaseModel):
    points: List[PointIn] = Field(..., min_length=1, max_length=100)


class CandidateIn(BaseModel):
    latitude: float
    longitude: float
    rank: Optional[int] = None
    probability: Optional[float] = None


class ReconRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    bearing_degrees: float = Field(..., ge=0, le=359)
    profile: str = "verify_approach"
    airframe: str = "small_quad"
    range_km: float = Field(3.0, ge=0.3, le=50)
    alt_m: Optional[float] = Field(None, ge=20, le=500)
    speed_mps: Optional[float] = Field(None, ge=2, le=40)
    mission_outcome: str = "unknown"
    asset_latitude: Optional[float] = Field(None, ge=-90, le=90)
    asset_longitude: Optional[float] = Field(None, ge=-180, le=180)
    candidates: list[CandidateIn] | None = None
    use_ai: bool = False
    lang: str = "en"
    ai_api_key: Optional[str] = None
    ai_base_url: Optional[str] = None
    ai_model: Optional[str] = None


class AnalyzeRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    bearing_degrees: float = Field(..., ge=0, le=359)
    signal_strength_dbm: float = -65
    asset_latitude: Optional[float] = Field(None, ge=-90, le=90)
    asset_longitude: Optional[float] = Field(None, ge=-180, le=180)
    mission_outcome: str = "unknown"
    use_ai: bool = False
    lang: str = "en"
    ai_api_key: Optional[str] = None
    ai_base_url: Optional[str] = None
    ai_model: Optional[str] = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "dol-c2",
        "version": VERSION,
        "engine": "defensive_causal_v3",
        "ai_role": "optional_assist",
        "region": "global",
        "region_enforce": THEATER_ENFORCE,
        "live_data": True,
        "kinematics": True,
        "recon": True,
        "apis": [
            "/api/predict",
            "/api/analyze",
            "/api/recon/plan",
            "/api/geocode",
            "/api/reverse",
            "/api/measure",
            "/api/measure/path",
            "/api/elevation",
            "/api/health",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/geocode")
def api_geocode(
    q: str = Query(..., min_length=2),
    limit: int = Query(8, ge=1, le=15),
    countrycodes: Optional[str] = Query(
        None, description="Optional ISO2 filter e.g. kr,us,ua — omit for worldwide"
    ),
):
    try:
        return {
            "success": True,
            "query": q,
            "results": search_places(q, limit, countrycodes),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=safe_error_detail(exc)) from exc


@app.get("/api/reverse")
def api_reverse(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
):
    assert_theater(lat, lng, "coordinate")
    try:
        return {"success": True, **reverse_geocode(lat, lng)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=safe_error_detail(exc)) from exc


@app.post("/api/measure")
def api_measure(body: MeasureRequest):
    assert_theater(body.from_point.latitude, body.from_point.longitude, "from_point")
    assert_theater(body.to_point.latitude, body.to_point.longitude, "to_point")
    try:
        return {
            "success": True,
            **measure_segment(
                (body.from_point.latitude, body.from_point.longitude),
                (body.to_point.latitude, body.to_point.longitude),
                include_elevation=body.include_elevation,
            ),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=safe_error_detail(exc)) from exc


@app.post("/api/measure/path")
def api_measure_path(body: MeasurePathRequest):
    for i, p in enumerate(body.points):
        assert_theater(p.latitude, p.longitude, f"points[{i}]")
    try:
        pts = [(p.latitude, p.longitude) for p in body.points]
        return measure_path(pts, include_elevation=body.include_elevation)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=safe_error_detail(exc)) from exc


@app.post("/api/elevation")
def api_elevation(body: ElevRequest):
    for i, p in enumerate(body.points):
        assert_theater(p.latitude, p.longitude, f"points[{i}]")
    try:
        pts = [(p.latitude, p.longitude) for p in body.points]
        elevs, src = fetch_elevations(pts)
        return {
            "success": True,
            "source": src,
            "points": [
                {"latitude": lat, "longitude": lng, "elevation_m": elev}
                for (lat, lng), elev in zip(pts, elevs)
            ],
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=safe_error_detail(exc)) from exc


@app.post("/api/predict")
def api_predict(body: PredictRequest):
    assert_theater(body.latitude, body.longitude, "observer")
    if body.asset_latitude is not None and body.asset_longitude is not None:
        assert_theater(body.asset_latitude, body.asset_longitude, "asset")
    try:
        observers = None
        if body.additional_observers:
            observers = [o.model_dump() for o in body.additional_observers]
        track = None
        if body.flight_track:
            track = [p.model_dump(exclude_none=True) for p in body.flight_track]
        result = predict(
            latitude=body.latitude,
            longitude=body.longitude,
            bearing_degrees=body.bearing_degrees,
            drone_type=body.drone_type,
            signal_strength_dbm=body.signal_strength_dbm,
            additional_observers=observers,
            asset_latitude=body.asset_latitude,
            asset_longitude=body.asset_longitude,
            flight_track=track,
            use_ai=body.use_ai,
            ai_api_key=body.ai_api_key,
            ai_base_url=body.ai_base_url,
            ai_model=body.ai_model,
            lang=body.lang,
        )
        try:
            result["place"] = reverse_geocode(body.latitude, body.longitude)
        except Exception:  # noqa: BLE001
            result["place"] = None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=safe_error_detail(exc)) from exc
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/recon/plan")
def api_recon(body: ReconRequest):
    assert_theater(body.latitude, body.longitude, "observer")
    try:
        target = None
        if body.asset_latitude is not None and body.asset_longitude is not None:
            assert_theater(body.asset_latitude, body.asset_longitude, "asset")
            target = (body.asset_latitude, body.asset_longitude)
        cands = [c.model_dump() for c in body.candidates] if body.candidates else None
        plan = build_recon_plan(
            (body.latitude, body.longitude),
            body.bearing_degrees,
            profile=body.profile,
            airframe=body.airframe,
            candidates=cands,
            target=target,
            range_km=body.range_km,
            alt_m=body.alt_m,
            speed_mps=body.speed_mps,
            mission_outcome=body.mission_outcome,
        )
        if body.use_ai:
            plan = enrich_recon_plan(
                plan,
                lang=body.lang,
                api_key=body.ai_api_key,
                base_url=body.ai_base_url,
                model=body.ai_model,
                force=bool(body.ai_api_key),
            )
        else:
            plan["ai"] = {"applied": False, "reason": "use_ai=false"}
        # never echo key
        return plan
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=safe_error_detail(exc)) from exc


@app.post("/api/analyze")
def api_analyze(body: AnalyzeRequest):
    """Defensive pipeline only — localization, risk, recon. No strike."""
    assert_theater(body.latitude, body.longitude, "observer")
    try:
        result = run_defensive_analysis(
            body.latitude,
            body.longitude,
            body.bearing_degrees,
            signal_strength_dbm=body.signal_strength_dbm,
            asset_latitude=body.asset_latitude,
            asset_longitude=body.asset_longitude,
            mission_outcome=body.mission_outcome,
            lang=body.lang,
            use_ai=body.use_ai,
            ai_api_key=body.ai_api_key,
            ai_base_url=body.ai_base_url,
            ai_model=body.ai_model,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result)
        return result
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=safe_error_detail(exc)) from exc


@app.get("/")
def index():
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html missing")
    return FileResponse(
        page,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
