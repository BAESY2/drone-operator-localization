# API

## GET `/api/health`

```json
{ "status": "ok", "live_data": true }
```

## POST `/api/predict`

실시간 Overpass 건물 + Open-Meteo/SRTM 고도 + 위성 타일 URL.

| Field | Required |
|-------|----------|
| latitude, longitude, bearing_degrees (0–359) | yes |
| drone_type, signal_strength_dbm | no |

응답에 `top_10_candidates`(OSM), `terrain_grid`, `imagery`가 포함됩니다. 합성 후보는 반환하지 않습니다.
