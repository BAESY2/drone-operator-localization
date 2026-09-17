# Drone Operator Localization

Worldwide defensive tool: **bearing + GPS** → live **OpenStreetMap buildings** anywhere on Earth, plus elevation, satellite, optional flight-track prior.

No mock geospatial dumps. Works in any country (OSM coverage varies by region).

## Legal notice

**DEFENSIVE USE ONLY.** Self-defense / civilian protection. Users assume full liability.

## Live APIs (no keys)

| API | Purpose |
|-----|---------|
| Overpass (OSM) | Buildings, roads, masts — **global** |
| Open-Meteo / OpenTopoData | Elevation — **global** |
| Esri / NASA GIBS / OSM tiles | Imagery — **global** |
| Nominatim | Place search & reverse geocode — **global** |
| Local `/api/measure` | Distance, bearing, elev delta |

## Run

```bash
pip install -r requirements.txt
python -m src
# → http://127.0.0.1:8000
```

### Endpoints

- `POST /api/predict` — localize (+ optional `flight_track`, asset)
- `GET /api/geocode?q=Kyiv` — worldwide place search (`countrycodes` optional filter)
- `GET /api/reverse?lat=&lng=` — address / country
- `POST /api/measure` — distance / bearing / elevation between two points
- `POST /api/measure/path` — multi-point path length
- `POST /api/elevation` — batch elevations
- `GET /api/health`

## UI

- EN / 한국어 toggle
- Search any city worldwide + regional presets
- GPS or map click for observer / asset
- Measure mode (A→B) via live elevation API
- Probability-colored candidates, corridor, threat rings

## License

MIT — see [LICENSE](LICENSE)
