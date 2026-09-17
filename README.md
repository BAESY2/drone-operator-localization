# Drone Operator Localization

방위각 + GPS로 **실제 OpenStreetMap 건물**을 조회하고, **Open-Meteo/SRTM 고도**와 **Esri/NASA 위성 타일**을 붙여 조종자 후보를 뽑는 오픈소스 도구입니다.

목 데이터, 더미 GeoJSON, 합성 후보는 없습니다.

## Legal notice

**DEFENSIVE USE ONLY.** 자기방어·민간 보호 목적. 사용 책임은 사용자에게 있습니다.

## Live data sources (API keys not required)

- Buildings / roads / masts: [OpenStreetMap Overpass](https://overpass-api.de/)
- Elevation: [Open-Meteo](https://open-meteo.com/) → fallback [OpenTopoData SRTM90](https://www.opentopodata.org/)
- Satellite: Esri World Imagery, [NASA GIBS](https://nasa-gibs.github.io/gibs-api-docs/)
- Relief tiles: OpenTopoMap

Please respect OSM tile/Overpass usage policies (identifying User-Agent is set).

## Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# CLI (Seoul City Hall, bearing east)
python src/predict_api.py 37.5665 126.9780 90

# Web map + API  http://127.0.0.1:8000
python -m src
```

`POST /api/predict`

```json
{
  "latitude": 37.5665,
  "longitude": 126.9780,
  "bearing_degrees": 90,
  "signal_strength_dbm": -65
}
```

## Tests

```bash
pytest tests/ -v
```

`test_live.py`는 실제 Overpass/고도 API를 호출합니다.

## License

MIT — [LICENSE](LICENSE)

Map data © OpenStreetMap contributors (ODbL). Imagery © Esri/Maxar, NASA EOSDIS GIBS.
