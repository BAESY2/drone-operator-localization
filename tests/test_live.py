"""Live public APIs — no mock datasets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.providers.elevation import fetch_elevations
from src.providers.imagery import satellite_layers
from src.predict_api import predict


def test_open_meteo_seoul():
    elevs, source = fetch_elevations([(37.5665, 126.9780)])
    assert source in {"open-meteo", "opentopodata-srtm90m"}
    assert len(elevs) == 1
    assert 0 <= elevs[0] < 900


def test_satellite_urls():
    layers = satellite_layers(
        37.5665,
        126.9780,
        {"south": 37.55, "north": 37.58, "west": 126.96, "east": 127.00},
    )
    assert "esri_world_imagery" in layers
    assert "nasa_gibs_modis" in layers
    assert "{z}" in layers["esri_world_imagery"]["url"]


def test_live_predict_seoul():
    result = predict(37.5665, 126.9780, 90, "unknown", -65)
    assert result["success"] is True
    assert result["metadata"]["live_data"] is True
    assert result["imagery"]
    assert result["terrain_grid"]["points"]
    assert result["metadata"]["osm_building_count"] >= 1
    assert result["primary_target"] is not None
    assert result["primary_target"]["source"] == "openstreetmap"
    assert "synth_" not in str(result["primary_target"]["id"])
