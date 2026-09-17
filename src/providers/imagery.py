"""무료 위성/항공 영상 URL (키 불필요, 표시 시 출처 표기)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict
from urllib.parse import urlencode

from src.utils import deg2num


def satellite_layers(
    lat: float,
    lng: float,
    bbox: Dict[str, float],
    zoom: int = 15,
) -> Dict[str, Dict[str, str]]:
    """지도에 바로 쓸 타일/익스포트 URL."""
    x, y = deg2num(lat, lng, zoom)
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    export = {
        "bbox": f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}",
        "bboxSR": 4326,
        "imageSR": 3857,
        "size": "768,512",
        "format": "jpg",
        "f": "image",
    }
    esri_export = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/export?{urlencode(export)}"
    )
    return {
        "esri_world_imagery": {
            "kind": "xyz",
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "preview": esri_export,
            "attribution": "Esri World Imagery (Maxar)",
        },
        "openstreetmap": {
            "kind": "xyz",
            "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "preview": f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png",
            "attribution": "OpenStreetMap contributors",
        },
        "nasa_gibs_modis": {
            "kind": "xyz",
            "url": (
                "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
                "MODIS_Terra_CorrectedReflectance_TrueColor/default/"
                f"{yesterday}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
            ),
            "preview": (
                "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
                "MODIS_Terra_CorrectedReflectance_TrueColor/default/"
                f"{yesterday}/GoogleMapsCompatible_Level9/{min(zoom, 9)}/{y}/{x}.jpg"
            ),
            "attribution": "NASA EOSDIS GIBS / Terra MODIS",
        },
        "opentopomap": {
            "kind": "xyz",
            "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            "preview": f"https://a.tile.opentopomap.org/{zoom}/{x}/{y}.png",
            "attribution": "OpenTopoMap (CC-BY-SA), SRTM",
        },
    }
