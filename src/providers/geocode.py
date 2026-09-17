"""전 세계 지오코딩 — OpenStreetMap Nominatim (키 불필요)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.providers.http import get_json

NOMINATIM = "https://nominatim.openstreetmap.org"


def search_places(query: str, limit: int = 8, countrycodes: Optional[str] = None) -> List[Dict[str, Any]]:
    """도시·주소·랜드마크 검색 (전 세계)."""
    params: Dict[str, Any] = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": max(1, min(limit, 15)),
    }
    if countrycodes:
        params["countrycodes"] = countrycodes
    data = get_json(
        f"{NOMINATIM}/search",
        params=params,
        cache_key=f"nominatim:search:{query}:{countrycodes}:{limit}",
        ttl_sec=24 * 3600,
        timeout=20,
    )
    results = []
    for item in data or []:
        addr = item.get("address") or {}
        results.append(
            {
                "display_name": item.get("display_name"),
                "latitude": float(item["lat"]),
                "longitude": float(item["lon"]),
                "type": item.get("type"),
                "class": item.get("class"),
                "country": addr.get("country"),
                "country_code": (addr.get("country_code") or "").upper(),
                "city": addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("municipality"),
                "boundingbox": item.get("boundingbox"),
            }
        )
    return results


def reverse_geocode(lat: float, lng: float) -> Dict[str, Any]:
    """좌표 → 주소/국가 (전 세계)."""
    data = get_json(
        f"{NOMINATIM}/reverse",
        params={
            "lat": f"{lat:.6f}",
            "lon": f"{lng:.6f}",
            "format": "json",
            "addressdetails": 1,
            "zoom": 16,
        },
        cache_key=f"nominatim:rev:{lat:.5f}:{lng:.5f}",
        ttl_sec=24 * 3600,
        timeout=20,
    )
    addr = (data or {}).get("address") or {}
    return {
        "display_name": (data or {}).get("display_name"),
        "latitude": lat,
        "longitude": lng,
        "country": addr.get("country"),
        "country_code": (addr.get("country_code") or "").upper(),
        "city": addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality"),
        "road": addr.get("road"),
        "postcode": addr.get("postcode"),
        "raw_address": addr,
    }
