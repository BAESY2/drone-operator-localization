"""
Open OSINT context: public CCTV tags + traffic/road density from OSM.

Uses only OpenStreetMap public tags — no private camera feeds.
Defensive review: know where sensors exist / traffic choke points.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode

from src.providers.http import get_json
from src.utils import distance_between_coords

Coord = Tuple[float, float]

OVERPASS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)


def _overpass(query: str) -> Dict[str, Any]:
    body = urlencode({"data": query}).encode("utf-8")
    last = None
    for ep in OVERPASS:
        try:
            return get_json(
                ep,
                method="POST",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                cache_key=f"osint:{query[:400]}",
                ttl_sec=6 * 3600,
                timeout=22,
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(str(last) if last else "overpass failed")


def fetch_open_cctv(center: Coord, radius_m: float = 900) -> Dict[str, Any]:
    lat, lng = center
    r = int(min(max(radius_m, 200), 1500))
    query = f"""
[out:json][timeout:20];
(
  node["man_made"="surveillance"](around:{r},{lat:.5f},{lng:.5f});
  node["surveillance"](around:{r},{lat:.5f},{lng:.5f});
  node["camera:type"](around:{r},{lat:.5f},{lng:.5f});
  node["amenity"="fountain"]["surveillance"](around:{r},{lat:.5f},{lng:.5f});
);
out body;
""".strip()
    try:
        payload = _overpass(query)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "cameras": [], "error": str(exc)[:120], "count": 0}

    cams = []
    for el in payload.get("elements") or []:
        if "lat" not in el or "lon" not in el:
            continue
        tags = el.get("tags") or {}
        cams.append(
            {
                "id": el.get("id"),
                "lat": float(el["lat"]),
                "lng": float(el["lon"]),
                "type": tags.get("surveillance")
                or tags.get("camera:type")
                or tags.get("man_made")
                or "camera",
                "operator": tags.get("operator") or tags.get("surveillance:zone"),
                "open_data": True,
                "note": "OSM public tag only — no live feed access",
            }
        )
    return {
        "ok": True,
        "count": len(cams),
        "cameras": cams[:40],
        "source": "openstreetmap",
        "policy": "Public OSM tags only; no unauthorized camera access",
    }


def fetch_traffic_proxy(center: Coord, radius_m: float = 700) -> Dict[str, Any]:
    """
    No live traffic API required — highway class density as congestion proxy.
    """
    lat, lng = center
    r = int(min(max(radius_m, 200), 1200))
    query = f"""
[out:json][timeout:20];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"](around:{r},{lat:.5f},{lng:.5f});
);
out tags center;
""".strip()
    try:
        payload = _overpass(query)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:120], "score": 0.3, "roads": []}

    weights = {
        "motorway": 1.0,
        "trunk": 0.9,
        "primary": 0.75,
        "secondary": 0.55,
        "tertiary": 0.4,
        "residential": 0.25,
    }
    roads = []
    score = 0.0
    for el in payload.get("elements") or []:
        tags = el.get("tags") or {}
        hw = str(tags.get("highway") or "")
        c = el.get("center") or {}
        if "lat" not in c:
            continue
        w = weights.get(hw, 0.2)
        score += w
        roads.append(
            {
                "highway": hw,
                "name": tags.get("name"),
                "lat": float(c["lat"]),
                "lng": float(c["lon"]),
                "weight": w,
            }
        )
    dens = max(0.0, min(1.0, score / 12.0))
    band = "heavy" if dens >= 0.7 else ("moderate" if dens >= 0.4 else "light")
    return {
        "ok": True,
        "score": round(dens, 3),
        "band": band,
        "road_count": len(roads),
        "roads_sample": roads[:25],
        "note": (
            "OSM road-class density proxy — not live congestion. "
            "Heavy → egress delay / civilian exposure risk."
        ),
        "source": "openstreetmap",
    }


def attach_osint_to_candidates(
    candidates: List[Dict[str, Any]],
    cctv: Dict[str, Any],
    traffic: Dict[str, Any],
) -> List[Dict[str, Any]]:
    cams = cctv.get("cameras") or []
    out = []
    for c in candidates:
        item = dict(c)
        lat = float(c.get("latitude") or c.get("lat") or 0)
        lng = float(c.get("longitude") or c.get("lng") or 0)
        near = []
        for cam in cams:
            d_m = distance_between_coords((lat, lng), (cam["lat"], cam["lng"])) * 1000
            if d_m <= 350:
                near.append({**cam, "distance_m": round(d_m, 1)})
        near.sort(key=lambda x: x["distance_m"])
        item["nearby_cctv"] = {
            "count": len(near),
            "cameras": near[:5],
            "review": (
                "Open CCTV tags nearby — cross-check timelines if legally available"
                if near
                else "No OSM surveillance tags within 350m"
            ),
        }
        item["traffic_context"] = {
            "band": traffic.get("band"),
            "score": traffic.get("score"),
            "note": traffic.get("note"),
        }
        # causal: dense traffic + many cams → harder covert OP, easier attribution
        exposure = min(
            1.0,
            0.15 * len(near) + 0.35 * float(traffic.get("score") or 0),
        )
        item["sensor_exposure"] = round(exposure, 3)
        reasons = list(item.get("reasons") or [])
        if near:
            reasons.append(f"osint:cctv_near={len(near)}")
        if traffic.get("band") == "heavy":
            reasons.append("osint:traffic_heavy")
        item["reasons"] = reasons
        out.append(item)
    return out
