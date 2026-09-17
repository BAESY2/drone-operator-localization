"""
OSM landcover + defensive concealment OSINT near a point.

Public Overpass tags only. Concealment proxies are labeled concealment_osint
for DEFENSIVE awareness (where someone might hide) — never strike targeting.
Structured output feeds egress sketches and candidate scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from src.providers.http import get_json
from src.utils import distance_between_coords

Coord = Tuple[float, float]

OVERPASS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

# landuse / natural / landcover → scoring buckets
LANDCOVER_BUCKETS = {
    "forest": "forest",
    "wood": "forest",
    "scrub": "rough",
    "heath": "rough",
    "meadow": "open_plain",
    "grassland": "open_plain",
    "grass": "open_plain",
    "farmland": "open_plain",
    "farmyard": "open_plain",
    "orchard": "open_plain",
    "vineyard": "open_plain",
    "wetland": "rough",
    "marsh": "rough",
    "swamp": "rough",
    "residential": "urban_road",
    "apartments": "urban_road",
    "commercial": "urban_road",
    "retail": "urban_road",
    "industrial": "urban_road",
    "construction": "urban_road",
}

# Explicit OSM labels we accept for landcover sampling
LANDCOVER_TAGS = (
    "forest",
    "wood",
    "scrub",
    "meadow",
    "grassland",
    "farmland",
    "wetland",
    "residential",
)

CONCEALMENT_POLICY = (
    "concealment_osint — open OSM tags only; defensive hide-site awareness; "
    "NOT for strike / targeting tasking"
)


def _overpass(query: str, cache_prefix: str = "terrain") -> Dict[str, Any]:
    body = urlencode({"data": query}).encode("utf-8")
    last = None
    for ep in OVERPASS:
        try:
            return get_json(
                ep,
                method="POST",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                cache_key=f"{cache_prefix}:{query[:420]}",
                ttl_sec=6 * 3600,
                timeout=22,
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(str(last) if last else "overpass failed")


def _el_center(el: Dict[str, Any]) -> Optional[Coord]:
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])
    c = el.get("center") or {}
    if "lat" in c and "lon" in c:
        return float(c["lat"]), float(c["lon"])
    return None


def _landcover_label(tags: Dict[str, Any]) -> Optional[str]:
    """Map OSM tags → one of LANDCOVER_TAGS (residential = urban fabric)."""
    natural = str(tags.get("natural") or "").lower()
    landuse = str(tags.get("landuse") or "").lower()
    landcover = str(tags.get("landcover") or "").lower()

    for key in (natural, landuse, landcover):
        if key in LANDCOVER_TAGS:
            return key
        if key == "wood":
            return "wood"
        if key in ("grass", "meadow"):
            return "meadow" if key == "meadow" else "grassland"
        if key in ("marsh", "swamp", "bog"):
            return "wetland"

    bld = str(tags.get("building") or "").lower()
    if landuse == "residential" or bld in {"residential", "apartments", "house"}:
        return "residential"
    return None


def fetch_landcover(center: Coord, radius_m: float = 800) -> Dict[str, Any]:
    """
    OSM landcover / natural / residential fabric near a point.
    Categories: forest, wood, scrub, meadow, grassland, farmland, wetland, residential.
    """
    lat, lng = center
    r = int(min(max(radius_m, 200), 2000))
    query = f"""
[out:json][timeout:22];
(
  way["natural"~"^(wood|forest|scrub|grassland|wetland|heath)$"](around:{r},{lat:.5f},{lng:.5f});
  relation["natural"~"^(wood|forest|scrub|grassland|wetland|heath)$"](around:{r},{lat:.5f},{lng:.5f});
  way["landuse"~"^(forest|meadow|farmland|farmyard|orchard|vineyard|grass|residential|industrial|commercial)$"](around:{r},{lat:.5f},{lng:.5f});
  relation["landuse"~"^(forest|meadow|farmland|farmyard|orchard|vineyard|grass|residential|industrial|commercial)$"](around:{r},{lat:.5f},{lng:.5f});
  way["landcover"~"^(trees|grass|scrub)$"](around:{r},{lat:.5f},{lng:.5f});
  node["natural"~"^(wood|forest|scrub|wetland)$"](around:{r},{lat:.5f},{lng:.5f});
);
out tags center;
""".strip()
    try:
        payload = _overpass(query, cache_prefix="landcover")
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc)[:120],
            "features": [],
            "counts": {},
            "fractions": {},
            "source": "openstreetmap",
        }

    counts: Dict[str, int] = {k: 0 for k in LANDCOVER_TAGS}
    features: List[Dict[str, Any]] = []
    for el in payload.get("elements") or []:
        tags = el.get("tags") or {}
        label = _landcover_label(tags)
        if not label:
            # map industrial/commercial into residential urban fabric bucket for counts
            lu = str(tags.get("landuse") or "").lower()
            if lu in ("industrial", "commercial", "retail"):
                label = "residential"
            elif str(tags.get("natural") or "").lower() == "heath":
                label = "scrub"
            elif lu in ("farmyard", "orchard", "vineyard", "grass"):
                label = "farmland" if lu != "grass" else "grassland"
            else:
                continue
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
        pt = _el_center(el)
        if not pt:
            continue
        d_m = distance_between_coords(center, pt) * 1000.0
        features.append(
            {
                "id": el.get("id"),
                "label": label,
                "lat": pt[0],
                "lng": pt[1],
                "distance_m": round(d_m, 1),
                "tags": {
                    k: tags[k]
                    for k in ("natural", "landuse", "landcover", "name")
                    if k in tags
                },
            }
        )

    total = sum(counts.values()) or 1
    fractions = {k: round(v / total, 3) for k, v in counts.items()}
    features.sort(key=lambda x: x["distance_m"])
    return {
        "ok": True,
        "radius_m": r,
        "counts": counts,
        "fractions": fractions,
        "feature_count": len(features),
        "features": features[:60],
        "source": "openstreetmap",
        "note": "OSM landcover/natural/residential fabric — defensive terrain context",
    }


def fetch_concealment_osint(center: Coord, radius_m: float = 900) -> Dict[str, Any]:
    """
    Defensive concealment proxies from open OSM tags only.
    Labeled concealment_osint — where someone might hide; never strike targeting.
    """
    lat, lng = center
    r = int(min(max(radius_m, 200), 2000))
    query = f"""
[out:json][timeout:22];
(
  node["military"="bunker"](around:{r},{lat:.5f},{lng:.5f});
  way["military"="bunker"](around:{r},{lat:.5f},{lng:.5f});
  node["bunker"](around:{r},{lat:.5f},{lng:.5f});
  way["bunker"](around:{r},{lat:.5f},{lng:.5f});
  node["man_made"="bunker"](around:{r},{lat:.5f},{lng:.5f});
  way["man_made"="bunker"](around:{r},{lat:.5f},{lng:.5f});
  node["man_made"="tunnel"](around:{r},{lat:.5f},{lng:.5f});
  way["man_made"="tunnel"](around:{r},{lat:.5f},{lng:.5f});
  node["tunnel"="yes"](around:{r},{lat:.5f},{lng:.5f});
  way["tunnel"="yes"](around:{r},{lat:.5f},{lng:.5f});
  node["natural"="cave_entrance"](around:{r},{lat:.5f},{lng:.5f});
  way["natural"="cave_entrance"](around:{r},{lat:.5f},{lng:.5f});
  node["landuse"="quarry"](around:{r},{lat:.5f},{lng:.5f});
  way["landuse"="quarry"](around:{r},{lat:.5f},{lng:.5f});
  node["historic"="ruins"](around:{r},{lat:.5f},{lng:.5f});
  way["historic"="ruins"](around:{r},{lat:.5f},{lng:.5f});
  node["ruins"="yes"](around:{r},{lat:.5f},{lng:.5f});
  way["ruins"="yes"](around:{r},{lat:.5f},{lng:.5f});
  node["abandoned"="yes"](around:{r},{lat:.5f},{lng:.5f});
  way["abandoned"="yes"](around:{r},{lat:.5f},{lng:.5f});
  node["building"="ruins"](around:{r},{lat:.5f},{lng:.5f});
  way["building"="ruins"](around:{r},{lat:.5f},{lng:.5f});
);
out tags center;
""".strip()
    try:
        payload = _overpass(query, cache_prefix="conceal")
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc)[:120],
            "sites": [],
            "count": 0,
            "label": "concealment_osint",
            "policy": CONCEALMENT_POLICY,
            "source": "openstreetmap",
        }

    sites: List[Dict[str, Any]] = []
    for el in payload.get("elements") or []:
        tags = el.get("tags") or {}
        kind = _concealment_kind(tags)
        if not kind:
            continue
        pt = _el_center(el)
        if not pt:
            continue
        d_m = distance_between_coords(center, pt) * 1000.0
        sites.append(
            {
                "id": el.get("id"),
                "kind": kind,
                "lat": pt[0],
                "lng": pt[1],
                "distance_m": round(d_m, 1),
                "name": tags.get("name"),
                "label": "concealment_osint",
                "open_data": True,
                "note": "Defensive hide-site proxy from public OSM — not a target list",
            }
        )
    sites.sort(key=lambda x: x["distance_m"])
    return {
        "ok": True,
        "label": "concealment_osint",
        "count": len(sites),
        "sites": sites[:40],
        "radius_m": r,
        "policy": CONCEALMENT_POLICY,
        "source": "openstreetmap",
    }


def _concealment_kind(tags: Dict[str, Any]) -> Optional[str]:
    mil = str(tags.get("military") or "").lower()
    man = str(tags.get("man_made") or "").lower()
    natural = str(tags.get("natural") or "").lower()
    landuse = str(tags.get("landuse") or "").lower()
    historic = str(tags.get("historic") or "").lower()
    building = str(tags.get("building") or "").lower()

    if mil == "bunker" or man == "bunker" or "bunker" in tags:
        return "bunker" if mil != "bunker" else "military_bunker"
    if mil and "bunker" in mil:
        return "military_bunker"
    if man == "tunnel" or str(tags.get("tunnel") or "").lower() == "yes":
        return "tunnel"
    if natural == "cave_entrance":
        return "cave_entrance"
    if landuse == "quarry":
        return "quarry"
    if historic == "ruins" or building == "ruins" or str(tags.get("ruins") or "").lower() == "yes":
        return "ruins"
    if str(tags.get("abandoned") or "").lower() in ("yes", "true", "1"):
        return "abandoned"
    return None


def classify_terrain_band(landcover: Dict[str, Any]) -> Dict[str, Any]:
    """
    Band: urban_road / forest / open_plain / rough / mixed with [0,1] scores.
    """
    fr = landcover.get("fractions") or {}
    counts = landcover.get("counts") or {}

    forest_n = float(counts.get("forest") or 0) + float(counts.get("wood") or 0)
    open_n = (
        float(counts.get("meadow") or 0)
        + float(counts.get("grassland") or 0)
        + float(counts.get("farmland") or 0)
    )
    rough_n = float(counts.get("scrub") or 0) + float(counts.get("wetland") or 0)
    urban_n = float(counts.get("residential") or 0)
    total = forest_n + open_n + rough_n + urban_n

    if total <= 0:
        # no OSM polygons — weak priors
        scores = {
            "urban_road": 0.25,
            "forest": 0.2,
            "open_plain": 0.3,
            "rough": 0.15,
            "mixed": 0.4,
        }
        band = "mixed"
        confidence = 0.15
    else:
        scores = {
            "urban_road": round(urban_n / total, 3),
            "forest": round(forest_n / total, 3),
            "open_plain": round(open_n / total, 3),
            "rough": round(rough_n / total, 3),
            "mixed": 0.0,
        }
        # entropy-like mixed: if no dominant class
        ranked = sorted(
            (("urban_road", scores["urban_road"]),
             ("forest", scores["forest"]),
             ("open_plain", scores["open_plain"]),
             ("rough", scores["rough"])),
            key=lambda x: x[1],
            reverse=True,
        )
        top_name, top_s = ranked[0]
        second_s = ranked[1][1]
        if top_s < 0.45 or (top_s - second_s) < 0.12:
            band = "mixed"
            scores["mixed"] = round(min(1.0, 1.0 - top_s + second_s * 0.5), 3)
        else:
            band = top_name
            scores["mixed"] = round(max(0.0, 1.0 - top_s), 3)
        confidence = round(min(1.0, 0.25 + total / 40.0), 3)

    return {
        "band": band,
        "scores": scores,
        "confidence": confidence,
        "dominant": band,
        "note": (
            f"Local terrain band={band} — informs egress mode bias "
            f"(urban→road, forest→cover, open→exposure, rough→slow off-road)."
        ),
    }


def terrain_score_for_candidate(
    candidate: Dict[str, Any],
    terrain: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Soft factors for candidate scoring / egress fitness from local terrain context.
    """
    band_info = terrain.get("band") or {}
    scores = band_info.get("scores") or {}
    concealment = terrain.get("concealment_osint") or {}
    sites = concealment.get("sites") or []

    lat = float(candidate.get("latitude") or candidate.get("lat") or 0)
    lng = float(candidate.get("longitude") or candidate.get("lng") or 0)
    near_hide = 0
    min_hide_m = None
    for s in sites:
        d_m = distance_between_coords((lat, lng), (s["lat"], s["lng"])) * 1000.0
        if d_m <= 400:
            near_hide += 1
            if min_hide_m is None or d_m < min_hide_m:
                min_hide_m = d_m

    # concealment prior: forest/rough + nearby hide proxies (defensive)
    cover = float(scores.get("forest") or 0) * 0.55 + float(scores.get("rough") or 0) * 0.35
    if near_hide:
        cover = min(1.0, cover + 0.15 + 0.05 * min(near_hide, 3))
    # urban helps blend / road egress; open_plain hurts concealment
    blend = float(scores.get("urban_road") or 0)
    exposure = float(scores.get("open_plain") or 0)
    egress_road_bias = min(1.0, blend * 0.7 + (1.0 - exposure) * 0.3)
    egress_offroad_bias = min(
        1.0,
        float(scores.get("forest") or 0) * 0.4
        + float(scores.get("rough") or 0) * 0.45
        + float(scores.get("open_plain") or 0) * 0.15,
    )

    return {
        "terrain_band": band_info.get("band"),
        "concealment_fit": round(cover, 3),
        "urban_blend": round(blend, 3),
        "open_exposure": round(exposure, 3),
        "egress_road_bias": round(egress_road_bias, 3),
        "egress_offroad_bias": round(egress_offroad_bias, 3),
        "nearby_concealment_osint": near_hide,
        "nearest_concealment_m": round(min_hide_m, 1) if min_hide_m is not None else None,
        "label": "concealment_osint",
        "policy": CONCEALMENT_POLICY,
    }


def attach_terrain_to_candidates(
    candidates: List[Dict[str, Any]],
    terrain: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out = []
    for c in candidates:
        item = dict(c)
        ts = terrain_score_for_candidate(item, terrain)
        item["terrain_context"] = ts
        reasons = list(item.get("reasons") or [])
        if ts.get("terrain_band"):
            reasons.append(f"terrain:{ts['terrain_band']}")
        if ts.get("nearby_concealment_osint"):
            reasons.append(
                f"concealment_osint:near={ts['nearby_concealment_osint']}"
            )
        item["reasons"] = reasons
        out.append(item)
    return out


def assess_terrain_context(
    lat: float,
    lng: float,
    radius_m: float = 800,
    concealment_radius_m: float = 900,
) -> Dict[str, Any]:
    """
    Full structured terrain context for egress + candidate scoring.
    """
    center: Coord = (lat, lng)
    landcover = fetch_landcover(center, radius_m=radius_m)
    concealment = fetch_concealment_osint(center, radius_m=concealment_radius_m)
    band = classify_terrain_band(landcover)

    return {
        "ok": bool(landcover.get("ok") or concealment.get("ok")),
        "center": {"lat": lat, "lng": lng},
        "landcover": landcover,
        "concealment_osint": concealment,
        "band": band,
        "egress_hints": {
            "prefer_road": band.get("band") in ("urban_road", "mixed"),
            "prefer_cover_routes": band.get("band") in ("forest", "rough"),
            "high_exposure": band.get("band") == "open_plain",
            "scores": band.get("scores"),
        },
        "policy": (
            "Defensive localization / egress / hide-awareness only. "
            "concealment_osint is never a strike target list."
        ),
        "source": "openstreetmap-overpass",
    }
