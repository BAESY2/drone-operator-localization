"""공통 HTTP 클라이언트. OSM 이용 정책상 식별 가능한 User-Agent가 필요하다."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import BASE_DIR, VERSION

USER_AGENT = (
    f"DroneOperatorLocalization/{VERSION} "
    "(open-source; https://github.com/baesy/drone-operator-localization)"
)
CACHE_DIR = BASE_DIR / ".cache" / "geo"
DEFAULT_TIMEOUT = 18


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]
    return CACHE_DIR / f"{digest}.json"


def cached_json(key: str, ttl_sec: int = 6 * 3600) -> Optional[Any]:
    path = _cache_path(key)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl_sec:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def store_json(key: str, payload: Any) -> None:
    path = _cache_path(key)
    path.write_text(json.dumps(payload), encoding="utf-8")


def get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    cache_key: Optional[str] = None,
    ttl_sec: int = 6 * 3600,
    method: str = "GET",
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    key = cache_key or f"{method}:{url}:{data or b''!r}"
    cached = cached_json(key, ttl_sec=ttl_sec)
    if cached is not None:
        return cached

    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    request = Request(url, data=data, headers=req_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"HTTP failed {url}: {exc}") from exc

    payload = json.loads(raw) if raw else {}
    store_json(key, payload)
    return payload


def get_bytes(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()
