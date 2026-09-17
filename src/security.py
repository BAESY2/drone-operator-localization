"""API security: key auth, rate limit, headers, theater gate, safe errors."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Callable, Optional

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config import (
    API_KEY,
    ALLOW_INSECURE_DEV,
    ENABLE_OPENAPI,
    RATE_LIMIT_PER_MIN,
    SECURITY_MODE,
    THEATER_BOUNDS,
    THEATER_ENFORCE,
)


PUBLIC_PATHS = {"/", "/api/health", "/favicon.ico"}
PUBLIC_PREFIXES = ("/static/",)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _is_localhost(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def require_api_key(request: Request) -> None:
    """Enforce API key unless insecure local-dev override is on."""
    path = request.url.path
    if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return
    if path in {"/docs", "/redoc", "/openapi.json"} and not ENABLE_OPENAPI:
        raise HTTPException(status_code=404, detail="Not found")

    if not API_KEY:
        if ALLOW_INSECURE_DEV and _is_localhost(_client_ip(request)):
            return
        raise HTTPException(
            status_code=503,
            detail="API_KEY not configured. Set API_KEY in .env",
        )

    provided = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if not provided or not hmac.compare_digest(provided, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def in_theater(lat: float, lng: float) -> bool:
    b = THEATER_BOUNDS
    return b["lat_min"] <= lat <= b["lat_max"] and b["lng_min"] <= lng <= b["lng_max"]


def assert_theater(lat: float, lng: float, label: str = "point") -> None:
    if not THEATER_ENFORCE:
        return
    if not in_theater(lat, lng):
        raise HTTPException(
            status_code=403,
            detail=f"{label} outside allowed region",
        )


def safe_error_detail(exc: Exception) -> str:
    if SECURITY_MODE == "strict":
        return "Upstream or processing error"
    return str(exc)[:300]


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_min: int = RATE_LIMIT_PER_MIN):
        super().__init__(app)
        self.limit = max(1, limit_per_min)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def _allow(self, key: str) -> bool:
        now = time.monotonic()
        window = 60.0
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if path == "/api/health":
            return await call_next(request)
        ip = _client_ip(request)
        bucket = f"{ip}:{path.split('/')[1:3]}"  # e.g. api:predict
        if not self._allow(bucket):
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request.state.request_id = secrets.token_hex(8)
        try:
            require_api_key(request)
        except HTTPException as exc:
            return Response(
                content=f'{{"detail":"{exc.detail}"}}',
                status_code=exc.status_code,
                media_type="application/json",
                headers={"X-Request-Id": request.state.request_id},
            )
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(self), camera=(), microphone=()"
        response.headers["Cache-Control"] = "no-store"
        if path_is_html(request.url.path):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com data:; "
                "img-src 'self' data: https: blob:; "
                "connect-src 'self' https:; "
                "frame-ancestors 'none'"
            )
        return response


def path_is_html(path: str) -> bool:
    return path == "/" or path.endswith(".html")


def fingerprint_key(key: Optional[str]) -> str:
    if not key:
        return "none"
    return hashlib.sha256(key.encode()).hexdigest()[:12]
