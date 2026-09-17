"""
AI assist for recon — OPTIONAL enrichment only.

Geometry / waypoints are ALWAYS computed in src/recon.py (deterministic).
AI may only return text notes, altitude bias, priority labels, warnings.
Supports per-request API key from local UI (never logged).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def _resolve_creds(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    force: bool = False,
) -> Optional[Dict[str, str]]:
    key = (api_key or os.getenv("AI_API_KEY", "")).strip()
    if not key:
        return None
    if not force and not api_key:
        if os.getenv("AI_RECON_ENABLE", "false").lower() not in ("1", "true", "yes"):
            return None
    return {
        "key": key,
        "base": (base_url or os.getenv("AI_BASE_URL", "https://api.openai.com/v1")).rstrip("/"),
        "model": model or os.getenv("AI_MODEL", "gpt-4o-mini"),
    }


def ai_enabled() -> bool:
    return _resolve_creds() is not None


def chat_json(
    system: str,
    user: Any,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    creds = _resolve_creds(api_key, base_url, model, force=bool(api_key))
    if not creds:
        return None
    payload = {
        "model": creds["model"],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user if isinstance(user, str) else json.dumps(user, ensure_ascii=False),
            },
        ],
    }
    req = urllib.request.Request(
        f"{creds['base']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {creds['key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError):
        return None


def enrich_recon_plan(
    plan: Dict[str, Any],
    lang: str = "en",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    out = dict(plan)
    creds = _resolve_creds(api_key, base_url, model, force=force or bool(api_key))
    if not creds:
        out["ai"] = {"applied": False, "reason": "no_ai_key"}
        return out

    summary = {
        "profile": plan.get("profile"),
        "waypoint_count": len(plan.get("waypoints") or []),
        "length_m": plan.get("length_m"),
        "settings": plan.get("settings"),
        "route_kinds": list({w.get("kind") for w in (plan.get("waypoints") or [])}),
    }
    parsed = chat_json(
        system=(
            "You assist DEFENSIVE ISR recon planning only. "
            "Never invent new lat/lng. Never suggest weapons or strike. "
            "Return strict JSON with keys: "
            "mission_notes (string), altitude_bias_m (number -30..50), "
            "route_priority (array of strings), warnings (array of strings). "
            f"Language for notes: {lang}."
        ),
        user="Plan summary:\n" + json.dumps(summary, ensure_ascii=False),
        api_key=creds["key"],
        base_url=creds["base"],
        model=creds["model"],
    )
    if not parsed:
        out["ai"] = {"applied": False, "provider": creds["base"], "reason": "request_failed"}
        return out

    bias = float(parsed.get("altitude_bias_m") or 0)
    bias = max(-30.0, min(50.0, bias))
    settings = dict(out.get("settings") or {})
    if bias:
        settings["alt_m"] = int(settings.get("alt_m", 120) + bias)
        for wp in out.get("waypoints") or []:
            wp["alt_m"] = int(wp.get("alt_m", settings["alt_m"]) + bias)
    out["settings"] = settings
    out["ai"] = {
        "applied": True,
        "provider": creds["base"],
        "model": creds["model"],
        "mission_notes": str(parsed.get("mission_notes") or "")[:800],
        "altitude_bias_m": bias,
        "route_priority": list(parsed.get("route_priority") or [])[:8],
        "warnings": list(parsed.get("warnings") or [])[:8],
    }
    return out
