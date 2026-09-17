"""Telegram message formatting helpers."""

from __future__ import annotations

from typing import Any, Dict


def format_result_message(result: Dict[str, Any]) -> str:
    """Format predict API JSON into a compact Telegram message."""
    if not result.get("success"):
        return f"Error: {result.get('error', 'unknown')}"

    lines = ["Estimated operator candidates (defensive use only)", ""]

    primary = result.get("primary_target") or {}
    if primary:
        lines.append(
            f"Primary: {primary.get('probability', 0) * 100:.0f}% "
            f"@ {primary.get('latitude', primary.get('lat')):.5f}, "
            f"{primary.get('longitude', primary.get('lng')):.5f}"
        )
        lines.append("")

    lines.append("Top candidates:")
    for cand in (result.get("top_10_candidates") or [])[:5]:
        lat = cand.get("latitude", cand.get("lat"))
        lng = cand.get("longitude", cand.get("lng"))
        lines.append(
            f"  #{cand.get('rank', '?')} "
            f"{cand.get('probability', 0) * 100:.0f}% — {lat:.5f}, {lng:.5f}"
        )

    drone = result.get("drone_info") or {}
    tactical = result.get("tactical_output") or {}
    escape = tactical.get("escape_vector") or {}
    danger = tactical.get("danger_zone") or {}

    lines.extend(
        [
            "",
            f"Estimated drone: {drone.get('estimated_type', 'unknown')} "
            f"({drone.get('estimated_range_km', '?')} km)",
            f"Escape bearing: {escape.get('bearing_degrees', '?')}° "
            f"/ {escape.get('recommended_distance_m', '?')} m",
            f"Danger radius: {danger.get('radius_m', '?')} m",
        ]
    )

    meta = result.get("metadata") or {}
    if meta.get("accuracy_note"):
        lines.append(f"Note: {meta['accuracy_note']}")

    return "\n".join(lines)
