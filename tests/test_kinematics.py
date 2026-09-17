"""BGU-inspired kinematics features."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kinematics import extract_kinematics_features, kinematics_candidate_weight


def _sample_track():
    # eastbound with climb then level — 12 points
    track = []
    for i in range(12):
        track.append(
            {
                "t": float(i),
                "lat": 37.5665 + 0.00005 * i,
                "lng": 126.9780 + 0.0004 * i,
                "alt_m": 40 + min(i, 6) * 3.0,
                "yaw_deg": 90 + (2 if i % 3 == 0 else 0),
            }
        )
    return track


def test_extract_features():
    feat = extract_kinematics_features(_sample_track())
    assert feat["usable"] is True
    assert feat["sample_count"] == 12
    assert feat["confidence"] >= 0.4
    assert feat["pilot_mode"] in {"los", "fpv", "unknown"}
    assert "climb_mps" in feat
    assert "aggression_index" in feat


def test_candidate_weight():
    feat = extract_kinematics_features(_sample_track())
    w = kinematics_candidate_weight(
        {"lat": 37.5660, "lng": 126.9700, "elevation_m": 50},
        feat,
    )
    assert 0.05 <= w <= 1.5
