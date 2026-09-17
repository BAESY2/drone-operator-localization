# Egress / Escape Route Prediction (Defensive)

**DEFENSIVE USE ONLY.** Predicts likely post-launch operator movement after a UAV event (success or failure) so defenders can continue tracking. Not for helping attackers evade.

## Pipeline position

```
OSM localize → ensemble → review → kinematics
  → tactical layers (drone approach = red/magenta)
  → egress prediction (road/off-road = amber/green/olive)
```

## Inputs

| Field | Meaning |
|-------|---------|
| `origin` | Estimated operator/launch candidate |
| `threat_axis_bearing` | Observed drone approach bearing (from observer) |
| `mission_outcome` | `success` \| `failure` \| `unknown` |
| `modes` | subset of foot / bike / motorcycle / car / armored / offroad |

## Logic (formalized)

1. **Egress axis** = `(threat_axis_bearing + 180) % 360` (away from defended sector), plus fan ±20/45/90°.
2. **Time budgets** by outcome (`data/egress_constants.json`): failure shorter/hastier; success longer/highway-biased.
3. **Mode speed** = `avg_speed_kmh * road_class_mult * (road_factor|offroad_factor)`.
4. **Reach distance** = `speed_kmh * (minutes/60)`.
5. **Road pull**: nearest OSM highway nodes/segments within radius; prefer class matching mode.
6. **Path sketch**: origin → snap to nearest suitable road point → step along preferred egress bearings to exit rings.
7. **Exit points**: ring intersections at each time budget (isochrone proxies) + high-class road nodes in egress fan.
8. **Scores**: alignment with egress axis (0.4) + road suitability (0.3) + outcome urgency fit (0.3).

## Map layers

| Layer | Color | Geometry |
|-------|-------|----------|
| Drone approach corridor | `#e05a4a` | polygon + centerline |
| Drone flight track | `#c44dff` | LineString from `flight_track` |
| Foot egress | `#6bcf6b` dashed | LineString |
| Bike | `#9bdf4a` | LineString |
| Motorcycle | `#3db8a8` | LineString |
| Car (roads) | `#d4a017` | LineString |
| Armored / offroad | `#b8860b` / `#8b6914` | LineString |
| Exit points | `#ffffff` | Point + ring |

## Constants

See `data/egress_constants.json`. Speeds are theater-average heuristics (UA/RU road mix), not GPS traces.

## API

`POST /api/predict` accepts:

```json
{
  "mission_outcome": "failure",
  "egress_modes": ["foot", "car", "motorcycle"]
}
```

Response: `tactical_layers.egress` + `tactical_layers.drone_flight`.
