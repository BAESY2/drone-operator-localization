"""Validate required project files exist."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
required = [
    ROOT / "config.py",
    ROOT / "requirements.txt",
    ROOT / "data" / "drone_specs.json",
    ROOT / "src" / "localize.py",
    ROOT / "src" / "predict_api.py",
]

ok = True
for path in required:
    if not path.exists():
        print(f"MISSING: {path}")
        ok = False
    else:
        print(f"OK: {path.relative_to(ROOT)}")

specs = ROOT / "data" / "drone_specs.json"
if specs.exists():
    data = json.loads(specs.read_text(encoding="utf-8"))
    print(f"drone models: {len(data)}")

sys.exit(0 if ok else 1)
