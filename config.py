# config.py
"""
Drone Operator Localization - 공통 설정
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
TESTS_DIR = BASE_DIR / "tests"

try:
    with open(DATA_DIR / "drone_specs.json", "r", encoding="utf-8") as f:
        DRONE_SPECS = json.load(f)
except FileNotFoundError:
    print(f"Warning: drone_specs.json not found at {DATA_DIR}")
    DRONE_SPECS = {}

DEFAULT_BEARING_TOLERANCE = 18
DEFAULT_RANGE_MIN_KM = 0.25
DEFAULT_RANGE_MAX_KM = 50

SCORING_WEIGHTS = {
    "elevation": 0.15,
    "los": 0.15,
    "military_history": 0.25,
    "tactical": 0.15,
    "road": 0.1,
    "building_type": 0.1,
    "cell_tower": 0.1,
}

assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 0.01, "Weights must sum to 1.0"

API_TIMEOUT = 10
API_MAX_CANDIDATES = 48
API_SHORTLIST = 18
API_FORMULA_POOL = 36
API_PORT = int(os.getenv("PORT", 3000))

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_NAMESPACE_ID = os.getenv("CF_NAMESPACE_ID")
CF_API_TOKEN = os.getenv("CF_API_TOKEN")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = os.getenv("TELEGRAM_API_URL", "http://127.0.0.1:8000/api/predict")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

DRONE_SPECS_PATH = DATA_DIR / "drone_specs.json"

PERFORMANCE_TARGETS = {
    "response_time_ms": 500,
    "accuracy_meters": 500,
    "accuracy_triangulation_meters": 100,
}

VERSION = "2.0.0"
PROJECT_NAME = "Drone Operator Localization"
GITHUB_REPO = "https://github.com/baesy/drone-operator-localization"
LICENSE = "MIT"

# --- Optional geo gate + security ---
REGION = os.getenv("REGION", "global")
REGION_BOUNDS = {
    "lat_min": float(os.getenv("REGION_LAT_MIN", "-90")),
    "lat_max": float(os.getenv("REGION_LAT_MAX", "90")),
    "lng_min": float(os.getenv("REGION_LNG_MIN", "-180")),
    "lng_max": float(os.getenv("REGION_LNG_MAX", "180")),
}
REGION_ENFORCE = os.getenv("REGION_ENFORCE", "false").lower() in ("1", "true", "yes")
# backwards-compatible aliases
THEATER = REGION
THEATER_BOUNDS = REGION_BOUNDS
THEATER_ENFORCE = REGION_ENFORCE
API_KEY = os.getenv("API_KEY", "").strip()
ALLOW_INSECURE_DEV = os.getenv("ALLOW_INSECURE_DEV", "true").lower() in ("1", "true", "yes")
SECURITY_MODE = os.getenv("SECURITY_MODE", "strict")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
ENABLE_OPENAPI = os.getenv("ENABLE_OPENAPI", "false").lower() in ("1", "true", "yes")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if o.strip()
]
BIND_HOST = os.getenv("BIND_HOST", "127.0.0.1")

LEGAL_NOTICE = """
LEGAL NOTICE

This tool is provided for DEFENSIVE purposes only.

PERMITTED USES:
- Military self-defense against drone attacks
- Civilian protection and emergency response

PROHIBITED USES:
- Offensive military operations
- Civilian targeting
- Commercial use without explicit authorization

Users assume FULL LEGAL LIABILITY for tool usage.
Developer assumes NO responsibility for misuse.
"""

if __name__ == "__main__":
    print(f"Configuration loaded from {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Version: {VERSION}")
    print(LEGAL_NOTICE)
