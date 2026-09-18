# Drone Operator Localization (DOL)

Defensive C2 console: **bearing + GPS** → live OSM buildings, elevation, weather, road-aware egress hypotheses, optional **your AI API** to narrow the shortlist.

**Repo:** https://github.com/BAESY2/drone-operator-localization

## Legal

**DEFENSIVE USE ONLY.** See [LEGAL.md](LEGAL.md). Users assume full liability. Outputs are probabilistic — not ground truth.

## Quick start (local)

```bash
git clone https://github.com/BAESY2/drone-operator-localization.git
cd drone-operator-localization
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
python -m src
# → http://127.0.0.1:8000
```

### `.env` (server)

| Variable | Purpose |
|----------|---------|
| `API_KEY` | Optional server gate (`X-API-Key`). With `ALLOW_INSECURE_DEV=true`, localhost can omit it. |
| `AI_API_KEY` | Optional default AI key (prefer UI local key) |
| `AI_BASE_URL` | OpenAI-compatible base (default `https://api.openai.com/v1`) |
| `AI_MODEL` | e.g. `gpt-4o-mini` |
| `ALLOW_INSECURE_DEV` | `true` for local testing without API key |

### UI local keys (browser)

Open the **API / AI keys** panel:

1. **Server API Key** — if you set `API_KEY` in `.env`
2. **AI API Key + Base URL + Model** — your OpenAI-compatible key
3. Check **AI narrow shortlist** (or just paste a key) → FIX uses AI only to **rerank/drop** ids from the formula pool (**never invents coordinates**)

Keys stay in `localStorage` on your machine.

## Live geo APIs (no key)

Overpass OSM · Open-Meteo elevation/weather · Esri/NASA/OSM tiles · Nominatim

## Accuracy model (v2)

Scores prioritize **AoA fit**, **RSSI↔range consistency**, **LOS**, **mid-elevation** (not tallest roof), road egress access. Confidence radius blends geometric `d·tan(σ_θ)/√n` with verify/human priors. See `src/accuracy_model.py` and `data/research_constants.json`.

## Endpoints

- `POST /api/predict` — localize (+ optional `use_ai`, `ai_api_key`, `flight_track`)
- `POST /api/analyze` — defensive brief + recon
- `POST /api/recon/plan` — recon route
- `GET /api/geocode` · `GET /api/reverse` · `POST /api/measure` · `GET /api/health`

## License

MIT — see [LICENSE](LICENSE). Legal limits in [LEGAL.md](LEGAL.md) still apply to use.
