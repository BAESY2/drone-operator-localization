# Deployment

```bash
pip install -r requirements.txt
python -m src
```

Open http://127.0.0.1:8000

The server calls Overpass, Open-Meteo, and OpenTopoData at request time. No GeoTIFF or building dumps are shipped.

Telegram:

```bash
copy .env.example .env
python telegram/bot.py
```
