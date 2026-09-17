"""python -m src → FastAPI (localhost by default)."""

import os

import uvicorn

from config import API_PORT, BIND_HOST
from src.server import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("BIND_HOST", BIND_HOST),
        port=int(os.getenv("PORT", API_PORT if API_PORT != 3000 else 8000)),
    )
