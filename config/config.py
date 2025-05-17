"""Centralised config loader (env-first, JSON fallback)."""
from pathlib import Path
import json
import os
from dotenv import load_dotenv
load_dotenv()


def load(path: str | None = None) -> dict:
    cfg = {}

    # ── 1. from .env  ───────────────────────────────
    for key in ["RDS_HOST", "RDS_PORT", "RDS_USER", "RDS_PW", "RDS_DB",
                "QWEN_API_KEY"]:
        if os.getenv(key):
            cfg[key] = os.getenv(key)

    # ── 2. optional JSON file  ──────────────────────
    if path and Path(path).exists():
        cfg.update(json.loads(Path(path).read_text()))

    # ── 3. re-shape to old structure  ───────────────
    return {
        "rds": {
            "host": cfg.get("RDS_HOST"),
            "port": int(cfg.get("RDS_PORT", 5432)),
            "user": cfg.get("RDS_USER"),
            "password": cfg.get("RDS_PW"),
            "database": cfg.get("RDS_DB")
        },
        "qwen": {
            "api_key": cfg.get("QWEN_API_KEY")
        }
    }
