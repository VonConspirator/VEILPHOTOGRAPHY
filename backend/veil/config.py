"""Settings. Everything private lives under one data folder that git ignores."""
from __future__ import annotations

import os
from pathlib import Path

PORT = int(os.environ.get("VEIL_PORT", "5195"))
MAX_UPLOAD_BYTES = 150 * 1024 * 1024
MAX_PIXELS = 120_000_000
ALLOWED_UPLOAD_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF"}
SESSION_SECONDS = 7 * 24 * 3600
LOGIN_MAX_FAILURES = 5
LOGIN_LOCKOUT_SECONDS = 300


def default_data_dir() -> Path:
    return Path(os.environ.get("VEIL_DATA_DIR") or Path(__file__).resolve().parents[2] / "data")


def allowed_hosts() -> set[str]:
    """Host names this server answers to. Anything else is refused (blocks DNS-rebinding attacks)."""
    extra = {h.strip().lower() for h in os.environ.get("VEIL_ALLOWED_HOSTS", "").split(",") if h.strip()}
    return {"localhost", "127.0.0.1", "[::1]"} | extra
