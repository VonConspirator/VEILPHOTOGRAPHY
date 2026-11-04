"""Which model Veil talks to. Stored in data/ai.json (owner-only file, git-ignored).
API keys live only there and are never sent back to the page."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from .providers import KINDS, AIError, is_local_url

DEFAULT = {"id": "local-ollama", "name": "Ollama (this computer)", "kind": "ollama",
           "base_url": "http://127.0.0.1:11434", "model": "qwen3.5:9b", "vision": True}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,31}$")
MAX_PROVIDERS = 12


class AISettings:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "ai.json"
        self._lock = threading.RLock()

    def _load(self) -> dict:
        try:
            db = json.loads(self.path.read_text())
            if isinstance(db.get("providers"), list) and db["providers"]:
                return db
        except (OSError, ValueError):
            pass
        return {"active": DEFAULT["id"], "providers": [dict(DEFAULT)]}

    def _save(self, db: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(db, f)
        tmp.replace(self.path)
        os.chmod(self.path, 0o600)

    # ---- reading
    def active(self) -> dict:
        db = self._load()
        for p in db["providers"]:
            if p["id"] == db["active"]:
                return p
        return db["providers"][0]

    def get(self, provider_id: str) -> dict | None:
        return next((p for p in self._load()["providers"] if p["id"] == provider_id), None)

    def public(self) -> dict:
        """What the page may see: no API keys, just whether one is saved."""
        db = self._load()
        out = []
        for p in db["providers"]:
            q = {k: v for k, v in p.items() if k != "api_key"}
            q["has_key"] = bool(p.get("api_key"))
            q["local"] = is_local_url(p["base_url"])
            out.append(q)
        return {"active": self.active()["id"], "providers": out}

    # ---- changing
    def save_provider(self, cfg: dict) -> None:
        pid = str(cfg.get("id", "")).strip().lower()
        if not ID_RE.match(pid):
            raise AIError("The id must be 1-32 lowercase letters, digits or dashes.")
        kind = cfg.get("kind")
        if kind not in KINDS:
            raise AIError("Type must be ollama, openai or anthropic.")
        base = str(cfg.get("base_url", "")).strip().rstrip("/")
        if not base.lower().startswith(("http://", "https://")):
            raise AIError("The address must start with http:// or https://.")
        model = str(cfg.get("model", "")).strip()[:120]
        if not model:
            raise AIError("Please name the model.")
        with self._lock:
            db = self._load()
            old = next((p for p in db["providers"] if p["id"] == pid), None)
            key = cfg.get("api_key")
            key = (old or {}).get("api_key", "") if key in (None, "") else str(key).strip()
            if key and not base.lower().startswith("https://") and not is_local_url(base):
                raise AIError("An API key may only be sent to an https:// address (or this computer).")
            new = {"id": pid, "name": str(cfg.get("name") or pid)[:60], "kind": kind, "base_url": base,
                   "model": model, "vision": bool(cfg.get("vision", False))}
            if key:
                new["api_key"] = key
            if old:
                db["providers"] = [new if p["id"] == pid else p for p in db["providers"]]
            else:
                if len(db["providers"]) >= MAX_PROVIDERS:
                    raise AIError("Too many providers. Remove one first.")
                db["providers"].append(new)
            self._save(db)

    def set_active(self, provider_id: str) -> None:
        with self._lock:
            db = self._load()
            if not any(p["id"] == provider_id for p in db["providers"]):
                raise AIError("That provider does not exist.")
            db["active"] = provider_id
            self._save(db)

    def remove(self, provider_id: str) -> None:
        with self._lock:
            db = self._load()
            rest = [p for p in db["providers"] if p["id"] != provider_id]
            if len(rest) == len(db["providers"]):
                raise AIError("That provider does not exist.")
            if not rest:
                raise AIError("Keep at least one provider.")
            db["providers"] = rest
            if db["active"] == provider_id:
                db["active"] = rest[0]["id"]
            self._save(db)
