"""One owner account, scrypt-hashed password, in-memory sessions, login lockout."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from pathlib import Path

from . import config

_lock = threading.Lock()


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32).hex()


class AuthStore:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "auth.json"
        self.sessions: dict[str, float] = {}
        self.failures: dict[str, list[float]] = {}

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {"user": None, "adult_ack": None}

    def _save(self, db: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(db, indent=2))
        tmp.chmod(0o600)
        tmp.replace(self.path)

    # ------------------------------------------------------------ account
    def has_user(self) -> bool:
        return self._load()["user"] is not None

    def create_owner(self, username: str, password: str) -> None:
        username = username.strip()
        if not (1 <= len(username) <= 40):
            raise ValueError("Username must be 1 to 40 characters.")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        with _lock:
            db = self._load()
            if db["user"] is not None:
                raise PermissionError("An owner account already exists.")
            salt = secrets.token_bytes(16)
            db["user"] = {"name": username, "salt": salt.hex(), "hash": _hash(password, salt)}
            self._save(db)

    def _locked_out(self, who: str) -> bool:
        now = time.time()
        recent = [t for t in self.failures.get(who, []) if now - t < config.LOGIN_LOCKOUT_SECONDS]
        self.failures[who] = recent
        return len(recent) >= config.LOGIN_MAX_FAILURES

    def login(self, username: str, password: str, who: str) -> str | None:
        """Returns a session token, or None for bad credentials. Raises PermissionError when locked out."""
        with _lock:
            if self._locked_out(who):
                raise PermissionError("Too many failed attempts. Wait a few minutes and try again.")
            user = self._load()["user"]
            ok = False
            if user is not None:
                expected = _hash(password, bytes.fromhex(user["salt"]))
                ok = hmac.compare_digest(expected, user["hash"]) and hmac.compare_digest(
                    username.strip().encode(), user["name"].encode())
            else:
                _hash(password, b"\0" * 16)  # same cost whether or not an account exists
            if not ok:
                self.failures.setdefault(who, []).append(time.time())
                return None
            self.failures.pop(who, None)
            token = secrets.token_urlsafe(32)
            self.sessions[token] = time.time()
            return token

    def session_ok(self, token: str | None) -> bool:
        if not token:
            return False
        started = self.sessions.get(token)
        if started is None:
            return False
        if time.time() - started > config.SESSION_SECONDS:
            self.sessions.pop(token, None)
            return False
        return True

    def logout(self, token: str | None) -> None:
        if token:
            self.sessions.pop(token, None)

    # ------------------------------------------------------------ adult-use notice
    def adult_acknowledged(self) -> bool:
        return self._load()["adult_ack"] is not None

    def acknowledge_adult_notice(self) -> None:
        with _lock:
            db = self._load()
            db["adult_ack"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save(db)
