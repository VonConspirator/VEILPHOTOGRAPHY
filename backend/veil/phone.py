"""Phone pairing: one-time QR tokens, paired devices, and the owner's on/off switch.

A phone scans the QR code (a one-time token, valid 5 minutes) and becomes a *pending* device. The owner approves
it at the computer. Only a hash of every secret is stored, so a copy of this file cannot be used to get in.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from pathlib import Path

TOKEN_SECONDS = 5 * 60
PENDING_SECONDS = 15 * 60
DEVICE_SECONDS = 90 * 24 * 3600
MAX_PENDING = 5
MAX_DEVICES = 20
REDEEM_MAX_TRIES = 8
REDEEM_WINDOW = 60

_lock = threading.RLock()


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


class PairingError(Exception):
    """Something the person can be told in plain words."""


class PhoneStore:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "phone.json"
        self.tries: dict[str, list[float]] = {}
        self._seen: dict[str, float] = {}

    # ------------------------------------------------------------ storage
    def _load(self) -> dict:
        try:
            db = json.loads(self.path.read_text())
        except Exception:
            db = {}
        db.setdefault("enabled", False)
        db.setdefault("tokens", {})    # sha256(token) -> expiry time
        db.setdefault("devices", {})   # id -> device record
        return db

    def _save(self, db: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(db, indent=2))
        tmp.chmod(0o600)
        tmp.replace(self.path)

    def _tidy(self, db: dict) -> None:
        now = time.time()
        db["tokens"] = {h: t for h, t in db["tokens"].items() if t > now}
        for d in db["devices"].values():
            if d["status"] == "pending" and now - d["created"] > PENDING_SECONDS:
                d["status"] = "denied"
            if d["status"] == "approved" and now - d["approved_at"] > DEVICE_SECONDS:
                d["status"] = "revoked"

    # ------------------------------------------------------------ the owner's switch
    def enabled(self) -> bool:
        return bool(self._load()["enabled"])

    def set_enabled(self, on: bool) -> None:
        with _lock:
            db = self._load()
            db["enabled"] = bool(on)
            if not on:
                db["tokens"] = {}  # any QR code still on screen stops working
            self._save(db)

    # ------------------------------------------------------------ QR tokens
    def new_token(self) -> tuple[str, int]:
        with _lock:
            db = self._load()
            self._tidy(db)
            if not db["enabled"]:
                raise PairingError("Switch phone access on first.")
            if sum(1 for d in db["devices"].values() if d["status"] == "pending") >= MAX_PENDING:
                raise PairingError("Too many phones are waiting. Approve or remove some first.")
            if sum(1 for d in db["devices"].values() if d["status"] == "approved") >= MAX_DEVICES:
                raise PairingError("That is the most phones Veil will pair. Remove one first.")
            token = secrets.token_urlsafe(32)
            db["tokens"][_hash(token)] = time.time() + TOKEN_SECONDS
            self._save(db)
            return token, TOKEN_SECONDS

    def redeem(self, token: str, name: str, who: str) -> tuple[str, dict]:
        """Turns a QR token into a pending device. Returns (device secret for the phone's cookie, device)."""
        with _lock:
            now = time.time()
            recent = [t for t in self.tries.get(who, []) if now - t < REDEEM_WINDOW]
            self.tries[who] = recent
            if len(recent) >= REDEEM_MAX_TRIES:
                raise PairingError("Too many tries. Wait a minute and scan a fresh QR code.")
            recent.append(now)
            db = self._load()
            self._tidy(db)
            if not db["enabled"]:
                raise PairingError("Phone access is switched off on the computer.")
            key = _hash(token or "")
            if db["tokens"].pop(key, None) is None:  # single use: removing it also uses it up
                self._save(db)
                raise PairingError("That QR code has expired or was already used. Make a new one on the computer.")
            if sum(1 for d in db["devices"].values() if d["status"] == "pending") >= MAX_PENDING:
                self._save(db)
                raise PairingError("Too many phones are waiting for approval.")
            secret = secrets.token_urlsafe(32)
            dev_id = secrets.token_hex(6)
            db["devices"][dev_id] = {
                "id": dev_id, "name": (name or "Phone").strip()[:60] or "Phone", "status": "pending",
                "secret_hash": _hash(secret), "created": now, "approved_at": 0, "last_seen": now,
            }
            self._save(db)
            return secret, self._public(db["devices"][dev_id])

    # ------------------------------------------------------------ devices
    @staticmethod
    def _public(d: dict) -> dict:
        return {k: d[k] for k in ("id", "name", "status", "created", "approved_at", "last_seen")}

    def device_for(self, secret: str | None) -> dict | None:
        """The device a cookie belongs to (any status), or None."""
        if not secret:
            return None
        key = _hash(secret)
        db = self._load()
        self._tidy(db)
        for d in db["devices"].values():
            if secrets.compare_digest(d["secret_hash"], key):
                return d
        return None

    def device_ok(self, secret: str | None) -> bool:
        """True only for an approved device, while phone access is on."""
        d = self.device_for(secret)
        if not d or d["status"] != "approved":
            return False
        if not self.enabled():
            return False
        self._touch(d["id"])
        return True

    def _touch(self, dev_id: str) -> None:
        now = time.time()
        if now - self._seen.get(dev_id, 0) < 60:
            return
        self._seen[dev_id] = now
        with _lock:
            db = self._load()
            if dev_id in db["devices"]:
                db["devices"][dev_id]["last_seen"] = now
                self._save(db)

    def state_for(self, secret: str | None) -> str:
        d = self.device_for(secret)
        return d["status"] if d else "none"

    def list(self) -> list[dict]:
        with _lock:
            db = self._load()
            self._tidy(db)
            self._save(db)
            return sorted((self._public(d) for d in db["devices"].values()), key=lambda d: -d["created"])

    def _set(self, dev_id: str, allowed_from: set[str], new: str) -> None:
        with _lock:
            db = self._load()
            self._tidy(db)
            d = db["devices"].get(dev_id)
            if not d:
                raise PairingError("That phone is not in the list.")
            if d["status"] not in allowed_from:
                raise PairingError(f"That phone is already {d['status']}.")
            d["status"] = new
            if new == "approved":
                d["approved_at"] = time.time()
            self._save(db)

    def approve(self, dev_id: str) -> None:
        self._set(dev_id, {"pending"}, "approved")

    def deny(self, dev_id: str) -> None:
        self._set(dev_id, {"pending"}, "denied")

    def revoke(self, dev_id: str) -> None:
        self._set(dev_id, {"approved"}, "revoked")

    def forget(self, dev_id: str) -> None:
        """Removes a device from the list for good (after it was denied or revoked, or when the owner clears it)."""
        with _lock:
            db = self._load()
            if db["devices"].pop(dev_id, None) is None:
                raise PairingError("That phone is not in the list.")
            self._save(db)
