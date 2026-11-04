"""Who may use this worker. A one-time pairing code, made at this machine, turns into a long secret token for the
other computer. Only hashes are kept: a copy of this file cannot be used to get in."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from pathlib import Path

CODE_SECONDS = 10 * 60
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"        # no 0/O, 1/I/L: easy to read out and type
CODE_LEN = 10
MAX_CLIENTS = 5
MAX_BAD_CODES = 5                                         # wrong guesses before every pending code is burned
PAIR_WINDOW = 60
PAIR_MAX_PER_WINDOW = 6

_lock = threading.RLock()


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def normalise_code(text: str) -> str:
    return "".join(c for c in (text or "").upper() if c.isalnum())


def pretty_code(code: str) -> str:
    return f"{code[:5]}-{code[5:]}"


class PairError(Exception):
    """Something the person can be told in plain words."""


class WorkerStore:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "worker-server.json"
        self._seen: dict[str, float] = {}
        self._tries: list[float] = []

    def _load(self) -> dict:
        try:
            db = json.loads(self.path.read_text())
        except Exception:
            db = {}
        db.setdefault("worker_id", "")
        db.setdefault("codes", {})       # sha256(code) -> expiry time
        db.setdefault("bad", 0)
        db.setdefault("clients", {})     # id -> {name, hash, created, last_seen}
        return db

    def _save(self, db: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(db, indent=2))
        tmp.chmod(0o600)
        tmp.replace(self.path)

    def worker_id(self) -> str:
        with _lock:
            db = self._load()
            if not db["worker_id"]:
                db["worker_id"] = secrets.token_hex(8)
                self._save(db)
            return db["worker_id"]

    # ------------------------------------------------------------ pairing (the code is made at THIS machine, never over the network)
    def new_code(self) -> tuple[str, int]:
        with _lock:
            db = self._load()
            now = time.time()
            db["codes"] = {h: t for h, t in db["codes"].items() if t > now}
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))
            db["codes"][_hash(code)] = now + CODE_SECONDS
            db["bad"] = 0
            self._save(db)
            return code, CODE_SECONDS

    def redeem(self, code: str, name: str) -> tuple[str, str]:
        """Trades a valid one-time code for (client id, token). The code works once."""
        with _lock:
            now = time.time()
            self._tries = [t for t in self._tries if now - t < PAIR_WINDOW]
            if len(self._tries) >= PAIR_MAX_PER_WINDOW:
                raise PairError("Too many tries. Wait a minute, then try again.")
            self._tries.append(now)
            db = self._load()
            db["codes"] = {h: t for h, t in db["codes"].items() if t > now}
            h = _hash(normalise_code(code))
            ok = any(hmac.compare_digest(h, known) for known in db["codes"])
            if not ok:
                db["bad"] += 1
                if db["bad"] >= MAX_BAD_CODES:
                    db["codes"] = {}                 # someone is guessing: burn every pending code
                    db["bad"] = 0
                self._save(db)
                raise PairError("That code is not right, or it has expired. Make a new one on the worker with: veil-worker pair")
            if len(db["clients"]) >= MAX_CLIENTS:
                raise PairError(f"This worker already has {MAX_CLIENTS} paired computers. Remove one first (veil-worker clients).")
            db["codes"] = {}                         # one code, one use
            db["bad"] = 0
            token = "vwk_" + secrets.token_urlsafe(32)
            cid = secrets.token_hex(4)
            db["clients"][cid] = {"name": (name or "Computer").strip()[:40] or "Computer", "hash": _hash(token),
                                  "created": now, "last_seen": now}
            self._save(db)
            return cid, token

    # ------------------------------------------------------------ checking a request
    def check(self, token: str) -> str | None:
        """The client id for a valid token, else None."""
        if not token or len(token) > 200:
            return None
        h = _hash(token)
        with _lock:
            db = self._load()
            for cid, c in db["clients"].items():
                if hmac.compare_digest(c["hash"], h):
                    now = time.time()
                    if now - self._seen.get(cid, 0) > 60:        # do not write the file on every request
                        self._seen[cid] = now
                        c["last_seen"] = now
                        self._save(db)
                    return cid
        return None

    def clients(self) -> list[dict]:
        with _lock:
            return [{"id": cid, "name": c["name"], "created": c["created"], "last_seen": c["last_seen"]}
                    for cid, c in self._load()["clients"].items()]

    def revoke(self, cid: str) -> bool:
        with _lock:
            db = self._load()
            if cid not in db["clients"]:
                return False
            del db["clients"][cid]
            self._save(db)
            return True
