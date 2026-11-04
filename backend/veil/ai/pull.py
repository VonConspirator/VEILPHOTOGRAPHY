"""Downloads a model into Ollama in the background so the page can show progress.
Only catalogue models, only when the owner clicks, one at a time."""
from __future__ import annotations

import json
import threading

import httpx

from .catalog import DOWNLOADABLE
from .providers import AIError


class Puller:
    def __init__(self):
        self._lock = threading.Lock()
        self.state: dict = {"running": False, "name": "", "status": "", "done": 0, "total": 0, "error": ""}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def start(self, base_url: str, name: str) -> None:
        if name not in DOWNLOADABLE:
            raise AIError("Only the models listed in Veil can be downloaded here.")
        with self._lock:
            if self.state["running"]:
                raise AIError("A download is already running.")
            self.state = {"running": True, "name": name, "status": "starting", "done": 0, "total": 0, "error": ""}
        threading.Thread(target=self._run, args=(base_url, name), daemon=True).start()

    def _set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    def _run(self, base_url: str, name: str) -> None:
        try:
            with httpx.stream("POST", f"{base_url}/api/pull", json={"model": name, "stream": True},
                              timeout=httpx.Timeout(None, connect=5.0)) as r:
                if r.status_code >= 400:
                    raise AIError(f"Ollama refused the download ({r.status_code}).")
                for line in r.iter_lines():
                    if not line:
                        continue
                    ev = json.loads(line)
                    if ev.get("error"):
                        raise AIError("Ollama could not download that model.")
                    self._set(status=str(ev.get("status", ""))[:60], done=int(ev.get("completed") or 0),
                              total=int(ev.get("total") or 0))
            self._set(running=False, status="done")
        except AIError as e:
            self._set(running=False, error=str(e))
        except (httpx.HTTPError, ValueError):
            self._set(running=False, error="The download failed. Is Ollama running?")
