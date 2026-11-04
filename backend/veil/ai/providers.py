"""Connections to language models. Three kinds: Ollama (local), any OpenAI-compatible service
(local or paid), and Anthropic. Each one takes text and, only when told to, one small photo."""
from __future__ import annotations

import base64
import ipaddress
from urllib.parse import urlsplit

import httpx

TIMEOUT = httpx.Timeout(180.0, connect=5.0)   # the first request can wait while a model loads into memory
KINDS = ("ollama", "openai", "anthropic")
ANTHROPIC_VERSION = "2023-06-01"


class AIError(Exception):
    """A problem the user can be told about."""


def is_local_url(url: str) -> bool:
    """True only for this computer itself. A LAN or Tailscale address counts as remote."""
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _b64(jpeg: bytes) -> str:
    return base64.b64encode(jpeg).decode()


class Provider:
    kind = ""

    def __init__(self, cfg: dict):
        self.id = cfg["id"]
        self.name = cfg.get("name") or cfg["id"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg.get("model", "")
        self.api_key = cfg.get("api_key") or ""
        self.vision = bool(cfg.get("vision", False))

    @property
    def local(self) -> bool:
        return is_local_url(self.base_url)

    def _post(self, url: str, **kw) -> dict:
        try:
            r = httpx.post(url, timeout=TIMEOUT, **kw)
        except httpx.ConnectError:
            raise AIError(f"Could not reach {self.name}. Is it running?") from None
        except httpx.TimeoutException:
            raise AIError(f"{self.name} took too long to answer.") from None
        except httpx.HTTPError:
            raise AIError(f"Could not talk to {self.name}.") from None
        if r.status_code in (401, 403):
            raise AIError(f"{self.name} refused the request. Check the API key.")
        if r.status_code == 404:
            raise AIError(f"{self.name} says the model '{self.model}' is not available. Check the model name.")
        if r.status_code == 429:
            raise AIError(f"{self.name} is rate limiting. Try again in a moment.")
        if r.status_code >= 400:
            raise AIError(f"{self.name} returned an error ({r.status_code}).")
        try:
            return r.json()
        except ValueError:
            raise AIError(f"{self.name} sent back something Veil could not read.") from None

    def _get(self, url: str, **kw) -> dict:
        try:
            r = httpx.get(url, timeout=httpx.Timeout(10.0, connect=3.0), **kw)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError):
            raise AIError(f"Could not reach {self.name}.") from None

    def chat(self, system: str, user: str, image_jpeg: bytes | None = None) -> str:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return []


class OllamaProvider(Provider):
    kind = "ollama"

    def chat(self, system, user, image_jpeg=None):
        msg = {"role": "user", "content": user}
        if image_jpeg:
            msg["images"] = [_b64(image_jpeg)]
        data = self._post(f"{self.base_url}/api/chat", json={
            # No "format" option: with some models (qwen3) Ollama then leaves "content" empty and
            # writes the answer into "thinking". The parser copes with plain text, so ask plainly.
            "model": self.model, "stream": False, "think": False,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, msg]})
        m = data.get("message") or {}
        return str(m.get("content") or m.get("thinking") or "")

    def list_models(self):
        return sorted(m["name"] for m in self._get(f"{self.base_url}/api/tags").get("models", []))


class OpenAIProvider(Provider):
    kind = "openai"

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def chat(self, system, user, image_jpeg=None):
        content: object = user
        if image_jpeg:
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(image_jpeg)}}]
        data = self._post(f"{self.base_url}/chat/completions", headers=self._headers(), json={
            "model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]})
        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            raise AIError(f"{self.name} sent back something Veil could not read.") from None

    def list_models(self):
        return sorted(m["id"] for m in self._get(f"{self.base_url}/models", headers=self._headers()).get("data", []))


class AnthropicProvider(Provider):
    kind = "anthropic"

    def chat(self, system, user, image_jpeg=None):
        content: object = user
        if image_jpeg:
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(image_jpeg)}},
                       {"type": "text", "text": user}]
        data = self._post(f"{self.base_url}/v1/messages",
                          headers={"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION},
                          json={"model": self.model, "max_tokens": 600, "temperature": 0, "system": system,
                                "messages": [{"role": "user", "content": content}]})
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError):
            raise AIError(f"{self.name} sent back something Veil could not read.") from None


def build(cfg: dict) -> Provider:
    cls = {"ollama": OllamaProvider, "openai": OpenAIProvider, "anthropic": AnthropicProvider}.get(cfg.get("kind"))
    if cls is None:
        raise AIError("Unknown provider type.")
    return cls(cfg)
