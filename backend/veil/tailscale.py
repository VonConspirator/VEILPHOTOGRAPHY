"""Finds this computer's Tailscale address. Tailscale is the private network that carries phone access."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TailscaleInfo:
    running: bool = False
    ipv4: str | None = None
    ipv6: str | None = None
    dns_name: str | None = None      # e.g. mycomputer.tailnet-name.ts.net
    host_name: str | None = None     # e.g. mycomputer
    problem: str | None = None       # plain-English reason when it is not usable
    hosts: tuple[str, ...] = field(default=(), compare=False)

    def names(self) -> set[str]:
        """Host names and addresses a phone may use in the address bar."""
        return {n.lower() for n in (self.ipv4, self.dns_name, self.host_name) if n}


class Tailscale:
    """Asks the `tailscale` program (cached for a few seconds, so it is cheap to call on every request)."""

    def __init__(self, ttl: float = 15.0):
        self.ttl = ttl
        self._cache: tuple[float, TailscaleInfo] | None = None

    def info(self) -> TailscaleInfo:
        now = time.monotonic()
        if self._cache and now - self._cache[0] < self.ttl:
            return self._cache[1]
        info = self._read()
        self._cache = (now, info)
        return info

    @staticmethod
    def _read() -> TailscaleInfo:
        try:
            out = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=4)
        except FileNotFoundError:
            return TailscaleInfo(problem="Tailscale is not installed on this computer.")
        except (subprocess.TimeoutExpired, OSError):
            return TailscaleInfo(problem="Tailscale did not answer.")
        if out.returncode != 0:
            return TailscaleInfo(problem="Tailscale is installed but not running or not signed in.")
        try:
            data = json.loads(out.stdout)
            me = data.get("Self") or {}
            ips = me.get("TailscaleIPs") or []
        except (ValueError, AttributeError):
            return TailscaleInfo(problem="Tailscale gave an answer Veil could not read.")
        v4 = next((i for i in ips if ":" not in i), None)
        v6 = next((i for i in ips if ":" in i), None)
        if data.get("BackendState") != "Running" or not v4:
            return TailscaleInfo(problem="Tailscale is not connected. Open Tailscale and sign in.")
        return TailscaleInfo(running=True, ipv4=v4, ipv6=v6, dns_name=(me.get("DNSName") or "").rstrip(".") or None,
                             host_name=me.get("HostName") or None)
