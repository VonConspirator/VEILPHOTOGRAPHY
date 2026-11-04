"""Who may talk to Veil, decided from the address the connection really came from.

- This computer (loopback)          -> "local":   everything, including managing phones.
- Tailscale network (100.64/10)     -> "tailnet": only if the owner switched phone access on, only paired
                                                  phones, never the management routes.
- Anything else (home Wi-Fi, the open internet, a forged header)           -> refused.

Headers such as X-Forwarded-For are never trusted: only the socket's peer address counts.
"""
from __future__ import annotations

import ipaddress

TAILSCALE4 = ipaddress.ip_network("100.64.0.0/10")
TAILSCALE6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")

# Routes only the owner at this computer may use, even when a phone is paired.
LOCAL_ONLY_PREFIXES = ("/api/auth", "/api/notice", "/api/phone")


def classify(peer: str | None) -> str:
    """'local', 'tailnet' or 'other' for a connection's peer address."""
    try:
        ip = ipaddress.ip_address((peer or "").strip().split("%")[0])
    except ValueError:
        return "other"
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if ip.is_loopback:
        return "local"
    if ip in TAILSCALE4 or ip in TAILSCALE6:
        return "tailnet"
    return "other"


def is_local_only_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in LOCAL_ONLY_PREFIXES)
