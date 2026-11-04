"""Run Veil:  python -m veil

Listens on this computer (127.0.0.1) and, when Tailscale is running, on the Tailscale address too, so a paired
phone can reach it. It never listens on your Wi-Fi or on every address.
"""
import socket

import uvicorn

from . import config
from .tailscale import Tailscale


def _listener(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.set_inheritable(True)
    return s


def main() -> None:
    hosts = ["127.0.0.1"]
    info = Tailscale().info()
    if info.running and info.ipv4:
        hosts.append(info.ipv4)
    config.LISTENING = hosts
    sockets = [_listener(h, config.PORT) for h in hosts]
    print(f"Veil is running.  On this computer:  http://localhost:{config.PORT}", flush=True)
    if len(hosts) > 1:
        print(f"Phones on your Tailscale network can reach it at http://{info.ipv4}:{config.PORT} once paired.", flush=True)
    else:
        print("Phone access is unavailable: " + (info.problem or "Tailscale is not running") +
              " Start Tailscale, then restart Veil.", flush=True)
    server = uvicorn.Server(uvicorn.Config("veil.main:app_factory", factory=True, log_level="warning",
                                           proxy_headers=False))  # never trust X-Forwarded-For
    server.run(sockets=sockets)


if __name__ == "__main__":
    main()
