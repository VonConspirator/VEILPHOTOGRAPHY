"""Run the worker:   python -m veil.worker [serve | pair | clients | revoke ID | status]

It listens on this machine (127.0.0.1) and on the Tailscale address, never on the local network or on every address.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

from .. import config
from ..tailscale import Tailscale
from . import DEFAULT_PORT
from .store import WorkerStore, pretty_code

PORT = int(__import__("os").environ.get("VEIL_WORKER_PORT", DEFAULT_PORT))


def _listener(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(family, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.set_inheritable(True)
    return s


USERSPACE_HELP = """
  Tailscale is running here in "userspace" mode (common on rented GPU machines inside containers), so this worker
  cannot listen on the Tailscale address directly. Make Tailscale forward to it with this one command, once:

      tailscale serve --bg --tcp {port} tcp://localhost:{port}

  Then your own computer can reach this worker at  {name}:{port}  (or the Tailscale address shown by: tailscale ip -4).
"""


def serve(args) -> int:
    from .server import create_app
    import uvicorn
    ts = Tailscale()
    info = ts.info()
    if not info.running and not args.loopback_only:
        print("The worker will not start: Tailscale is not running here.\n"
              f"  {info.problem}\n"
              "  Sign in first (sudo tailscale up), then start the worker again.\n"
              "  (Only for testing on one machine: --loopback-only.)", flush=True)
        return 1
    hosts, userspace = ["127.0.0.1"], False
    sockets = []
    for h in hosts:
        sockets.append(_listener(h, PORT))
    if info.running and info.ipv4:
        try:
            sockets.append(_listener(info.ipv4, PORT)); hosts.append(info.ipv4)
        except OSError:
            userspace = True
    config.LISTENING = hosts
    store = WorkerStore(config.default_data_dir())
    print(f"Veil worker is running (id {store.worker_id()}).", flush=True)
    if len(hosts) > 1:
        print(f"  Your other computer can reach it at  {info.ipv4}:{PORT}" + (f"  or  {info.dns_name}:{PORT}" if info.dns_name else ""), flush=True)
    elif userspace:
        print(USERSPACE_HELP.format(port=PORT, name=info.dns_name or info.ipv4), flush=True)
    else:
        print("  Listening on this machine only (testing mode).", flush=True)
    clients = store.clients()
    print(f"  {len(clients)} computer(s) paired. To pair another, run in a second terminal:  veil-worker pair", flush=True)
    app = create_app(config.default_data_dir(), store=store)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", proxy_headers=False))
    server.run(sockets=sockets)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="veil-worker", description="Veil GPU worker")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the worker (the default)")
    s.add_argument("--loopback-only", action="store_true", help="testing only: listen on this machine only")
    sub.add_parser("pair", help="make a one-time pairing code (valid 10 minutes)")
    sub.add_parser("clients", help="list the computers paired with this worker")
    r = sub.add_parser("revoke", help="stop trusting a paired computer")
    r.add_argument("id")
    args = p.parse_args(argv)
    cmd = args.cmd or "serve"
    if cmd == "serve":
        if not hasattr(args, "loopback_only"):
            args.loopback_only = False
        return serve(args)
    store = WorkerStore(config.default_data_dir())
    if cmd == "pair":
        code, secs = store.new_code()
        print(f"\n  Pairing code:   {pretty_code(code)}\n\n  Type it into Veil on your own computer (Worker, then Pair) within {secs // 60} minutes.\n"
              "  It works once. If anyone guesses wrong five times, all pending codes are cancelled.\n")
        return 0
    if cmd == "clients":
        rows = store.clients()
        if not rows:
            print("No computers are paired with this worker.")
        for c in rows:
            seen = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["last_seen"]))
            print(f"  {c['id']}  {c['name']:<24} last used {seen}")
        return 0
    if cmd == "revoke":
        ok = store.revoke(args.id)
        print("Removed." if ok else "No paired computer has that id (see: veil-worker clients).")
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
