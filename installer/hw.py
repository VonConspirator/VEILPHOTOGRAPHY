"""Finds out what this computer has. Every function that talks to the system takes the command runner as an argument so the
tests can feed it fake output."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Hardware:
    arch: str = ""
    os_name: str = ""
    distro_id: str = ""
    distro_like: list[str] = field(default_factory=list)
    codename: str = ""                     # the Ubuntu/Debian release name tools such as Tailscale's repository need
    python: tuple[int, int, int] = (0, 0, 0)
    has_venv: bool = False
    has_pip: bool = False
    gpus: list[tuple[str, float]] = field(default_factory=list)      # NVIDIA only: (name, VRAM in GB)
    other_gpu: str = ""                    # e.g. "AMD Radeon ..." (found, but not supported yet)
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    has_sudo: bool = False
    is_root: bool = False
    has_apt: bool = False
    has_zstd: bool = False
    has_curl: bool = False
    ollama_path: str = ""
    ollama_running: bool = False
    ollama_models: list[dict] = field(default_factory=list)          # {"name", "size", "digest"}
    tailscale_path: str = ""
    tailscale_state: str = ""              # "", "stopped", "needs-login", "running"
    tailscale_ip: str = ""
    systemd_user: bool = False

    @property
    def vram_gb(self) -> float:
        return max((v for _, v in self.gpus), default=0.0)

    @property
    def debian_family(self) -> bool:
        return self.has_apt and (self.distro_id in ("debian", "ubuntu", "linuxmint", "pop", "elementary", "zorin") or
                                 any(x in ("debian", "ubuntu") for x in self.distro_like))


def parse_os_release(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def parse_meminfo(text: str) -> float:
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.M)
    return round(int(m.group(1)) / 1024 / 1024, 1) if m else 0.0


def parse_nvidia_smi(text: str) -> list[tuple[str, float]]:
    """`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` -> [(name, GB)]"""
    gpus = []
    for line in text.strip().splitlines():
        if "," in line:
            name, mem = line.rsplit(",", 1)
            try:
                gpus.append((name.strip(), round(float(mem.strip()) / 1024, 1)))
            except ValueError:
                pass
    return gpus


def parse_ollama_list(text: str) -> list[dict]:
    """The table printed by `ollama list`: NAME  ID  SIZE  MODIFIED."""
    rows = []
    for line in text.strip().splitlines()[1:]:
        m = re.match(r"^(\S+)\s+(\S+)\s+([\d.]+)\s*(KB|MB|GB|TB)\b", line)
        if m:
            mult = {"KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}[m.group(4)]
            rows.append({"name": m.group(1), "digest": m.group(2), "size": int(float(m.group(3)) * mult)})
    return rows


def parse_tailscale_status(text: str) -> tuple[str, str]:
    """(state, ip) from `tailscale status --json`."""
    try:
        d = json.loads(text)
    except ValueError:
        return "", ""
    state = {"Running": "running", "NeedsLogin": "needs-login", "Stopped": "stopped", "NoState": "stopped"}.get(d.get("BackendState", ""), "")
    ips = (d.get("Self") or {}).get("TailscaleIPs") or d.get("TailscaleIPs") or []
    return state, next((i for i in ips if "." in i), "")


def _run(cmd: list[str], timeout: int = 8) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def find_ollama() -> str:
    for p in (shutil.which("ollama"), os.path.expanduser("~/.local/bin/ollama"), "/usr/local/bin/ollama", "/usr/bin/ollama"):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return ""


def ollama_tags(host: str = "http://127.0.0.1:11434", timeout: float = 1.5) -> list[dict] | None:
    """Asks a running Ollama for its models; None if it is not running."""
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=timeout) as r:
            data = json.load(r)
        return [{"name": m["name"], "size": int(m.get("size", 0)), "digest": m.get("digest", "")} for m in data.get("models", [])]
    except (OSError, ValueError, KeyError):
        return None


def nearest_existing(path: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    return p or "/"


def detect(target: str, run=_run) -> Hardware:
    h = Hardware(arch=platform.machine(), os_name=platform.system(), python=sys.version_info[:3])
    try:
        osr = parse_os_release(open("/etc/os-release").read())
    except OSError:
        osr = {}
    h.distro_id = osr.get("ID", "").lower()
    h.distro_like = osr.get("ID_LIKE", "").lower().split()
    codename = osr.get("UBUNTU_CODENAME") or osr.get("VERSION_CODENAME", "")      # Mint reports its Ubuntu base here
    h.codename = codename
    try:
        h.ram_gb = parse_meminfo(open("/proc/meminfo").read())
    except OSError:
        pass
    h.disk_free_gb = round(shutil.disk_usage(nearest_existing(target)).free / 1e9, 1)
    h.gpus = parse_nvidia_smi(run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])) if shutil.which("nvidia-smi") else []
    if not h.gpus:
        for line in run(["lspci"]).splitlines():
            if re.search(r"VGA|3D|Display", line) and re.search(r"AMD|ATI|Radeon|Intel", line, re.I):
                h.other_gpu = line.split(": ", 1)[-1].strip()
                break
    h.is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    h.has_sudo = bool(shutil.which("sudo"))
    h.has_apt = bool(shutil.which("apt-get"))
    h.has_zstd = bool(shutil.which("zstd"))
    h.has_curl = bool(shutil.which("curl"))
    h.has_venv = _can(sys.executable, "import venv, ensurepip")
    h.has_pip = True
    h.ollama_path = find_ollama()
    tags = ollama_tags()
    h.ollama_running = tags is not None
    if tags is not None:
        h.ollama_models = tags
    elif h.ollama_path:
        h.ollama_models = parse_ollama_list(run([h.ollama_path, "list"]))
    h.tailscale_path = shutil.which("tailscale") or ""
    if h.tailscale_path:
        h.tailscale_state, h.tailscale_ip = parse_tailscale_status(run([h.tailscale_path, "status", "--json"]))
    h.systemd_user = bool(shutil.which("systemctl")) and _can("systemctl", "--user", "show-environment", raw=True)
    return h


def _can(*cmd, raw: bool = False) -> bool:
    c = list(cmd) if raw else [cmd[0], "-c", cmd[1]]
    try:
        return subprocess.run(c, capture_output=True, timeout=8).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
