"""The steps that change the computer. Every command is shown before it runs; `sudo` commands are only run after the owner
agrees, and nothing here ever types a password. With dry_run nothing is changed at all."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import net, plan

RECORD = Path("~/.config/veil/install.json").expanduser()
LAUNCHER = Path("~/.local/bin/veil").expanduser()
DESKTOP = Path("~/.local/share/applications/veil.desktop").expanduser()
UNIT = Path("~/.config/systemd/user/ollama.service").expanduser()
LOG = Path("~/.local/state/veil/install.log").expanduser()
OLLAMA_PREFIX = Path("~/.local").expanduser()


class StepError(Exception):
    """A step failed; the message says what to do next."""


@dataclass
class Runner:
    dry_run: bool = False
    assume_yes: bool = False
    no_sudo: bool = False
    is_root: bool = False
    ask: object = None                       # callable(question, default) -> bool
    say: object = print
    log_path: Path = LOG
    ran: list = field(default_factory=list)

    def _log(self, text: str) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(text if text.endswith("\n") else text + "\n")
        except OSError:
            pass

    def confirm(self, question: str, default: bool = True) -> bool:
        if self.assume_yes:
            return True
        return bool(self.ask(question, default)) if self.ask else default

    def run(self, cmd: list[str], *, sudo: bool = False, why: str = "", cwd: Path | None = None, env: dict | None = None,
            input: bytes | None = None, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess | None:
        full = (["sudo"] if sudo and not self.is_root else []) + [str(c) for c in cmd]
        shown = " ".join(shlex.quote(c) for c in full)
        if why:
            self.say(f"  {why}")
        self.say(f"    $ {shown}")
        self.ran.append(shown)
        if self.dry_run:
            return None
        if sudo and not self.is_root:
            if self.no_sudo:
                raise StepError("This step needs administrator rights and --no-sudo was given. Run the command above yourself, then run the installer again.")
            if not self.confirm("This step needs your computer's administrator password (you type it here, Veil never sees it). Run it now?"):
                raise StepError("You chose not to run an administrator step. Run the command above yourself, then run the installer again.")
        self._log("$ " + shown)
        try:
            proc = subprocess.Popen(full, cwd=cwd, env={**os.environ, **(env or {})}, stdin=subprocess.PIPE if input is not None else None,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False)
            if input is not None:
                proc.stdin.write(input); proc.stdin.close()
            tail = []
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                self._log(line)
                tail.append(line); tail = tail[-12:]
                if not quiet and line and ("\r" in line or len(line) < 200):
                    self.say("      " + line[:150])
                elif quiet and len(tail) and (time.monotonic() - getattr(self, "_dot", 0)) > 8:
                    self._dot = time.monotonic()
                    sys.stdout.write("."); sys.stdout.flush()
            code = proc.wait()
        except OSError as e:
            raise StepError(f"Could not run {full[0]}: {e}") from None
        if check and code != 0:
            raise StepError(f"A step failed (exit code {code}): {shown}\n" + "\n".join("    " + t for t in tail[-6:]) + f"\n  The full log is in {self.log_path}.")
        return subprocess.CompletedProcess(full, code, "\n".join(tail), "")


# ---------------------------------------------------------------- system packages
def apt_packages(r: Runner, packages: list[str]) -> None:
    r.run(["apt-get", "update"], sudo=True, why="Updating the list of available packages.", quiet=True)
    r.run(["apt-get", "install", "-y", *packages], sudo=True, why="Installing: " + ", ".join(packages) + ".", quiet=True,
          env={"DEBIAN_FRONTEND": "noninteractive"})


def manual_prereq_help(hw, missing: list[str]) -> str:
    """For distributions other than Debian/Ubuntu/Mint: what to run by hand."""
    d = hw.distro_id
    if d in ("fedora", "rhel", "centos", "rocky", "almalinux") or "fedora" in hw.distro_like:
        cmd = "sudo dnf install -y python3 python3-pip curl zstd"
    elif d in ("arch", "manjaro", "endeavouros") or "arch" in hw.distro_like:
        cmd = "sudo pacman -S --needed python python-pip curl zstd"
    elif d in ("opensuse-tumbleweed", "opensuse-leap", "sles") or "suse" in hw.distro_like:
        cmd = "sudo zypper install python3 python3-pip curl zstd"
    else:
        cmd = "install python 3.10 or newer (with venv and pip), curl and zstd with your package manager"
    return (f"This computer is not Debian, Ubuntu or Mint, so Veil's installer does not install system packages for you yet.\n"
            f"  Missing: {', '.join(missing)}.\n  Run this yourself, then start the installer again:\n    {cmd}")


# ---------------------------------------------------------------- Python environment
def make_venv(r: Runner, venv: Path) -> Path:
    py = venv / "bin" / "python"
    if py.exists():
        r.say(f"  The Python environment {venv} already exists; keeping it.")
        return py
    r.run([sys.executable, "-m", "venv", str(venv)], why="Creating Veil's own Python environment (nothing is installed system-wide).")
    return py


def libs_ready(py: Path) -> bool:
    """Are the libraries already in Veil's environment? Looks for them without importing them (importing creates cache folders)."""
    if not py.exists():
        return False
    code = ("import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in "
            "('torch', 'cv2', 'fastapi', 'timm', 'sam2', 'mediapipe', 'PIL', 'numpy', 'veil')) else 1)")
    return subprocess.run([str(py), "-I", "-c", code], capture_output=True).returncode == 0


def install_libs(r: Runner, py: Path, repo: Path, cpu: bool, sdxl: bool) -> None:
    pip = [str(py), "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check"]
    r.run([*pip, "--upgrade", "pip"], why="Updating pip inside Veil's environment.", quiet=True)
    if cpu:
        r.run([*pip, "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"], why="Installing PyTorch (CPU build).", quiet=True)
    extras = "editing,faces" + (",generative" if sdxl else "")
    r.run([*pip, "-e", f"{repo}[{extras}]"], why="Installing Veil and its libraries (this is the longest step).", quiet=True)
    r.run([*pip, "--no-deps", "mediapipe"], why="Installing MediaPipe without its OpenCV dependency.", quiet=True)


def self_check(py: Path, repo: Path, models_dir: Path, tier_models: list[str]) -> list[str]:
    """Runs inside Veil's environment: libraries import, models verify, and the engine says it is ready. Returns problems."""
    code = (
        "import sys, json\n"
        "from pathlib import Path\n"
        "from veil.vision import registry\n"
        "from veil.vision.engine import Engine\n"
        f"root = Path({str(models_dir)!r})\n"
        "bad = []\n"
        f"for n in {tier_models!r}:\n"
        "    try: registry.verify(root, n)\n"
        "    except Exception as e: bad.append(str(e))\n"
        "st = Engine(root).status()\n"
        "print(json.dumps({'bad': bad, 'ready': st['ready'], 'device': st['device'], 'faces': st['faces_libs']}))\n"
    )
    p = subprocess.run([str(py), "-I", "-c", code], capture_output=True, text=True, cwd=str(repo / "backend"), env={**os.environ, "PYTHONPATH": str(repo / "backend")})
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return [f"The self-check could not run: {p.stderr.strip()[-300:]}"]
    problems = list(d["bad"])
    if not d["ready"]:
        problems.append("Veil's editing engine says it is not ready.")
    if not d["faces"]:
        problems.append("The face library (mediapipe) did not load.")
    return problems


# ---------------------------------------------------------------- Ollama (installed for this user only, no sudo)
def safe_members(names: list[str]) -> list[str]:
    """Names in the archive that would land outside the install folder (absolute paths or '..')."""
    return [n for n in names if n.startswith("/") or ".." in Path(n).parts]


def install_ollama(r: Runner, hw, prefix: Path = OLLAMA_PREFIX, show=net.bar) -> Path:
    exe = prefix / "bin" / "ollama"
    if exe.exists():
        r.say("  Ollama is already installed for you; keeping it.")
        return exe
    o = plan.OLLAMA
    r.say(f"  Downloading Ollama {o['version']} ({plan.human(o['size'])}), then checking its SHA-256.")
    tmp = prefix / "share" / "veil-install" / "ollama-linux-amd64.tar.zst"
    r.say(f"    $ download {o['url']}")
    if not r.dry_run:
        net.fetch(o["url"], tmp, o["size"], sha256=o["sha256"], label="Ollama")
        names = subprocess.run(["tar", "--zstd", "-tf", str(tmp)], capture_output=True, text=True)
        if names.returncode != 0:
            raise StepError("Veil could not read the Ollama archive (is 'zstd' installed?).")
        bad = safe_members(names.stdout.split("\n"))
        if bad:
            tmp.unlink(missing_ok=True)
            raise StepError("The Ollama archive contains paths outside its folder, so it was not unpacked: " + ", ".join(bad[:3]))
    r.run(["tar", "--zstd", "-xf", str(tmp), "-C", str(prefix)], why=f"Unpacking Ollama into {prefix} (your own folder, no administrator rights).")
    if not r.dry_run:
        tmp.unlink(missing_ok=True)
        if not exe.exists():
            raise StepError("Ollama did not unpack where expected.")
    return exe


def ollama_unit_text(exe: Path) -> str:
    return ("[Unit]\nDescription=Ollama (installed by Veil's installer, for this user only)\nAfter=network-online.target\n\n"
            f"[Service]\nExecStart={exe} serve\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n")


def ollama_service(r: Runner, hw, exe: Path) -> bool:
    """Starts Ollama at login, for this user. Returns False when this computer has no user systemd (the launcher then starts it)."""
    if not hw.systemd_user:
        r.say("  This computer has no user service manager here, so the 'veil' command will start Ollama when needed.")
        return False
    r.say(f"  Writing {UNIT} so Ollama starts when you log in.")
    if not r.dry_run:
        UNIT.parent.mkdir(parents=True, exist_ok=True)
        UNIT.write_text(ollama_unit_text(exe))
    r.run(["systemctl", "--user", "daemon-reload"], check=False)
    r.run(["systemctl", "--user", "enable", "--now", "ollama.service"], why="Turning the service on.", check=False)
    return True


def wait_ollama(timeout: float = 40) -> bool:
    from .hw import ollama_tags
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if ollama_tags() is not None:
            return True
        time.sleep(1)
    return False


def blob_digest(exe: Path, model: str) -> str:
    """sha256 of the model's weights file, read from `ollama show --modelfile`."""
    out = subprocess.run([str(exe), "show", "--modelfile", model], capture_output=True, text=True).stdout
    m = re.search(r"^FROM\s+.*sha256-([0-9a-f]{64})\s*$", out, re.M)
    return m.group(1) if m else ""


def pull_llm(r: Runner, hw, reg, exe: Path, model: str, started: list) -> str:
    """Makes sure the model is there. Returns "" if its weights match the recorded digest, otherwise a note for the owner."""
    from .hw import ollama_tags
    names = (model, model + ":latest")
    local = any(x["name"] in names for x in hw.ollama_models)
    if not r.dry_run and ollama_tags() is None:
        r.say("  Starting Ollama (it is stopped again afterwards).")
        proc = subprocess.Popen([str(exe), "serve"], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        started.append(proc)
        if not wait_ollama():
            raise StepError("Ollama did not start. Look at the log or run 'ollama serve' in a terminal to see why.")
    if not r.dry_run and not local:                         # the list can only be read from a running Ollama
        local = any(x["name"] in names for x in (ollama_tags() or []))
    want = reg.LLMS[model]["weights"]
    if local:
        r.say(f"  You already have {model}; reusing it.")
        got = blob_digest(exe, model) if not r.dry_run else ""
        return "" if (r.dry_run or got == want or not got) else (f"Your copy of {model} is an older or different build than the one Ollama serves today "
                                                                 f"(your weights {got[:12]}, today's {want[:12]}). It still works, so it was kept.")
    r.run([str(exe), "pull", model], why=f"Downloading {model} ({plan.human(reg.LLMS[model]['size'])}). Ollama checks every part of it. A dot appears every few seconds.", quiet=True)
    if r.dry_run:
        return ""
    got = blob_digest(exe, model)
    if got != want:
        return (f"{model} was downloaded and Ollama accepted it, but its weights ({got[:12] or 'unknown'}) are not the build recorded in Veil "
                f"({want[:12]}). Ollama tags can be re-published; this is probably a newer build. Check ollama.com/library/qwen3.5 if you want to be sure.")
    return ""


def write_ai_settings(r: Runner, data_dir: Path, model: str) -> str:
    """Tells Veil which Ollama model was set up (data/ai.json, owner-only). Settings you already have are never overwritten."""
    path = data_dir / "ai.json"
    if path.exists():
        return (f"Your existing language-model settings in {path} were kept. If you want Veil to use {model}, open Model in Veil and choose it.")
    r.say(f"  Telling Veil to use {model} ({path}).")
    if r.dry_run:
        return ""
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"active": "local-ollama", "providers": [{"id": "local-ollama", "name": "Ollama (this computer)", "kind": "ollama",
                                                    "base_url": "http://127.0.0.1:11434", "model": model, "vision": True}]}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return ""


# ---------------------------------------------------------------- Tailscale (Debian / Ubuntu / Mint)
def tailscale_repo(hw) -> tuple[str, str]:
    """(distribution, release name) of Tailscale's apt repository to use. Mint and other derivatives use their Ubuntu/Debian base."""
    d = "ubuntu" if (hw.distro_id in ("ubuntu", "linuxmint", "pop", "elementary", "zorin") or "ubuntu" in hw.distro_like) else "debian"
    return d, hw.codename


def tailscale_install(r: Runner, hw, show_text=print) -> None:
    dist, code = tailscale_repo(hw)
    if not code:
        raise StepError("Veil could not tell which Debian/Ubuntu release this is, so it cannot pick Tailscale's package source. See https://tailscale.com/download/linux")
    base = f"https://pkgs.tailscale.com/stable/{dist}/{code}"
    if r.dry_run:
        for c in (f"download {base}.noarmor.gpg  ->  /usr/share/keyrings/tailscale-archive-keyring.gpg (needs administrator)",
                  f"download {base}.tailscale-keyring.list  ->  /etc/apt/sources.list.d/tailscale.list (needs administrator)",
                  "apt-get update && apt-get install -y tailscale"):
            r.say("    $ " + c)
        return
    key, lst = (_get(base + ".noarmor.gpg"), _get(base + ".tailscale-keyring.list"))
    r.run(["tee", "/usr/share/keyrings/tailscale-archive-keyring.gpg"], sudo=True, input=key, quiet=True,
          why="Adding Tailscale's signing key (so apt can check its packages).")
    r.run(["tee", "/etc/apt/sources.list.d/tailscale.list"], sudo=True, input=lst, quiet=True, why="Adding Tailscale's package source.")
    r.run(["apt-get", "update"], sudo=True, quiet=True)
    r.run(["apt-get", "install", "-y", "tailscale"], sudo=True, why="Installing Tailscale.", quiet=True, env={"DEBIAN_FRONTEND": "noninteractive"})


def _get(url: str) -> bytes:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "veil-installer"}), timeout=30) as resp:
            return resp.read()
    except OSError as e:
        raise StepError(f"Could not fetch {url}: {e}") from None


# ---------------------------------------------------------------- launcher, menu entry, record
def launcher_text(venv: Path, models: Path, data: Path, ollama: Path | None, ollama_needs_start: bool) -> str:
    q = shlex.quote
    start = ""
    if ollama and ollama_needs_start:
        start = (f"if ! {q(str(venv / 'bin' / 'python'))} -I -c \"import urllib.request as u; u.urlopen('http://127.0.0.1:11434/api/tags', timeout=1)\" 2>/dev/null; then\n"
                 f"  mkdir -p {q(str(data))}; nohup {q(str(ollama))} serve >{q(str(data / 'ollama.log'))} 2>&1 &\nfi\n")
    return ("#!/bin/sh\n# Starts Veil. Written by Veil's installer; safe to delete (uninstall.sh removes it).\n"
            f"export VEIL_MODELS_DIR={q(str(models))}\nexport VEIL_DATA_DIR={q(str(data))}\n{start}"
            f"exec {q(str(venv / 'bin' / 'python'))} -m veil \"$@\"\n")


def write_launcher(r: Runner, venv: Path, models: Path, data: Path, ollama: Path | None, needs_start: bool) -> None:
    r.say(f"  Writing the 'veil' command: {LAUNCHER}")
    if r.dry_run:
        return
    LAUNCHER.parent.mkdir(parents=True, exist_ok=True)
    LAUNCHER.write_text(launcher_text(venv, models, data, ollama, needs_start))
    LAUNCHER.chmod(0o755)


def write_desktop(r: Runner) -> None:
    r.say(f"  Adding Veil to your application menu: {DESKTOP}")
    if r.dry_run:
        return
    DESKTOP.parent.mkdir(parents=True, exist_ok=True)
    DESKTOP.write_text("[Desktop Entry]\nType=Application\nName=Veil\nComment=Private photo editor\n"
                       f"Exec={LAUNCHER}\nTerminal=true\nCategories=Graphics;Photography;\n")


def write_record(r: Runner, record: dict) -> None:
    if r.dry_run:
        return
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    try:
        old = json.loads(RECORD.read_text())
    except (OSError, ValueError):
        pass
    old.update(record)
    RECORD.write_text(json.dumps(old, indent=1))


def read_record() -> dict:
    try:
        return json.loads(RECORD.read_text())
    except (OSError, ValueError):
        return {}
