"""What to install: the recommendation, and the exact list of downloads with sizes and licenses. No side effects."""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_registry():
    """Reads backend/veil/vision/registry.py directly (it needs nothing outside the standard library)."""
    spec = importlib.util.spec_from_file_location("veil_registry", ROOT / "backend" / "veil" / "vision" / "registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GB = 1e9
# Sizes measured on 2026-11-04 (PyPI file sizes for the NVIDIA build; the CPU figure is an estimate).
LIBS_GPU = {"download": 3.15 * GB, "installed": 6.1 * GB, "note": "PyTorch with NVIDIA CUDA libraries (2.4 GB of it), OpenCV, FastAPI, diffusers and others"}
LIBS_CPU = {"download": 0.4 * GB, "installed": 1.3 * GB, "note": "PyTorch (CPU build), OpenCV, FastAPI and others (size estimated)"}
MEDIAPIPE = 43 * 1e6
OLLAMA = {
    "version": "v0.40.14", "size": 1525335029, "extracted": 4.0 * GB,
    "sha256": "993e83d08c3d13d17c55009a3e2664c8a25445e644aec3a68bef5abb6abce67d",     # the digest GitHub lists for the release file
    "url": "https://github.com/ollama/ollama/releases/download/v0.40.14/ollama-linux-amd64.tar.zst",
}
TAILSCALE_SIZE = 41 * 1e6
HEADROOM = 1.25


@dataclass
class Item:
    kind: str            # system | libs | model | llm | ollama | tailscale
    name: str
    size: float          # bytes to download (0 if already there)
    license: str
    source: str
    note: str = ""
    have: bool = False   # already installed / downloaded
    need_sudo: bool = False
    key: str = ""        # registry model name, ollama model, etc.


@dataclass
class Choices:
    tier: str = "Recommended"
    sdxl: bool = False
    llm: str = "ollama"            # ollama | online | none
    tailscale: bool = False
    cpu: bool = False
    worker: bool = False           # set this computer up as a GPU worker for another computer (no language model, Tailscale required)
    models_dir: str = ""
    data_dir: str = ""


@dataclass
class Plan:
    choices: Choices
    items: list[Item] = field(default_factory=list)
    disk_needed: float = 0.0
    disk_free: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def download_total(self) -> float:
        return sum(i.size for i in self.items if not i.have)

    @property
    def disk_ok(self) -> bool:
        return self.disk_free >= self.disk_needed


def recommend(hw, reg) -> tuple[str, list[str]]:
    """Minimum / Recommended / High-end from the graphics card, memory and free space."""
    why = []
    v = hw.vram_gb
    if not hw.gpus:
        why.append("No NVIDIA graphics card was found, so Veil would run on the processor: slower, and only the Minimum set is practical."
                   + (f" (Found: {hw.other_gpu}. AMD and Intel graphics are not supported yet.)" if hw.other_gpu else ""))
        tier = "Minimum"
    elif v >= 15.5 and hw.ram_gb >= 30:
        tier = "High-end"
        why.append(f"Your graphics card has {v:.0f} GB of video memory and the computer has {hw.ram_gb:.0f} GB of memory.")
    elif v >= 7.5:
        tier = "Recommended"
        why.append(f"Your graphics card has {v:.0f} GB of video memory, which suits the Recommended set.")
    else:
        tier = "Minimum"
        why.append(f"Your graphics card has {v:.0f} GB of video memory, which suits the smaller Minimum set.")
    if hw.ram_gb and hw.ram_gb < 12 and tier != "Minimum":
        tier = "Minimum"
        why.append(f"The computer has only {hw.ram_gb:.0f} GB of memory, so the Minimum set is safer.")
    return tier, why


def build(hw, reg, choices: Choices, libs_present: bool = False) -> Plan:
    plan = Plan(choices=choices)
    t = reg.TIERS[choices.tier]
    gpu = bool(hw.gpus) and not choices.cpu
    libs = LIBS_GPU if gpu else LIBS_CPU
    missing = [p for p, ok in (("python3-venv", hw.has_venv), ("curl", hw.has_curl)) if not ok]
    if choices.worker:
        choices.llm = "none"
        choices.tailscale = True
    if choices.llm == "ollama" and not hw.has_zstd and not hw.ollama_path:
        missing.append("zstd")
    if missing:
        if hw.debian_family:
            plan.items.append(Item("system", "System packages: " + ", ".join(missing), 0, "Debian/Ubuntu free-software licenses", "your distribution (apt)",
                                   "small, a few MB", need_sudo=True, key=" ".join(missing)))
        else:
            plan.items.append(Item("system", "System packages you install yourself: " + ", ".join(missing), 0, "your distribution's licenses", "your distribution",
                                   "the installer prints the command to run, then stops", need_sudo=True, key=" ".join(missing)))
    plan.items.append(Item("libs", "Python libraries" + (" (NVIDIA build)" if gpu else " (CPU build)"), libs["download"],
                           ("PyTorch BSD-3, NVIDIA CUDA libraries under NVIDIA's license, most others Apache-2.0/MIT/BSD" if gpu else "PyTorch BSD-3, most others Apache-2.0/MIT/BSD"), "pypi.org" + ("" if gpu else " and download.pytorch.org"),
                           libs["note"], have=libs_present, key="gpu" if gpu else "cpu"))
    plan.items.append(Item("libs", "MediaPipe library", MEDIAPIPE, "Apache-2.0", "pypi.org", "installed without its OpenCV dependency, as Veil needs", key="mediapipe"))
    names = list(t["models"]) + (["sdxl-inpaint"] if choices.sdxl and gpu else [])
    for n in names:
        m = reg.MODELS[n]
        size = sum(f["size"] for f in m["files"])
        have = reg.installed(Path(choices.models_dir), n, deep=False)
        plan.items.append(Item("model", m["label"], size, m["license"], _host(m["files"][0]["url"]), m["use"], have=have, key=n))
    if choices.llm == "ollama":
        llm = reg.LLMS[t["llm"]]
        local = next((x for x in hw.ollama_models if x["name"] in (t["llm"], t["llm"] + ":latest")), None)
        if not hw.ollama_path:
            plan.items.append(Item("ollama", f"Ollama {OLLAMA['version']} (for you only, no sudo)", OLLAMA["size"], "MIT",
                                   "github.com/ollama/ollama", "runs the language model; checked against the SHA-256 GitHub lists for the file", key="ollama"))
        plan.items.append(Item("llm", f"Language model {t['llm']}", llm["size"], llm["license"], "ollama.com",
                               "you already have it, so it is reused" if local else "Ollama checks every part of it", have=bool(local), key=t["llm"]))
    if choices.tailscale and hw.tailscale_state != "running":
        if hw.debian_family or hw.tailscale_path:
            plan.items.append(Item("tailscale", "Tailscale (" + ("worker access" if choices.worker else "phone access") + ")", 0 if hw.tailscale_path else TAILSCALE_SIZE, "BSD-3-Clause", "pkgs.tailscale.com",
                                   ("the private network your own computer reaches this worker through" if choices.worker else "private network between your computer and phone"), have=bool(hw.tailscale_path), need_sudo=not hw.tailscale_path, key="tailscale"))
        else:
            plan.items.append(Item("tailscale", "Tailscale (you install it yourself)", 0, "BSD-3-Clause", "tailscale.com/download/linux",
                                   "not Debian/Ubuntu/Mint, so the installer shows the steps and you install it", key="tailscale"))
    # disk: peak use = what is downloaded plus what is unpacked, and a quarter extra
    peak = 0.0
    for i in plan.items:
        if i.have:
            continue
        if i.kind == "libs" and i.key in ("gpu", "cpu"):
            peak += libs["download"] + libs["installed"]
        elif i.kind == "ollama":
            peak += OLLAMA["size"] + OLLAMA["extracted"]
        else:
            peak += i.size
    plan.disk_needed = peak * HEADROOM
    plan.disk_free = hw.disk_free_gb * GB
    if not plan.disk_ok:
        plan.warnings.append(f"This needs about {plan.disk_needed / GB:.0f} GB free and {hw.disk_free_gb:.0f} GB is available where Veil will be installed.")
    if choices.sdxl and not gpu:
        plan.warnings.append("The generative model (SDXL) needs an NVIDIA card, so it was left out.")
    if gpu and choices.tier == "Minimum" and hw.vram_gb >= 12:
        plan.warnings.append("Your card could run a bigger set; Minimum is smaller and faster but less precise.")
    if gpu and choices.tier == "High-end" and hw.vram_gb < 15.5:
        plan.warnings.append(f"High-end is made for 16 GB or more of video memory; your card has {hw.vram_gb:.0f} GB, so it may run out of memory.")
    if gpu and choices.tier == "Recommended" and hw.vram_gb < 7.5:
        plan.warnings.append(f"Recommended is made for 8 GB or more of video memory; your card has {hw.vram_gb:.0f} GB.")
    return plan


def _host(url: str) -> str:
    return url.split("/")[2]


def human(n: float) -> str:
    if n >= GB:
        return f"{n / GB:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{n:.0f} B" if n else "—"
