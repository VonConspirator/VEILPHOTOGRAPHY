#!/usr/bin/env python3
"""Writes Veil's hash-checked lock files for the Python this script runs under.

    .venv/bin/python lock/make_lock.py          # run inside a Veil environment that passes the full test suite

What it does: it takes the exact versions installed in the running environment (the ones the tests passed with) for Veil's
libraries and everything they need, asks PyPI (and, for the processor build, PyTorch's own index) for the SHA-256 of every
file that could be installed on 64-bit Linux, and writes:

    lock/py<NNN>/gpu.lock        NVIDIA build                   (pip install --require-hashes --no-deps -r gpu.lock)
    lock/py<NNN>/gpu-sdxl.lock   NVIDIA build + generative fill
    lock/py<NNN>/cpu.lock        processor-only build
    lock/py<NNN>/source.lock     the few packages that exist only as source (built after the others, from hash-checked source)
    lock/MANIFEST.json           which Pythons have locks

Nothing is downloaded except index metadata; the files themselves are only ever fetched by the installer, and pip refuses any
file whose SHA-256 is not listed. Needs the `packaging` library (it is installed with Veil's environment).
"""
from __future__ import annotations

import datetime
import html
import importlib.metadata as md
import json
import re
import sys
try:
    import tomllib
except ImportError:                            # Python 3.10
    import tomli as tomllib
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parent.parent
PY = sys.version_info[:2]
EXCLUDE = {"opencv-contrib-python"}            # mediapipe asks for it; Veil uses opencv-python-headless and installs mediapipe without its dependencies
CPU_DROP = ("nvidia-", "cuda-", "triton")      # CUDA pieces: not part of the processor-only build
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
UA = {"User-Agent": "veil-lock-maker (https://github.com/)"}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


# ---------------------------------------------------------------- which packages
def installed() -> dict[str, md.Distribution]:
    return {canonicalize_name(d.metadata["Name"]): d for d in md.distributions()}


def closure(dists: dict, roots: list[str]) -> dict[str, str]:
    """name -> installed version, for the roots and everything they need on this machine."""
    out: dict[str, str] = {}
    todo = [(Requirement(r), None) for r in roots]
    seen_extras: dict[str, set] = {}
    while todo:
        req, _ = todo.pop()
        name = canonicalize_name(req.name)
        if name in EXCLUDE:
            continue
        if name not in dists:
            raise SystemExit(f"{req.name} is needed but not installed here. Run this inside a full Veil environment (editing, faces, generative).")
        extras = set(req.extras)
        if name in out and extras <= seen_extras.get(name, set()):
            continue
        out[name] = dists[name].version
        seen_extras.setdefault(name, set()).update(extras)
        for line in dists[name].requires or []:
            r = Requirement(line)
            if r.marker is not None and not any(r.marker.evaluate({"extra": e}) for e in (seen_extras[name] or {""}) | {""}):
                continue
            todo.append((r, name))
    return out


def project_roots() -> tuple[list[str], list[str]]:
    p = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    base = list(p["dependencies"]) + p["optional-dependencies"]["editing"] + p["optional-dependencies"]["faces"] + ["mediapipe"]
    return base, base + p["optional-dependencies"]["generative"]


# ---------------------------------------------------------------- which files
def lower_bound(spec: str) -> tuple[int, int]:
    for s in SpecifierSet(spec or ""):
        if s.operator in (">=", ">", "~=", "==") and re.match(r"^\d+\.\d+", s.version):
            a, b = s.version.split(".")[:2]
            return int(a), int(b) + (1 if s.operator == ">" else 0)
    return (3, 0)


def wheel_ok(filename: str, target: int) -> bool:
    """A 64-bit Linux (glibc) wheel that CPython 3.<target> can use, or a pure-Python wheel. Wheels for other Python versions are not listed."""
    try:
        _, _, _, tags = parse_wheel_filename(filename)
    except Exception:
        return False
    for t in tags:
        plat_ok = t.platform == "any" or re.match(r"^(manylinux\w*|linux)_x86_64$", t.platform) is not None or \
            (t.platform.startswith("manylinux") and t.platform.endswith("_x86_64"))
        if not plat_ok:
            continue
        i = t.interpreter
        if i in ("py3", "py2.py3", "py30"):
            return True
        m = re.match(r"^(cp|py)3(\d+)$", i)
        if not m:
            continue
        minor = int(m.group(2))
        if m.group(1) == "py" or t.abi == "abi3":
            if minor <= target:                      # a stable-ABI or pure wheel built for this Python or an older one
                return True
        elif minor == target and t.abi in (i, "none"):
            return True
    return False


def pypi_files(name: str, version: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    """([(filename, sha256) wheels], [(filename, sha256) sdists], requires_python) for one release."""
    d = json.loads(fetch(f"https://pypi.org/pypi/{name}/{version}/json"))
    wheels, sdists = [], []
    for u in d["urls"]:
        item = (u["filename"], u["digests"]["sha256"])
        (wheels if u["packagetype"] == "bdist_wheel" else sdists if u["packagetype"] == "sdist" else []).append(item)
    return wheels, sdists, d["info"].get("requires_python") or ""


def torch_cpu_files(name: str, version: str, pymin: int) -> list[tuple[str, str]]:
    page = fetch(f"{TORCH_CPU_INDEX}/{name}/").decode()
    out = []
    for m in re.finditer(r'href="([^"]+)"', page):
        href = html.unescape(m.group(1))
        fn, _, frag = href.partition("#sha256=")
        fn = fn.rsplit("/", 1)[-1].replace("%2B", "+")
        if frag and fn.startswith(f"{name}-{version}+cpu-") and wheel_ok(fn, pymin):
            out.append((fn, frag))
    return out


# ---------------------------------------------------------------- writing
def block(name: str, version: str, files: list[tuple[str, str]]) -> str:
    hashes = sorted({h for _, h in files})
    return f"{name}=={version} \\\n" + " \\\n".join(f"    --hash=sha256:{h}" for h in hashes) + "\n"


def write(path: Path, header: list[str], blocks: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"# {h}" if h and not h.startswith("--") else h for h in header) + "\n\n" + "\n".join(blocks))


def main() -> int:
    dists = installed()
    base_roots, sdxl_roots = project_roots()
    sets = {"gpu": closure(dists, base_roots), "gpu-sdxl": closure(dists, sdxl_roots)}
    sets["cpu"] = {n: v for n, v in sets["gpu"].items() if not n.startswith(CPU_DROP)}
    # the oldest Python every locked package allows, from the index
    meta, pymin = {}, 0
    for n in sorted(set(sets["gpu-sdxl"])):
        w, s, rp = pypi_files(n, sets["gpu-sdxl"][n])
        meta[n] = (w, s)
        pymin = max(pymin, lower_bound(rp)[1] if lower_bound(rp)[0] == 3 else 0)
    pymin = max(pymin, 10)
    print(f"Oldest Python every package allows: 3.{pymin}.  Locking for the Python running this script: 3.{PY[1]}.")
    if PY[1] < pymin:
        raise SystemExit("This Python is older than the packages allow.")
    source_only = {n for n, (w, s) in meta.items() if not [x for x in w if wheel_ok(x[0], PY[1])]}
    print("Source-only packages (built from hash-checked source):", ", ".join(sorted(source_only)) or "none")
    gen = datetime.date.today().isoformat()
    outdir = ROOT / "lock" / f"py3{PY[1]}"
    common = [f"Written by lock/make_lock.py on {gen} for CPython 3.{PY[1]} on 64-bit Linux (glibc). Do not edit by hand.",
              "Install with:  pip install --require-hashes --no-deps -r <this file>   (pip refuses any file whose SHA-256 is not listed here)"]
    for kind in ("gpu", "gpu-sdxl", "cpu"):
        blocks = []
        for n, v in sorted(sets[kind].items()):
            if n in source_only:
                continue
            if kind == "cpu" and n in ("torch", "torchvision"):
                files = torch_cpu_files(n, v, PY[1])
                if not files:
                    raise SystemExit(f"{n} {v}+cpu is not on {TORCH_CPU_INDEX}. Re-run in an environment whose torch version has a CPU build.")
                blocks.append(block(n, v + "+cpu", files))
                continue
            files = [x for x in meta[n][0] if wheel_ok(x[0], PY[1])]
            blocks.append(block(n, v, files))
        header = common + ([f"The processor-only build: torch and torchvision come from {TORCH_CPU_INDEX}.", "",
                            f"--index-url {TORCH_CPU_INDEX}", "--extra-index-url https://pypi.org/simple"] if kind == "cpu" else [])
        write(outdir / f"{kind}.lock", header, blocks)
        print(f"  {kind}.lock: {len(blocks)} packages")
    src = [block(n, sets["gpu-sdxl"][n], meta[n][1]) for n in sorted(source_only)]
    write(outdir / "source.lock", common + ["Source-only packages. Install AFTER the other lock, with:",
                                           "pip install --require-hashes --no-deps --no-build-isolation -r source.lock"], src)
    manifest_path = ROOT / "lock" / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        manifest = {"format": 1, "platform": "linux-x86_64", "pythons": {}}
    manifest["pythons"][f"3.{PY[1]}"] = {"dir": outdir.name, "generated": gen, "torch": sets["gpu"]["torch"],
                                         "files": ["gpu.lock", "gpu-sdxl.lock", "cpu.lock", "source.lock"]}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
