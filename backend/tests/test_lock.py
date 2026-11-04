"""The hash-checked library lists (lock/): their shape, their coverage, and that the installer really makes pip enforce them."""
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lock"))
from installer import plan, system  # noqa: E402
import make_lock  # noqa: E402

LOCK = ROOT / "lock" / "py312"
MANIFEST = json.loads((ROOT / "lock" / "MANIFEST.json").read_text())


def entries(path: Path) -> dict[str, tuple[str, list[str]]]:
    """name -> (version, [sha256...]) for a lock file; fails on any line that is not a hash-pinned requirement."""
    out, cur = {}, None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--index-url") or line.startswith("--extra-index-url"):
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][^\s\;]*)\s*\\?$", line)
        if m:
            cur = re.sub(r"[-_.]+", "-", m.group(1)).lower()
            assert cur not in out, f"{cur} listed twice"
            out[cur] = (m.group(2), [])
            continue
        h = re.match(r"^--hash=sha256:([0-9a-f]{64})\s*\\?$", line)
        assert h and cur, f"unexpected line in {path.name}: {line[:80]}"
        out[cur][1].append(h.group(1))
    return out


def test_the_manifest_matches_the_files():
    e = MANIFEST["pythons"]["3.12"]
    assert MANIFEST["platform"] == "linux-x86_64" and set(e["files"]) == {"gpu.lock", "gpu-sdxl.lock", "cpu.lock", "source.lock"}
    assert all((ROOT / "lock" / e["dir"] / f).is_file() for f in e["files"])


@pytest.mark.parametrize("name", ["gpu.lock", "gpu-sdxl.lock", "cpu.lock", "source.lock"])
def test_every_line_is_an_exact_version_with_at_least_one_sha256(name):
    e = entries(LOCK / name)
    assert e
    for pkg, (ver, hashes) in e.items():
        assert hashes and len(set(hashes)) == len(hashes), pkg
    text = (LOCK / name).read_text()
    for bad in ("git+", "http://", "-e ", "file:", ">=", "<=", "~=", "@ ", "--trusted-host", "--find-links", "--no-index"):
        assert bad not in text, (name, bad)
    assert not re.search(r"^--index-url (?!https://download\.pytorch\.org/whl/cpu$)", text, re.M)
    assert not re.search(r"^--extra-index-url (?!https://pypi\.org/simple$)", text, re.M)


def test_the_lists_cover_everything_veil_needs():
    gpu, sdxl, cpu, src = (entries(LOCK / n) for n in ("gpu.lock", "gpu-sdxl.lock", "cpu.lock", "source.lock"))
    import tomllib
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    wanted = set()
    for spec in proj["dependencies"] + proj["optional-dependencies"]["editing"] + proj["optional-dependencies"]["faces"]:
        wanted.add(re.sub(r"[-_.]+", "-", re.match(r"^[A-Za-z0-9._-]+", spec).group(0)).lower())
    wanted.add("mediapipe")
    for name in wanted:
        assert name in gpu or name in src, name
        assert name in cpu or name in src, name
    gen = {re.sub(r"[-_.]+", "-", re.match(r"^[A-Za-z0-9._-]+", s).group(0)).lower() for s in proj["optional-dependencies"]["generative"]}
    assert gen <= set(sdxl) and not (gen & set(gpu)) and set(gpu) <= set(sdxl)
    assert "setuptools" in gpu and "setuptools" in cpu             # the source-only libraries are built with it, never with a download


def test_source_only_packages_are_separate_and_not_in_the_other_lists():
    src = entries(LOCK / "source.lock")
    assert set(src) == {"sam2", "iopath", "antlr4-python3-runtime"}
    for n in ("gpu.lock", "gpu-sdxl.lock", "cpu.lock"):
        assert not set(src) & set(entries(LOCK / n))


def test_gpu_and_cpu_builds_differ_where_they_must():
    gpu, cpu = entries(LOCK / "gpu.lock"), entries(LOCK / "cpu.lock")
    assert gpu["torch"][0] == "2.15.0" and cpu["torch"][0] == "2.15.0+cpu" and cpu["torchvision"][0].endswith("+cpu")
    assert any(n.startswith("nvidia-") for n in gpu) and "triton" in gpu
    assert not any(n.startswith(("nvidia-", "cuda-")) or n == "triton" for n in cpu)
    assert "--index-url https://download.pytorch.org/whl/cpu" in (LOCK / "cpu.lock").read_text()
    assert "--index-url" not in (LOCK / "gpu.lock").read_text()
    assert "opencv-contrib-python" not in gpu and "opencv-python-headless" in gpu    # mediapipe's OpenCV is never installed


def test_only_64_bit_linux_and_pure_python_files_are_accepted():
    ok = make_lock.wheel_ok
    assert ok("numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", 12)
    assert ok("pillow-12.0.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", 12)
    assert ok("six-1.16.0-py2.py3-none-any.whl", 12) and ok("idna-3.7-py3-none-any.whl", 12)
    assert ok("cryptography-45-cp311-abi3-manylinux_2_28_x86_64.whl", 12)
    assert ok("torch-2.15.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl", 12)
    for bad in ("numpy-2.5.4-cp312-cp312-macosx_11_0_arm64.whl", "numpy-2.5.4-cp312-cp312-win_amd64.whl", "numpy-2.5.4-cp312-cp312-manylinux_2_28_aarch64.whl",
                "numpy-2.5.4-cp312-cp312-musllinux_1_2_x86_64.whl", "numpy-2.5.4-cp311-cp311-manylinux_2_28_x86_64.whl", "numpy-2.5.4-pp310-pypy310_pp73-manylinux_2_28_x86_64.whl",
                "numpy-2.5.4-cp312-cp312-linux_aarch64.whl", "old-1.0-py2-none-any.whl", "numpy-2.5.4.tar.gz", "not-a-wheel.whl"):
        assert not ok(bad, 12), bad


def test_the_pinned_versions_are_exactly_what_the_tests_ran_with():
    """The lock was made from this environment, so this environment must match the NVIDIA list (a stale lock fails here)."""
    mism = system.lock_mismatches(Path(sys.executable), LOCK / "gpu-sdxl.lock")
    assert mism == [], mism


def test_a_mismatch_is_reported_by_name():
    mism = system.lock_mismatches(Path(sys.executable), LOCK / "cpu.lock")
    assert any(m.startswith("torch:") and "2.15.0+cpu" in m for m in mism)


# ---------------------------------------------------------------- the installer makes pip enforce them
class Cap:
    """A Runner that records commands and runs nothing."""
    def __init__(self):
        self.ran, self.said, self.envs = [], [], []
        self.dry_run = True

    def say(self, *a, **k): self.said.append(" ".join(map(str, a)))

    def run(self, cmd, **k):
        self.ran.append([str(c) for c in cmd]); self.envs.append(k.get("env"))


@pytest.mark.parametrize("cpu,sdxl,lockname", [(False, False, "gpu.lock"), (False, True, "gpu-sdxl.lock"), (True, False, "cpu.lock")])
def test_the_installer_asks_pip_to_require_hashes(cpu, sdxl, lockname, tmp_path):
    r = Cap()
    system.install_libs(r, Path("/v/bin/python"), tmp_path, cpu=cpu, sdxl=sdxl, minor=12)
    main, source = r.ran
    assert main[-1].endswith(f"py312/{lockname}") and source[-1].endswith("py312/source.lock")
    for cmd in (main, source):
        assert "--require-hashes" in cmd and "--no-deps" in cmd and "-r" in cmd
    assert "--only-binary=:all:" in main                       # a package without an approved wheel fails loudly instead of building from the net
    assert "--no-build-isolation" in source and r.envs[1] == {"SAM2_BUILD_CUDA": "0"}
    assert any("Pointing Veil's environment at the program" in x for x in r.said)          # the .pth step runs after the two pip steps
    flat = " ".join(" ".join(c) for c in r.ran)
    assert "--upgrade" not in flat and " torch " not in flat and "-e " not in flat and "mediapipe" not in flat      # nothing is installed outside the lists


def test_a_python_without_a_list_stops_with_a_plain_message(tmp_path):
    with pytest.raises(system.StepError, match="no hash-checked library list for Python 3.10") as e:
        system.install_libs(Cap(), Path("/v/bin/python"), tmp_path, cpu=False, sdxl=False, minor=10)
    assert "--unlocked" in str(e.value) and "3.12" in str(e.value)


def test_unlocked_is_an_explicit_choice_and_says_so(tmp_path):
    r = Cap()
    system.install_libs(r, Path("/v/bin/python"), tmp_path, cpu=True, sdxl=False, unlocked=True, minor=10)
    assert any("NOTHING here is hash-checked" in s for s in r.said)
    assert any("--upgrade" in c for c in r.ran) and any("-e" in c for c in r.ran)


def test_the_program_is_linked_with_a_pth_file_not_a_build(tmp_path, monkeypatch):
    class R(Cap):
        def __init__(self): super().__init__(); self.dry_run = False
    site = tmp_path / "site"; site.mkdir()
    monkeypatch.setattr(system, "purelib", lambda py: site)
    system.write_pth(R(), Path("/v/bin/python"), tmp_path)
    assert (site / "veil-studio.pth").read_text().strip() == str(tmp_path.resolve() / "backend")
    dry = Cap(); (site / "veil-studio.pth").unlink()
    system.write_pth(dry, Path("/v/bin/python"), tmp_path)
    assert not (site / "veil-studio.pth").exists()


def _wheel(path: Path, name="veillocktest", version="1.0") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{name}/__init__.py", "VALUE = 1\n")
        z.writestr(f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        z.writestr(f"{name}-{version}.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(f"{name}-{version}.dist-info/RECORD", "")
    data = buf.getvalue()
    (path / f"{name}-{version}-py3-none-any.whl").write_bytes(data)
    return data


def _pip_with_installers_flags(tmp_path, req_text):
    """Runs the installer's own flags (taken from install_libs) with real pip, against a local folder instead of the internet."""
    import hashlib
    r = Cap()
    system.install_libs(r, Path(sys.executable), tmp_path, cpu=False, sdxl=False, minor=12)
    flags = [a for a in r.ran[0][r.ran[0].index("install") + 1:] if a not in ("--no-cache-dir",)]
    flags = flags[: flags.index("-r")]
    req = tmp_path / "t.lock"; req.write_text(req_text)
    target = tmp_path / "out"
    return subprocess.run([sys.executable, "-m", "pip", "install", *flags, "--no-index", "--find-links", str(tmp_path), "--target", str(target), "-r", str(req)],
                          capture_output=True, text=True), target


def test_pip_installs_a_file_with_the_right_hash_and_refuses_a_changed_one(tmp_path):
    import hashlib
    data = _wheel(tmp_path)
    good = hashlib.sha256(data).hexdigest()
    ok, target = _pip_with_installers_flags(tmp_path, f"veillocktest==1.0 \\\n    --hash=sha256:{good}\n")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert (target / "veillocktest" / "__init__.py").exists()
    # the same name and version, but the file was swapped for another: refused, nothing installed
    swapped = tmp_path / "swap"; swapped.mkdir()
    (swapped / "veillocktest-1.0-py3-none-any.whl").write_bytes(data + b"tampered")
    bad = subprocess.run([sys.executable, "-m", "pip", "install", "--require-hashes", "--no-deps", "--only-binary=:all:", "--no-index", "--find-links", str(swapped),
                          "--target", str(tmp_path / "out2"), "-r", str(_write(tmp_path / "t2.lock", f"veillocktest==1.0 \\\n    --hash=sha256:{good}\n"))],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "hash" in (bad.stdout + bad.stderr).lower()
    assert not (tmp_path / "out2").exists()


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def test_pip_refuses_a_line_with_no_hash_at_all(tmp_path):
    _wheel(tmp_path)
    res, target = _pip_with_installers_flags(tmp_path, "veillocktest==1.0\n")
    assert res.returncode != 0 and not target.exists()
