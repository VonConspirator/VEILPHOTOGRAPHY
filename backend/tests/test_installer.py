"""Phase 8: the installer. Parsers, recommendation, the plan, verified downloads, safety of the Ollama archive, and dry-run /
uninstall in a throwaway home folder. (The real, clean-machine run is done in a container.)"""
import hashlib
import http.server
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from installer import cli, hw, net, plan, system  # noqa: E402

reg = plan.load_registry()


@pytest.fixture(autouse=True)
def never_touch_the_real_machine(tmp_path, monkeypatch):
    """No test in this file may write to the real home folder or run a real install step (an earlier version of one test did)."""
    for name, rel in (("RECORD", ".config/veil/install.json"), ("LAUNCHER", ".local/bin/veil"), ("WORKER_LAUNCHER", ".local/bin/veil-worker"), ("DESKTOP", ".local/share/applications/veil.desktop"),
                      ("UNIT", ".config/systemd/user/ollama.service"), ("LOG", ".local/state/veil/install.log")):
        monkeypatch.setattr(system, name, tmp_path / "fakehome" / rel)
    monkeypatch.setattr(system, "OLLAMA_PREFIX", tmp_path / "fakehome" / ".local")

    def refuse(*a, **k):
        raise AssertionError("an in-process test tried to run the real install steps")
    monkeypatch.setattr(cli, "_real_run_steps", cli.run_steps, raising=False)     # only dry runs may use it (see the worker test)
    monkeypatch.setattr(cli, "run_steps", refuse)


def H(**kw):
    base = dict(arch="x86_64", distro_id="ubuntu", codename="noble", python=(3, 12, 3), has_venv=True, has_apt=True, has_zstd=True, has_curl=True,
                ram_gb=32.0, disk_free_gb=500.0, gpus=[("NVIDIA GeForce RTX 3060", 12.0)], is_root=False, has_sudo=True)
    base.update(kw)
    return hw.Hardware(**base)


# ---------------------------------------------------------------- parsers
def test_os_release_for_mint_uses_its_ubuntu_base():
    d = hw.parse_os_release('NAME="Linux Mint"\nID=linuxmint\nID_LIKE="ubuntu debian"\nVERSION_CODENAME=zena\nUBUNTU_CODENAME=noble\n')
    assert d["ID"] == "linuxmint" and d["UBUNTU_CODENAME"] == "noble"
    h = H(distro_id="linuxmint", distro_like=["ubuntu", "debian"], codename="noble")
    assert h.debian_family and system.tailscale_repo(h) == ("ubuntu", "noble")


def test_debian_and_non_debian_families():
    assert system.tailscale_repo(H(distro_id="debian", codename="bookworm")) == ("debian", "bookworm")
    assert H(distro_id="debian").debian_family
    assert not H(distro_id="fedora", has_apt=False).debian_family
    assert not H(distro_id="arch", has_apt=False).debian_family
    assert not H(distro_id="ubuntu", has_apt=False).debian_family        # no apt-get, so no automatic package steps


def test_meminfo_and_nvidia_smi_parsers():
    assert hw.parse_meminfo("MemTotal:       32559804 kB\nMemFree: 1 kB\n") == 31.1
    assert hw.parse_meminfo("nonsense") == 0.0
    assert hw.parse_nvidia_smi("NVIDIA GeForce RTX 3090, 24576\nNVIDIA GeForce RTX 3090, 24576\n") == [("NVIDIA GeForce RTX 3090", 24.0)] * 2
    assert hw.parse_nvidia_smi("NVIDIA A100-SXM4-40GB, 40960\n") == [("NVIDIA A100-SXM4-40GB", 40.0)]
    assert hw.parse_nvidia_smi("") == [] and hw.parse_nvidia_smi("No devices were found") == [] and hw.parse_nvidia_smi("x, N/A") == []


def test_ollama_list_parser():
    out = ("NAME                    ID              SIZE      MODIFIED\nqwen3.5:9b              6488c96fa5fa    6.6 GB    3 months ago\n"
           "qwen3-embedding:0.6b    ac6da0dfba84    639 MB    3 months ago\n")
    assert [(m["name"], m["size"]) for m in hw.parse_ollama_list(out)] == [("qwen3.5:9b", 6_600_000_000), ("qwen3-embedding:0.6b", 639_000_000)]
    assert hw.parse_ollama_list("") == []


@pytest.mark.parametrize("state,expect", [("Running", "running"), ("NeedsLogin", "needs-login"), ("Stopped", "stopped"), ("Weird", "")])
def test_tailscale_status_parser(state, expect):
    s, ip = hw.parse_tailscale_status(json.dumps({"BackendState": state, "Self": {"TailscaleIPs": ["100.64.0.9", "fd7a::1"]}}))
    assert s == expect and ip == "100.64.0.9"
    assert hw.parse_tailscale_status("not json") == ("", "")


# ---------------------------------------------------------------- the recommendation
@pytest.mark.parametrize("gpus,ram,tier", [
    ([], 32, "Minimum"), ([("x", 4.0)], 32, "Minimum"), ([("x", 6.0)], 32, "Minimum"), ([("x", 8.0)], 32, "Recommended"),
    ([("x", 12.0)], 32, "Recommended"), ([("x", 16.0)], 32, "High-end"), ([("x", 24.0)], 64, "High-end"), ([("x", 24.0)], 16, "Recommended"),
    ([("x", 12.0)], 8, "Minimum"), ([("x", 40.0)], 128, "High-end"),
])
def test_recommendation(gpus, ram, tier):
    got, why = plan.recommend(H(gpus=gpus, ram_gb=ram), reg)
    assert got == tier and why


def test_recommendation_explains_amd_is_not_supported_yet():
    tier, why = plan.recommend(H(gpus=[], other_gpu="Advanced Micro Devices [AMD/ATI] Navi 31"), reg)
    assert tier == "Minimum" and "AMD" in " ".join(why) and "not supported yet" in " ".join(why)


# ---------------------------------------------------------------- the registry the installer reads
def test_tiers_only_use_real_pinned_models():
    for t, spec in reg.TIERS.items():
        assert spec["llm"] in reg.LLMS and set(spec["models"]) <= set(reg.MODELS), t
    for name, m in reg.MODELS.items():
        for f in m["files"]:
            assert f["size"] > 0 and f["url"].startswith("https://") and ("sha256" in f or "git_sha1" in f) and "@" not in str(f.get("sha256", "")), (name, f["path"])
    for n, l in reg.LLMS.items():
        assert len(l["weights"]) == 64 and l["license"] == "Apache-2.0" and l["size"] > 1e9
    assert reg.TIER_ORDER == ("Minimum", "Recommended", "High-end")


def test_each_tier_has_one_variant_of_each_choice_and_no_sam_small():
    assert "sam2-small" not in reg.MODELS
    for t, spec in reg.TIERS.items():
        groups = [reg.MODELS[n].get("group") for n in spec["models"]]
        assert groups.count("selection") == 1 and groups.count("matte") == 1, t
    assert reg.TIERS["Minimum"]["models"][:2] == ["sam2-tiny", "birefnet-lite"] and "sam2-large" in reg.TIERS["High-end"]["models"]


def test_the_sam_configs_exist_in_the_installed_library():
    sam2 = pytest.importorskip("sam2")
    for n, m in reg.MODELS.items():
        if m.get("sam_config"):
            assert (Path(sam2.__file__).parent / m["sam_config"]).is_file(), n


def test_installer_reads_the_registry_without_httpx(tmp_path):
    code = ("import sys; sys.modules['httpx'] = None; import importlib.util as u; "
            f"s = u.spec_from_file_location('r', {str(ROOT / 'backend/veil/vision/registry.py')!r}); m = u.module_from_spec(s); s.loader.exec_module(m); print(len(m.MODELS))")
    assert int(subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True).stdout) >= 8


# ---------------------------------------------------------------- the plan
def fake_models(tmp_path, names):
    for n in names:
        for f in reg.MODELS[n]["files"]:
            p = tmp_path / f["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as fh:
                fh.truncate(f["size"])


def test_plan_lists_exactly_the_tier_models_with_registry_sizes(tmp_path):
    for tier in reg.TIER_ORDER:
        c = plan.Choices(tier=tier, models_dir=str(tmp_path / tier), llm="ollama")
        p = plan.build(H(gpus=[("x", 24.0)], ram_gb=64), reg, c)
        models = [i for i in p.items if i.kind == "model"]
        assert [i.key for i in models] == reg.TIERS[tier]["models"]
        assert all(i.size == sum(f["size"] for f in reg.MODELS[i.key]["files"]) for i in models)
        assert next(i for i in p.items if i.kind == "llm").key == reg.TIERS[tier]["llm"]
        assert all(i.license and i.source for i in p.items)


def test_plan_does_not_count_what_is_already_there(tmp_path):
    fake_models(tmp_path, reg.TIERS["Recommended"]["models"])
    c = plan.Choices(tier="Recommended", models_dir=str(tmp_path), llm="none")
    h = H()
    p = plan.build(h, reg, c)
    assert all(i.have for i in p.items if i.kind == "model")
    assert abs(p.download_total - (plan.LIBS_GPU["download"] + plan.MEDIAPIPE)) < 1


def test_plan_sdxl_only_with_an_nvidia_card_and_when_asked(tmp_path):
    c = plan.Choices(tier="Recommended", sdxl=True, models_dir=str(tmp_path), llm="none")
    assert "sdxl-inpaint" in [i.key for i in plan.build(H(), reg, c).items]
    assert "sdxl-inpaint" not in [i.key for i in plan.build(H(gpus=[]), reg, plan.Choices(tier="Minimum", sdxl=True, cpu=True, models_dir=str(tmp_path), llm="none")).items]
    assert "sdxl-inpaint" not in [i.key for i in plan.build(H(), reg, plan.Choices(tier="Recommended", models_dir=str(tmp_path), llm="none")).items]


def test_plan_reuses_an_existing_ollama_and_model(tmp_path):
    c = plan.Choices(tier="Recommended", models_dir=str(tmp_path), llm="ollama")
    fresh = plan.build(H(), reg, c)
    assert any(i.kind == "ollama" for i in fresh.items) and not next(i for i in fresh.items if i.kind == "llm").have
    have = plan.build(H(ollama_path="/x/ollama", ollama_models=[{"name": "qwen3.5:9b", "size": 1, "digest": "d"}]), reg, c)
    assert not any(i.kind == "ollama" for i in have.items) and next(i for i in have.items if i.kind == "llm").have
    assert have.download_total < fresh.download_total - 6e9


def test_plan_tailscale_item_and_sudo_marks(tmp_path):
    c = plan.Choices(tier="Recommended", models_dir=str(tmp_path), llm="none", tailscale=True)
    p = plan.build(H(has_venv=False), reg, c)
    assert next(i for i in p.items if i.kind == "tailscale").need_sudo and next(i for i in p.items if i.kind == "system").need_sudo
    assert "python3-venv" in next(i for i in p.items if i.kind == "system").key
    assert not any(i.kind == "tailscale" for i in plan.build(H(tailscale_state="running", tailscale_path="/usr/bin/tailscale"), reg, c).items)
    assert not any(i.kind == "system" for i in plan.build(H(), reg, c).items)
    other = plan.build(H(distro_id="fedora", has_apt=False, has_curl=False), reg, c)            # not Debian: the owner is told to install them
    assert "curl" in next(i for i in other.items if i.kind == "system").key and "yourself" in next(i for i in other.items if i.kind == "system").name


def test_plan_needs_zstd_only_when_it_will_unpack_ollama(tmp_path):
    c = plan.Choices(tier="Minimum", models_dir=str(tmp_path), llm="ollama")
    assert "zstd" in next(i for i in plan.build(H(has_zstd=False), reg, c).items if i.kind == "system").key
    assert not any(i.kind == "system" for i in plan.build(H(has_zstd=False, ollama_path="/x"), reg, c).items)
    assert not any(i.kind == "system" for i in plan.build(H(has_zstd=False), reg, plan.Choices(tier="Minimum", models_dir=str(tmp_path), llm="none")).items)


def test_cpu_build_is_much_smaller_and_disk_warnings_work(tmp_path):
    c = plan.Choices(tier="Minimum", cpu=True, models_dir=str(tmp_path), llm="none")
    cpu = plan.build(H(gpus=[]), reg, c)
    gpu = plan.build(H(), reg, plan.Choices(tier="Minimum", models_dir=str(tmp_path), llm="none"))
    assert cpu.download_total < gpu.download_total - 2.5e9
    tight = plan.build(H(disk_free_gb=3.0), reg, plan.Choices(tier="High-end", sdxl=True, models_dir=str(tmp_path), llm="ollama"))
    assert not tight.disk_ok and any("GB free" in w for w in tight.warnings)
    assert plan.build(H(gpus=[("x", 4.0)]), reg, plan.Choices(tier="High-end", models_dir=str(tmp_path), llm="none")).warnings


def test_human_sizes():
    assert plan.human(0) == "—" and plan.human(232589) == "233 KB" and plan.human(324e6) == "324 MB" and plan.human(6.9e9) == "6.9 GB"


# ---------------------------------------------------------------- verified downloads
class Handler(http.server.BaseHTTPRequestHandler):
    files: dict = {}
    ranges = True
    hits: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        Handler.hits.append((self.path, self.headers.get("Range")))
        data = Handler.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        if rng and Handler.ranges:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            body = data[start:]
        else:
            self.send_response(200)
            body = data
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    Handler.files, Handler.ranges, Handler.hits = {}, True, []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", Handler
    srv.shutdown()


DATA = os.urandom(300_000)
SHA = hashlib.sha256(DATA).hexdigest()
quiet = lambda *a, **k: None


def test_download_verifies_and_keeps_the_file(server, tmp_path):
    url, h = server
    h.files["/a.bin"] = DATA
    net.fetch(url + "/a.bin", tmp_path / "d" / "a.bin", len(DATA), SHA, show=quiet)
    assert (tmp_path / "d" / "a.bin").read_bytes() == DATA and not list(tmp_path.rglob("*.part"))


def test_a_file_that_is_already_right_is_not_downloaded_again(server, tmp_path):
    url, h = server
    h.files["/a.bin"] = DATA
    (tmp_path / "a.bin").write_bytes(DATA)
    net.fetch(url + "/a.bin", tmp_path / "a.bin", len(DATA), SHA, show=quiet)
    assert h.hits == []


def test_download_resumes_a_partial_file(server, tmp_path):
    url, h = server
    h.files["/a.bin"] = DATA
    (tmp_path / "a.bin.part").write_bytes(DATA[:100_000])
    net.fetch(url + "/a.bin", tmp_path / "a.bin", len(DATA), SHA, show=quiet)
    assert (tmp_path / "a.bin").read_bytes() == DATA and h.hits[0][1] == "bytes=100000-"


def test_download_restarts_when_the_server_ignores_resume(server, tmp_path):
    url, h = server
    h.files["/a.bin"], h.ranges = DATA, False
    (tmp_path / "a.bin.part").write_bytes(b"garbage" * 1000)
    net.fetch(url + "/a.bin", tmp_path / "a.bin", len(DATA), SHA, show=quiet)
    assert (tmp_path / "a.bin").read_bytes() == DATA


def test_a_wrong_checksum_deletes_the_file_and_is_never_used(server, tmp_path):
    url, h = server
    h.files["/a.bin"] = DATA[:-1] + b"X"                       # same size, one byte different
    with pytest.raises(net.DownloadError, match="checksum"):
        net.fetch(url + "/a.bin", tmp_path / "a.bin", len(DATA), SHA, show=quiet)
    assert not (tmp_path / "a.bin").exists() and not (tmp_path / "a.bin.part").exists()


def test_a_wrong_size_is_refused(server, tmp_path):
    url, h = server
    h.files["/a.bin"] = DATA[:1000]
    with pytest.raises(net.DownloadError):
        net.fetch(url + "/a.bin", tmp_path / "a.bin", len(DATA), SHA, show=quiet, retries=1)
    assert not (tmp_path / "a.bin").exists()


def test_a_corrupt_existing_file_is_replaced(server, tmp_path):
    url, h = server
    h.files["/a.bin"] = DATA
    (tmp_path / "a.bin").write_bytes(b"x" * len(DATA))
    net.fetch(url + "/a.bin", tmp_path / "a.bin", len(DATA), SHA, show=quiet)
    assert (tmp_path / "a.bin").read_bytes() == DATA


def test_missing_file_and_missing_checksum(server, tmp_path):
    url, h = server
    with pytest.raises(net.DownloadError, match="404"):
        net.fetch(url + "/nope", tmp_path / "n", 10, "0" * 64, show=quiet)
    h.files["/a.bin"] = DATA
    with pytest.raises(net.DownloadError, match="No checksum"):
        net.fetch(url + "/a.bin", tmp_path / "a.bin", len(DATA), None, None, show=quiet)
    assert not (tmp_path / "a.bin").exists() and h.hits == [("/nope", None)]


def test_git_sha1_files_are_checked_too(server, tmp_path):
    url, h = server
    h.files["/c.json"] = b'{"a": 1}'
    sha1 = hashlib.sha1(b"blob 8\0" + b'{"a": 1}').hexdigest()
    net.fetch(url + "/c.json", tmp_path / "c.json", 8, git_sha1=sha1, show=quiet)
    with pytest.raises(net.DownloadError, match="checksum"):
        net.fetch(url + "/c.json", tmp_path / "bad.json", 8, git_sha1="0" * 40, show=quiet)


def test_fetch_model_downloads_every_file_and_checks_again(server, tmp_path):
    url, h = server
    one, two = os.urandom(5000), os.urandom(7000)
    h.files.update({"/1": one, "/2": two})
    fake = type("R", (), {"MODELS": {"m": {"files": [
        {"path": "m/1.bin", "size": 5000, "sha256": hashlib.sha256(one).hexdigest(), "url": url + "/1"},
        {"path": "m/2.bin", "size": 7000, "sha256": hashlib.sha256(two).hexdigest(), "url": url + "/2"}]}},
        "file_ok": staticmethod(lambda root, f, deep=True: reg.file_ok(root, f, deep))})
    net.fetch_model(fake, tmp_path, "m", show=quiet)
    assert (tmp_path / "m/1.bin").read_bytes() == one and (tmp_path / "m/2.bin").read_bytes() == two


# ---------------------------------------------------------------- the Runner: nothing runs without agreement
def runner(**kw):
    out = []
    return system.Runner(say=out.append, log_path=kw.pop("log", Path(os.devnull)), **kw), out


def test_dry_run_runs_nothing(tmp_path):
    r, out = runner(dry_run=True)
    marker = tmp_path / "ran"
    assert r.run(["touch", str(marker)], sudo=True, why="test") is None
    assert not marker.exists() and any("touch" in x for x in out) and any("sudo" in x for x in out)


def test_a_sudo_step_needs_a_yes(tmp_path):
    marker = tmp_path / "ran"
    r, _ = runner(ask=lambda q, d: False)
    with pytest.raises(system.StepError, match="administrator"):
        r.run(["touch", str(marker)], sudo=True)
    assert not marker.exists()
    r2, _ = runner(no_sudo=True, assume_yes=True)
    with pytest.raises(system.StepError, match="--no-sudo"):
        r2.run(["touch", str(marker)], sudo=True)
    assert not marker.exists()


def test_a_root_user_runs_without_sudo_prefix(tmp_path):
    r, out = runner(is_root=True)
    r.run(["true"], sudo=True)
    assert out[-1].strip() == "$ true"


def test_a_failing_step_explains_itself(tmp_path):
    r, _ = runner(log=tmp_path / "log.txt")
    with pytest.raises(system.StepError, match="exit code 3"):
        r.run(["sh", "-c", "echo boom; exit 3"])
    assert "boom" in (tmp_path / "log.txt").read_text()


def test_installer_never_asks_for_or_passes_a_password():
    src = (ROOT / "installer" / "system.py").read_text() + (ROOT / "installer" / "cli.py").read_text()
    assert "getpass" not in src and "sudo -S" not in src and "-S" not in "".join(l for l in src.splitlines() if "sudo" in l and "[" in l and "\"-S\"" in l)


# ---------------------------------------------------------------- Ollama (no sudo) and its archive
def make_archive(tmp_path, members):
    tar = tmp_path / "o.tar"
    with tarfile.open(tar, "w") as t:
        for name, data in members:
            src = tmp_path / "src.tmp"
            src.write_bytes(data)
            t.add(src, arcname=name)
    subprocess.run(["zstd", "-q", "-f", str(tar), "-o", str(tmp_path / "o.tar.zst")], check=True)
    return (tmp_path / "o.tar.zst").read_bytes()


zstd = pytest.mark.skipif(not shutil.which("zstd"), reason="zstd is not installed")


def test_unsafe_archive_paths_are_spotted():
    assert system.safe_members(["bin/ollama", "lib/ollama/a.so", "/etc/passwd", "../evil", "a/../../b"]) == ["/etc/passwd", "../evil", "a/../../b"]
    assert system.safe_members(["bin/ollama", "lib/ollama/libx..so"]) == []


@zstd
def test_ollama_installs_into_the_users_folder_only(server, tmp_path, monkeypatch):
    url, h = server
    blob = make_archive(tmp_path, [("bin/ollama", b"#!/bin/sh\necho hi\n"), ("lib/ollama/x.so", b"x")])
    h.files["/o.tar.zst"] = blob
    monkeypatch.setattr(plan, "OLLAMA", {**plan.OLLAMA, "url": url + "/o.tar.zst", "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
    r, _ = runner()
    prefix = tmp_path / "home" / ".local"
    exe = system.install_ollama(r, H(), prefix=prefix)
    assert exe == prefix / "bin" / "ollama" and exe.read_text().startswith("#!/bin/sh") and (prefix / "lib/ollama/x.so").exists()
    assert not (prefix / "share/veil-install/ollama-linux-amd64.tar.zst").exists()           # the archive is cleaned up
    assert system.install_ollama(r, H(), prefix=prefix) == exe                                  # second run keeps it


@zstd
def test_an_archive_with_paths_outside_its_folder_is_refused(server, tmp_path, monkeypatch):
    url, h = server
    blob = make_archive(tmp_path, [("bin/ollama", b"x"), ("../escaped.txt", b"owned")])
    h.files["/o.tar.zst"] = blob
    monkeypatch.setattr(plan, "OLLAMA", {**plan.OLLAMA, "url": url + "/o.tar.zst", "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
    r, _ = runner()
    prefix = tmp_path / "home" / ".local"
    with pytest.raises(system.StepError, match="outside"):
        system.install_ollama(r, H(), prefix=prefix)
    assert not (tmp_path / "home" / "escaped.txt").exists() and not (prefix / "bin" / "ollama").exists()


@zstd
def test_an_archive_that_does_not_match_its_checksum_is_never_unpacked(server, tmp_path, monkeypatch):
    url, h = server
    blob = make_archive(tmp_path, [("bin/ollama", b"x")])
    h.files["/o.tar.zst"] = blob
    monkeypatch.setattr(plan, "OLLAMA", {**plan.OLLAMA, "url": url + "/o.tar.zst", "size": len(blob), "sha256": "0" * 64})
    r, _ = runner()
    with pytest.raises(net.DownloadError, match="checksum"):
        system.install_ollama(r, H(), prefix=tmp_path / "home" / ".local")
    assert not (tmp_path / "home" / ".local" / "bin").exists()


def test_the_ollama_pin_is_a_real_release_digest():
    o = plan.OLLAMA
    assert len(o["sha256"]) == 64 and o["url"].startswith("https://github.com/ollama/ollama/releases/download/" + o["version"] + "/") and o["size"] > 1e9


def test_ollama_unit_runs_as_the_user():
    t = system.ollama_unit_text(Path("/home/u/.local/bin/ollama"))
    assert "ExecStart=/home/u/.local/bin/ollama serve" in t and "WantedBy=default.target" in t and "User=" not in t and "multi-user" not in t


def test_no_user_systemd_means_the_launcher_starts_ollama(tmp_path):
    r, _ = runner(dry_run=True)
    assert system.ollama_service(r, H(systemd_user=False), Path("/x/ollama")) is False


# ---------------------------------------------------------------- the launcher
@pytest.mark.parametrize("name", ["plain", "with space", "it's", 'dq"uote', "dollar$HOME", "semi;colon", "back`tick`"])
def test_launcher_survives_awkward_folder_names(tmp_path, name):
    d = tmp_path / name
    (d / "venv" / "bin").mkdir(parents=True)
    (d / "venv" / "bin" / "python").write_text('#!/bin/sh\necho "MODELS=$VEIL_MODELS_DIR DATA=$VEIL_DATA_DIR ARGS=$*"\n')
    (d / "venv" / "bin" / "python").chmod(0o755)
    text = system.launcher_text(d / "venv", d / "m", d / "d", None, False)
    script = d / "veil"
    script.write_text(text)
    assert subprocess.run(["sh", "-n", str(script)]).returncode == 0
    out = subprocess.run(["sh", str(script), "--port", "1"], capture_output=True, text=True).stdout
    assert out.strip() == f"MODELS={d / 'm'} DATA={d / 'd'} ARGS=-m veil --port 1"


def test_launcher_starts_user_level_ollama_only_when_asked():
    base = Path("/h/venv")
    assert "serve" not in system.launcher_text(base, Path("/m"), Path("/d"), None, False)
    assert "serve" not in system.launcher_text(base, Path("/m"), Path("/d"), Path("/h/ollama"), False)
    t = system.launcher_text(base, Path("/m"), Path("/d"), Path("/h/ollama"), True)
    assert "/h/ollama serve" in t and "nohup" in t


def test_manual_help_for_fedora_and_arch():
    f = system.manual_prereq_help(H(distro_id="fedora", has_apt=False), ["curl"])
    a = system.manual_prereq_help(H(distro_id="arch", has_apt=False), ["curl"])
    assert "sudo dnf install" in f and "zstd" in f and "sudo pacman -S" in a and "zstd" in a and "not Debian" in f


# ---------------------------------------------------------------- whole program, in a throwaway home
def run_installer(tmp_path, *args, stdin=""):
    env = {**os.environ, "HOME": str(tmp_path / "home"), "XDG_CONFIG_HOME": "", "VEIL_PORT": "5999"}
    (tmp_path / "home").mkdir(exist_ok=True)
    return subprocess.run([str(ROOT / "install.sh"), *args], capture_output=True, text=True, env=env, input=stdin, timeout=120)


def snapshot(p: Path):
    return sorted(str(x.relative_to(p)) for x in p.rglob("*"))


def test_dry_run_shows_the_plan_and_changes_nothing(tmp_path):
    r = run_installer(tmp_path, "--dry-run", "--yes", "--tier", "Recommended", "--llm", "ollama", "--tailscale", "--sdxl",
                      "--models-dir", str(tmp_path / "models"), "--data-dir", str(tmp_path / "data"))
    assert r.returncode == 0, r.stdout + r.stderr
    for must in ("Everything this will download or install", "Segment Anything 2.1 (base+)", "Apache-2.0", "CreativeML Open RAIL++-M", "Download total:",
                 "Dry run finished. Nothing was changed or downloaded."):
        assert must in r.stdout, must
    assert snapshot(tmp_path / "home") == [] and not (tmp_path / "models").exists() and not (tmp_path / "data").exists()


def test_dry_run_for_each_tier_and_cpu(tmp_path):
    for extra in (["--tier", "Minimum"], ["--tier", "High-end"], ["--tier", "Minimum", "--cpu"], ["--tier", "Minimum", "--llm", "none", "--no-tailscale"]):
        r = run_installer(tmp_path, "--dry-run", "--yes", *extra, "--models-dir", str(tmp_path / "m"), "--data-dir", str(tmp_path / "d"))
        assert r.returncode == 0, (extra, r.stdout[-600:], r.stderr[-300:])
    cpu = run_installer(tmp_path, "--dry-run", "--yes", "--tier", "Minimum", "--cpu", "--models-dir", str(tmp_path / "m"), "--data-dir", str(tmp_path / "d"))
    assert "CPU build" in cpu.stdout and "download.pytorch.org/whl/cpu" in cpu.stdout and "SAM" not in cpu.stdout.replace("Segment", "")
    assert "tiny" in cpu.stdout and "BiRefNet lite" in cpu.stdout


def test_a_too_small_disk_is_refused_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(hw, "detect", lambda target, run=None: H(disk_free_gb=2.0))
    monkeypatch.setattr(system, "libs_ready", lambda py: False)
    rc = cli.main(["--yes", "--tier", "High-end", "--llm", "none", "--models-dir", str(tmp_path / "m"), "--data-dir", str(tmp_path / "d")])
    assert rc == 1 and not (tmp_path / "m").exists()


def test_unsupported_machines_are_told_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hw, "detect", lambda target, run=None: H(arch="aarch64"))
    assert cli.main(["--dry-run", "--yes"]) == 2 and "x86_64" in capsys.readouterr().out
    monkeypatch.setattr(hw, "detect", lambda target, run=None: H(python=(3, 8, 1)))
    assert cli.main(["--dry-run", "--yes"]) == 2


def test_uninstall_removes_the_install_but_never_the_photos(tmp_path):
    home = tmp_path / "home"
    data, models, venv = tmp_path / "data", tmp_path / "models", tmp_path / "v"
    (data / "photos" / "abc").mkdir(parents=True); (data / "photos" / "abc" / "original.jpg").write_bytes(b"precious")
    (models / "sam2").mkdir(parents=True); (models / "sam2" / "x.pt").write_bytes(b"m")
    (venv / "bin").mkdir(parents=True)
    launcher = home / ".local" / "bin" / "veil"; launcher.parent.mkdir(parents=True); launcher.write_text("x")
    rec = home / ".config" / "veil" / "install.json"; rec.parent.mkdir(parents=True)
    rec.write_text(json.dumps({"venv": str(venv), "launcher": str(launcher), "desktop_entry": "", "models_dir": str(models), "data_dir": str(data)}))
    r = run_installer(tmp_path, "--uninstall", "--yes")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not venv.exists() and not launcher.exists() and not rec.exists()
    assert (data / "photos" / "abc" / "original.jpg").read_bytes() == b"precious"            # photos are never touched
    assert (models / "sam2" / "x.pt").exists()                                                  # models are kept unless you say so
    assert "NEVER touched" in r.stdout


def test_uninstall_with_nothing_installed_is_harmless(tmp_path):
    r = run_installer(tmp_path, "--uninstall", "--yes")
    assert r.returncode == 0 and "nothing to remove" in r.stdout


def test_uninstall_dry_run_removes_nothing(tmp_path):
    home = tmp_path / "home"; venv = tmp_path / "v"; (venv / "bin").mkdir(parents=True)
    rec = home / ".config" / "veil" / "install.json"; rec.parent.mkdir(parents=True)
    rec.write_text(json.dumps({"venv": str(venv), "launcher": "", "desktop_entry": "", "models_dir": "", "data_dir": str(tmp_path / "data")}))
    assert run_installer(tmp_path, "--uninstall", "--dry-run").returncode == 0 and venv.exists() and rec.exists()


def test_install_sh_explains_a_missing_python(tmp_path):
    env = {"PATH": "/nonexistent", "HOME": str(tmp_path)}
    r = subprocess.run(["/bin/sh", str(ROOT / "install.sh")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "python3" in r.stdout and "Then run ./install.sh again." in r.stdout


def test_the_readme_covers_the_installer_and_amd_plans():
    t = (ROOT / "README.md").read_text()
    for must in ("./install.sh", "AMD", "planned", "Fedora", "Arch", "--dry-run", "uninstall"):
        assert must in t, must


# ---------------------------------------------------------------- Veil is told which model was installed
def test_the_installed_model_becomes_veils_active_model(tmp_path):
    from veil.ai.settings import AISettings
    sys.path.insert(0, str(ROOT / "backend"))
    r, _ = runner()
    assert system.write_ai_settings(r, tmp_path / "data", "qwen3.5:2b") == ""
    cfg = AISettings(tmp_path / "data").active()
    assert cfg["model"] == "qwen3.5:2b" and cfg["kind"] == "ollama" and cfg["id"] == "local-ollama"
    assert oct((tmp_path / "data" / "ai.json").stat().st_mode & 0o777) == "0o600"


def test_existing_model_settings_are_never_overwritten(tmp_path):
    (tmp_path / "data").mkdir()
    mine = {"active": "mine", "providers": [{"id": "mine", "name": "x", "kind": "ollama", "base_url": "http://127.0.0.1:11434", "model": "llama3", "vision": False}]}
    (tmp_path / "data" / "ai.json").write_text(json.dumps(mine))
    r, _ = runner()
    note = system.write_ai_settings(r, tmp_path / "data", "qwen3.5:2b")
    assert "kept" in note and json.loads((tmp_path / "data" / "ai.json").read_text()) == mine


def test_dry_run_writes_no_settings(tmp_path):
    r, _ = runner(dry_run=True)
    system.write_ai_settings(r, tmp_path / "data", "qwen3.5:2b")
    assert not (tmp_path / "data").exists()


# ---------------------------------------------------------------- worker mode (phase 9)
def worker_choices(**kw):
    return plan.Choices(tier="Recommended", models_dir="/nonexistent-models", data_dir="/nonexistent-data", worker=True, **kw)


def test_worker_plan_has_no_language_model_and_always_has_tailscale():
    p = plan.build(H(tailscale_state=""), reg, worker_choices(llm="ollama", tailscale=False))
    kinds = [i.kind for i in p.items]
    assert "llm" not in kinds and "ollama" not in kinds and "tailscale" in kinds
    assert p.choices.llm == "none" and p.choices.tailscale


def test_worker_plan_skips_zstd_which_only_ollama_needs():
    p = plan.build(H(has_zstd=False, tailscale_state="running"), reg, worker_choices())
    assert not [i for i in p.items if i.kind == "system" and "zstd" in i.key]


def test_worker_questions_never_ask_about_the_language_model_or_phone(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(cli, "ask_choice", lambda q, *a, **k: asked.append(q) or 0)
    monkeypatch.setattr(cli, "ask_yes_no", lambda q, *a, **k: asked.append(q) or True)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    args = cli.parser().parse_args(["--worker", "--models-dir", str(tmp_path / "m"), "--data-dir", str(tmp_path / "d"), "--tier", "Minimum"])
    c = cli.questions(args, H(), reg, tmp_path)
    assert c.worker and c.llm == "none" and c.tailscale
    assert not [q for q in asked if "typed" in q.lower() or "Ollama" in q or "phone" in q.lower()]


def test_the_worker_launcher_runs_the_worker_and_nothing_else(tmp_path):
    text = system.worker_launcher_text(Path("/v/.venv"), Path("/m dir"), Path("/d"))
    assert "-m veil.worker" in text and "exec /v/.venv/bin/python -I" in text and "'/m dir'" in text
    assert "ollama" not in text.lower() and " -m veil " not in text


def test_worker_dry_run_as_root_on_a_rented_machine(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hw, "detect", lambda target, run=None: H(is_root=True, tailscale_state=""))
    monkeypatch.setattr(system, "libs_ready", lambda py: False)
    monkeypatch.setattr(cli, "run_steps", cli._real_run_steps)           # a DRY run only: it prints and changes nothing
    rc = cli.main(["--worker", "--dry-run", "--yes", "--tier", "Recommended", "--models-dir", str(tmp_path / "m"), "--data-dir", str(tmp_path / "d")])
    out = capsys.readouterr().out
    assert rc == 0 and "GPU worker mode" in out and "normal on a rented GPU machine" in out
    assert "Ollama" not in out.split("Everything this will download")[1].split("Download total")[0]
    assert "tailscale up --auth-key" in out and "userspace-networking" in out and "tailscale serve --bg --tcp 5196" in out
    assert "veil-worker" in out and not (tmp_path / "m").exists()


def test_normal_install_still_refuses_root_without_yes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hw, "detect", lambda target, run=None: H(is_root=True))
    assert cli.main(["--dry-run", "--tier", "Minimum", "--models-dir", str(tmp_path / "m"), "--data-dir", str(tmp_path / "d")]) == 2
    assert "administrator (root)" in capsys.readouterr().out


def test_worker_install_does_not_overwrite_the_veil_command_or_record(tmp_path, monkeypatch):
    r = system.Runner(dry_run=False, assume_yes=True, is_root=False, say=lambda *a, **k: None)
    system.write_record(r, {"launcher": "/x/veil", "venv": "/x/v", "data_dir": "/x/d"})
    system.write_record(r, {"worker_launcher": "/x/veil-worker", "venv": "/x/v"})
    rec = system.read_record()
    assert rec["launcher"] == "/x/veil" and rec["worker_launcher"] == "/x/veil-worker"


def test_uninstall_lists_the_worker_command(tmp_path, capsys):
    r = system.Runner(dry_run=False, assume_yes=True, say=lambda *a, **k: None)
    system.write_record(r, {"venv": str(tmp_path / "v"), "worker_launcher": str(tmp_path / "bin" / "veil-worker"), "data_dir": str(tmp_path / "data")})
    assert cli.main(["--uninstall", "--dry-run"]) == 0
    assert "veil-worker" in capsys.readouterr().out
