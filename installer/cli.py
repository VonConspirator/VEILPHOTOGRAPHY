"""Veil installer: asks a few questions, shows every download, then sets everything up. Python standard library only."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import hw as hwmod, net, plan, system

TAILSCALE_COMPUTER = """
  Phone access, step by step (Tailscale makes a private network between your computer and your phone; nothing is opened to the internet):

  On this computer
    1. Tailscale is installed (or was just installed by the step above).
    2. Run:  sudo tailscale up
       It prints a web address that starts with https://login.tailscale.com/ . Open it in your browser.
    3. Sign in (Google, Microsoft, Apple or GitHub, as you prefer). If you have no account, the page creates a free one.
    4. Click "Connect". The terminal command finishes by itself.
"""
TAILSCALE_PHONE = """
  On your phone
    5. Install the "Tailscale" app from the App Store (iPhone) or Play Store (Android).
    6. Open it and sign in with the SAME account as in step 3, then switch the toggle at the top to On.

  In Veil (on this computer)
    7. Start Veil, click "Phone" at the top, tick "Allow phone access", click "Show QR code".
    8. On your phone, open the camera and point it at the code. Tap the link. The phone says it is waiting.
    9. Back on the computer, click "Approve" next to the phone. Done.
"""
TAILSCALE_STEPS = TAILSCALE_COMPUTER + TAILSCALE_PHONE

WORKER_TAILSCALE = """
  Connecting this worker to your Tailscale network (a worker only ever listens on Tailscale, never on the internet or your network):

    1. On this machine run:  sudo tailscale up      (as root, just:  tailscale up)
       It prints an address that starts with https://login.tailscale.com/ . Open it in any browser, on any device.
    2. Sign in with the SAME account you use on your own computer, then click "Connect".
       Rented machine you only reach through a terminal? You can use an auth key instead: in your browser open
       https://login.tailscale.com/admin/settings/keys , click "Generate auth key", tick "Ephemeral", leave "Reusable" off,
       click "Generate", copy the key, and run here:  tailscale up --auth-key=PASTE_THE_KEY_HERE   (treat the key as a password).
    3. Check it:  tailscale ip -4      prints this machine's Tailscale address (it starts with 100.).
    4. Container without /dev/net/tun (common on rented GPU services)? Start Tailscale in userspace mode instead:
         tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &
       then step 1 or 2, then run once:  tailscale serve --bg --tcp 5196 tcp://localhost:5196
"""

LICENSE_SDXL = ("The generative model (Stable Diffusion XL Inpainting) is under the CreativeML Open RAIL++-M license. It is not Apache or MIT: "
                "it has use restrictions that apply to you. In short, you agree not to use it to break the law, to exploit or harm minors, to make or "
                "spread harmful false information, to harass or defame, to discriminate, for medical advice, or for law-enforcement profiling. "
                "Full text: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md")


def say(*a, **k):
    print(*a, **k)
    sys.stdout.flush()


def ask_yes_no(q: str, default: bool = True) -> bool:
    if not sys.stdin.isatty():
        return default
    hint = "Y/n" if default else "y/N"
    while True:
        a = input(f"{q} [{hint}] ").strip().lower()
        if not a:
            return default
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False


def ask_choice(q: str, options: list[tuple[str, str]], default: int = 0) -> int:
    say(q)
    for i, (title, about) in enumerate(options, 1):
        say(f"  {i}) {title}" + (f"\n       {about}" if about else ""))
    if not sys.stdin.isatty():
        return default
    while True:
        a = input(f"Your choice [{default + 1}]: ").strip()
        if not a:
            return default
        if a.isdigit() and 1 <= int(a) <= len(options):
            return int(a) - 1


def ask_text(q: str, default: str) -> str:
    if not sys.stdin.isatty():
        return default
    a = input(f"{q} [{default}]: ").strip()
    return a or default


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="install.sh", description="Set up Veil on this computer.")
    p.add_argument("--yes", "-y", action="store_true", help="accept the recommended answers and run without asking (for scripts)")
    p.add_argument("--dry-run", action="store_true", help="show everything that would happen; change and download nothing")
    p.add_argument("--tier", choices=["Minimum", "Recommended", "High-end"], help="choose the set instead of the recommended one")
    p.add_argument("--sdxl", dest="sdxl", action="store_true", default=None, help="include the optional generative model (6.9 GB)")
    p.add_argument("--no-sdxl", dest="sdxl", action="store_false")
    p.add_argument("--llm", choices=["ollama", "online", "none"], help="how typed commands are understood")
    p.add_argument("--tailscale", dest="tailscale", action="store_true", default=None, help="set up Tailscale for phone access")
    p.add_argument("--no-tailscale", dest="tailscale", action="store_false")
    p.add_argument("--cpu", action="store_true", help="install the processor-only build even if an NVIDIA card is present")
    p.add_argument("--models-dir", help="where models are stored (default: models/ next to the installer)")
    p.add_argument("--data-dir", help="where photos and settings are stored (default: data/ next to the installer)")
    p.add_argument("--no-sudo", action="store_true", help="never run administrator commands; print them for you to run")
    p.add_argument("--desktop-entry", dest="desktop", action="store_true", default=None, help="add Veil to the application menu")
    p.add_argument("--no-desktop-entry", dest="desktop", action="store_false")
    p.add_argument("--force", action="store_true", help="continue even if the disk looks too small")
    p.add_argument("--unlocked", action="store_true", help="install the newest compatible libraries WITHOUT the hash-checked list (only for a Python the list does not cover)")
    p.add_argument("--worker", action="store_true", help="set this computer up as a GPU worker for your other computer (no Ollama, Tailscale required)")
    p.add_argument("--uninstall", action="store_true", help="remove what the installer set up (never your photos)")
    return p


def show_hardware(h) -> None:
    say("\nWhat I found on this computer")
    say(f"  System:        {h.distro_id or h.os_name} {h.codename}  ({h.arch}), Python {'.'.join(map(str, h.python))}")
    say("  Graphics card: " + ("; ".join(f"{n}, {v:.0f} GB video memory" for n, v in h.gpus) if h.gpus else
                              ("none usable" + (f" ({h.other_gpu}; AMD/Intel graphics are not supported yet)" if h.other_gpu else " (no NVIDIA card)"))))
    say(f"  Memory:        {h.ram_gb:.0f} GB     Free disk space here: {h.disk_free_gb:.0f} GB")
    say("  Ollama:        " + ((h.ollama_path + (", running" if h.ollama_running else ", not running")) if h.ollama_path else "not installed")
        + (f"; {len(h.ollama_models)} model(s): " + ", ".join(m['name'] for m in h.ollama_models[:6]) + ("…" if len(h.ollama_models) > 6 else "") if h.ollama_models else ""))
    say("  Tailscale:     " + ((h.tailscale_state or "installed") + (f" ({h.tailscale_ip})" if h.tailscale_ip else "") if h.tailscale_path else "not installed"))


def show_plan(p, reg) -> None:
    say("\nEverything this will download or install")
    say(f"  {'What':<58}{'Size':>9}  License")
    for i in p.items:
        mark = "  (already there)" if i.have else ("  (needs your password)" if i.need_sudo else "")
        say(f"  {i.name[:57]:<58}{plan.human(i.size):>9}  {i.license[:40]}{mark}")
        say(f"      from {i.source}. {i.note}")
    say(f"\n  Download total: {plan.human(p.download_total)}.   Free space needed while installing: about {plan.human(p.disk_needed)} (you have {plan.human(p.disk_free)}).")
    for w in p.warnings:
        say(f"  ! {w}")


def questions(args, h, reg, repo: Path) -> plan.Choices:
    c = plan.Choices(models_dir=args.models_dir or str(repo / "models"), data_dir=args.data_dir or str(repo / "data"))
    c.cpu = bool(args.cpu) or not h.gpus
    c.worker = bool(args.worker)
    c.unlocked = bool(args.unlocked)
    interactive = sys.stdin.isatty() and not args.yes
    say("\nA few questions")
    # 1. what for
    if args.sdxl is not None:
        c.sdxl = args.sdxl
    elif interactive and h.gpus:
        i = ask_choice("\n1. What will you mainly do with Veil?", [
            ("Touch-ups: skin, teeth, eyes, hair colour, makeup, removing objects and backgrounds, hiding faces", "No extra download."),
            ("Everything above, plus generated faces, hairstyle changes, makeup removal and facial hair", "Adds a 6.9 GB generative model with use restrictions."),
        ])
        c.sdxl = i == 1
    # 2. typed commands (a worker never runs the language model)
    if c.worker:
        c.llm = "none"
    elif args.llm:
        c.llm = args.llm
    elif interactive:
        i = ask_choice("\n2. Veil lets you type what you want ('make the skin smoother'). Which model should read your words?", [
            ("A model on this computer (Ollama)", "Private: nothing leaves your computer. I set it up for you."),
            ("An online service (OpenAI-compatible or Anthropic)", "You add your own key later inside Veil. Only your words are sent, never the photo, unless you allow it."),
            ("None, I will use the buttons", ""),
        ])
        c.llm = ["ollama", "online", "none"][i]
    # 3. phone (a worker always needs Tailscale)
    if c.worker:
        c.tailscale = True
    elif args.tailscale is not None:
        c.tailscale = args.tailscale
    elif interactive:
        c.tailscale = ask_yes_no("\n3. Do you want to use Veil from your phone? (sets up Tailscale, a free private network)", False)
    # 4. where
    if interactive and not (args.models_dir and args.data_dir):
        say("\n4. Where should things be kept? Models are large; settings are yours." if c.worker else "\n4. Where should things be kept? Models are large; photos and settings are yours.")
        c.models_dir = args.models_dir or ask_text("   Models folder", c.models_dir)
        c.data_dir = args.data_dir or ask_text("   Photos and settings folder", c.data_dir)
    # tier
    tier, why = plan.recommend(h, reg)
    if args.tier:
        c.tier = args.tier
    elif interactive:
        say("\nMy recommendation: " + tier)
        for w in why:
            say("  " + w)
        names = list(reg.TIER_ORDER)
        i = ask_choice("Which set do you want?", [(f"{t}{'  (recommended)' if t == tier else ''}",
                       f"{reg.TIERS[t]['about']} Language model: {reg.TIERS[t]['llm']}.") for t in names], default=names.index(tier))
        c.tier = names[i]
    else:
        c.tier = tier
        say(f"\nRecommended set: {tier}. " + " ".join(why))
    if c.cpu and c.tier != "Minimum" and not args.tier:
        c.tier = "Minimum"
    return c


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = plan.ROOT
    reg = plan.load_registry()
    say("Veil installer" + ("  (GPU worker mode)" if args.worker else "") + ("  (dry run: nothing will be changed)" if args.dry_run else ""))
    if args.uninstall:
        return uninstall(args)
    if sys.platform != "linux":
        say("This installer is for Linux.")
        return 2
    target = args.models_dir or str(repo / "models")
    h = hwmod.detect(target)
    if h.arch not in ("x86_64", "amd64"):
        say(f"This computer is {h.arch}. Veil's installer supports 64-bit Intel/AMD (x86_64) only for now.")
        return 2
    if h.python < (3, 10):
        say(f"Python {'.'.join(map(str, h.python[:2]))} is too old; Veil needs 3.10 or newer.")
        return 2
    if h.is_root and args.worker:
        say("You are running as root, which is normal on a rented GPU machine, so administrator steps run directly without sudo.")
    elif h.is_root and not args.yes:
        say("You are running as the administrator (root). Veil is meant to be installed as your normal user. Please run it without sudo.")
        return 2
    show_hardware(h)
    r = system.Runner(dry_run=args.dry_run, assume_yes=args.yes, no_sudo=args.no_sudo, is_root=h.is_root, ask=ask_yes_no, say=say)
    c = questions(args, h, reg, repo)
    venv = repo / ".venv"
    libs_present = system.libs_ready(venv / "bin" / "python")
    p = plan.build(h, reg, c, libs_present=libs_present)
    if c.sdxl and not c.cpu:
        say("\n" + LICENSE_SDXL)
        if not (args.yes or args.dry_run) and not ask_yes_no("Do you accept these restrictions for the generative model?", False):
            c.sdxl = False
            p = plan.build(h, reg, c, libs_present=libs_present)
            say("  OK, the generative model is left out.")
    show_plan(p, reg)
    if not p.disk_ok and not args.force:
        if args.yes or args.dry_run or not ask_yes_no("\nThere may not be enough disk space. Continue anyway?", False):
            say("\nNot enough free space. Free some up, pick a smaller set, or choose another folder, then run the installer again (or use --force).")
            return 0 if args.dry_run else 1
    if not args.dry_run and not args.yes and not ask_yes_no("\nGo ahead and install this?", True):
        say("Nothing was changed.")
        return 0
    try:
        return run_steps(args, h, reg, c, p, r, repo, venv)
    except (system.StepError, net.DownloadError) as e:
        say(f"\nStopped: {e}")
        say("Nothing is half-installed in a way that matters: run the installer again and it continues where it stopped.")
        return 1


def run_steps(args, h, reg, c, p, r, repo: Path, venv: Path) -> int:
    t = reg.TIERS[c.tier]
    models = Path(c.models_dir).expanduser()
    data = Path(c.data_dir).expanduser()
    notes: list[str] = []
    say("\n1/6  System packages")
    sysitem = next((i for i in p.items if i.kind == "system"), None)
    if sysitem:
        pk = sysitem.key.split()
        if h.debian_family:
            system.apt_packages(r, pk)
        else:
            say(system.manual_prereq_help(h, pk)); return 2
    else:
        say("  Nothing to install.")
    if not r.dry_run and not hwmod._can(sys.executable, "import venv, ensurepip"):
        raise system.StepError("Python's 'venv' is still missing. On Debian/Ubuntu/Mint: sudo apt install python3-venv")
    say("\n2/6  Python environment and libraries")
    py = system.make_venv(r, venv)
    if system.libs_ready(py) and not r.dry_run:
        say("  The libraries are already installed and load correctly.")
    else:
        system.install_libs(r, py, repo, cpu=c.cpu, sdxl=c.sdxl and not c.cpu, unlocked=c.unlocked)
        if not c.unlocked and not r.dry_run:
            main = plan.lock_for(sys.version_info[1], c.cpu, c.sdxl and not c.cpu)[0]
            wrong = system.lock_mismatches(py, main)
            if wrong:
                raise system.StepError("The installed libraries do not match the hash-checked list:\n    " + "\n    ".join(wrong[:8]))
            say(f"  Checked: every library installed is exactly the version pinned in lock/{main.parent.name}/{main.name}.")
    say("\n3/6  Models (each file is checked against its recorded SHA-256)")
    names = list(t["models"]) + (["sdxl-inpaint"] if c.sdxl and not c.cpu else [])
    for n in names:
        m = reg.MODELS[n]
        say(f"  {m['label']}  ({plan.human(sum(f['size'] for f in m['files']))}, {m['license'].split(' (')[0]})")
        if r.dry_run:
            continue
        if reg.installed(models, n, deep=True):
            say("    already there and verified"); continue
        net.fetch_model(reg, models, n)
        say("    downloaded and verified")
    ollama_exe, needs_start = None, False
    started: list = []
    say("\n4/6  Language model")
    if c.worker:
        say("  Not needed: a worker only runs the heavy picture jobs. Typed commands are read on your own computer.")
    elif c.llm == "ollama":
        ollama_exe = Path(h.ollama_path) if h.ollama_path else system.install_ollama(r, h)
        user_level = str(ollama_exe).startswith(str(Path.home()))
        has_service = False
        if not h.ollama_path:
            has_service = system.ollama_service(r, h, ollama_exe)
        needs_start = user_level and not has_service and not h.ollama_running
        note = system.pull_llm(r, h, reg, ollama_exe, t["llm"], started)
        if note:
            notes.append(note)
        for proc in started:
            proc.terminate()
        note = system.write_ai_settings(r, data, t["llm"])
        if note:
            notes.append(note)
    elif c.llm == "online":
        say("  Nothing to install. In Veil, open Model and add your online service and key.")
    else:
        say("  Skipped. Veil works with its buttons.")
    say("\n5/6  " + ("Tailscale (the only way a worker is reached)" if c.worker else "Phone access (Tailscale)"))
    if c.tailscale:
        if h.tailscale_path:
            say("  Tailscale is already installed.")
        elif h.debian_family:
            system.tailscale_install(r, h)
        else:
            say("  This is not Debian/Ubuntu/Mint, so install Tailscale yourself: https://tailscale.com/download/linux")
        if c.worker:
            say("" if h.tailscale_state == "running" else WORKER_TAILSCALE)
        else:
            say(TAILSCALE_PHONE if h.tailscale_state == 'running' else TAILSCALE_STEPS)
        if not r.dry_run and not args.yes and shutil.which("tailscale") and ask_yes_no("Run 'sudo tailscale up' now? (it prints a web address to open)", True):
            r.run(["tailscale", "up"], sudo=True, why="Connecting this computer to your Tailscale account.", check=False)
        if not r.dry_run and shutil.which("tailscale"):
            state, ip = hwmod.parse_tailscale_status(subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True).stdout)
            say("  Tailscale is connected." + (f" This computer's address: {ip}" if ip else "") if state == "running" else
                "  Tailscale is not signed in yet. Do steps 2-4 above whenever you are ready; Veil works on this computer without it.")
    else:
        say("  Skipped. You can run the installer again later and choose phone access.")
    say("\n6/6  Finishing")
    if c.worker:
        system.write_worker_launcher(r, venv, models, data)
    else:
        system.write_launcher(r, venv, models, data, ollama_exe if needs_start else None, needs_start)
    want_desktop = False if c.worker else args.desktop if args.desktop is not None else (ask_yes_no("Add Veil to your application menu?", True) if (sys.stdin.isatty() and not args.yes) else False)
    if want_desktop:
        system.write_desktop(r)
    rec_worker = {"worker_launcher": str(system.WORKER_LAUNCHER)} if c.worker else {}
    system.write_record(r, {**rec_worker, "repo": str(repo), "venv": str(venv), "models_dir": str(models), "data_dir": str(data), "tier": c.tier, "llm": c.llm,
                            "ollama_installed_by_installer": bool(ollama_exe and not h.ollama_path), "ollama_exe": str(ollama_exe or ""),
                            "ollama_unit": str(system.UNIT) if (ollama_exe and not h.ollama_path and h.systemd_user) else "",
                            **({} if c.worker else {"launcher": str(system.LAUNCHER)}), "desktop_entry": str(system.DESKTOP) if want_desktop else "",
                            "model_names": names, "tailscale_installed_by_installer": bool(c.tailscale and not h.tailscale_path)})
    if r.dry_run:
        say("\nDry run finished. Nothing was changed or downloaded.")
        return 0
    say("\nChecking the finished install")
    problems = system.self_check(py, repo, models, names)
    if problems:
        for x in problems:
            say(f"  ! {x}")
        say("The install finished but the check found problems (above). The log is " + str(system.LOG))
        return 1
    say("  Libraries load, every model matches its checksum, and the editing engine is ready.")
    for n in notes:
        say(f"  Note: {n}")
    if c.worker:
        say("\nThe Veil GPU worker is installed.")
        say("  Start it:        veil-worker       (leave it running; it prints the address your own computer should use)")
        if not shutil.which("veil-worker"):
            say(f"                   (if 'veil-worker' is not found, open a new terminal, or run {system.WORKER_LAUNCHER})")
        say("  Pair it:         in a second terminal on this machine run  veil-worker pair  and type the code into Veil on your own computer (GPU worker button).")
        say("  Stop paying:     if this machine is rented, remember to stop or delete it at the provider when you are done.")
        say("  Remove it:  ./install.sh --uninstall   (never touches your photos)")
        return 0
    say("\nVeil is installed.")
    say("  Start it:   veil        (or choose Veil in your application menu)")
    if not shutil.which("veil"):
        say(f"              (if 'veil' is not found, open a new terminal, or run {system.LAUNCHER})")
    say("  Then open:  http://localhost:5195  in your browser. The first time you will create your owner account.")
    say("  Remove it:  ./install.sh --uninstall   (never touches your photos)")
    return 0


def uninstall(args) -> int:
    rec = system.read_record()
    if not rec:
        say("There is no record of an install here, so there is nothing to remove.")
        return 0
    say("This will remove:")
    items = [("Veil's Python environment", rec.get("venv")), ("the 'veil' command", rec.get("launcher")), ("the 'veil-worker' command", rec.get("worker_launcher")),
             ("the menu entry", rec.get("desktop_entry"))]
    for n, pth in items:
        if pth:
            say(f"  - {n}: {pth}")
    say(f"  Your photos and settings in {rec.get('data_dir')} are NEVER touched.")
    if args.dry_run:
        return 0
    if not (args.yes or ask_yes_no("Remove these?", False)):
        return 0
    for _, pth in items:
        if pth:
            q = Path(pth)
            shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
    models = Path(rec.get("models_dir", ""))
    reg = plan.load_registry()
    known = sorted({str(Path(f["path"]).parts[0]) for m in reg.MODELS.values() for f in m["files"]})
    if models.is_dir() and (ask_yes_no(f"Also delete the downloaded models in {models}? (you can download them again)", False)):
        for d in known:
            shutil.rmtree(models / d, ignore_errors=True)
        say("  Models removed (the folder itself and anything else in it were left alone).")
    if rec.get("ollama_installed_by_installer"):
        say(f"  Ollama was installed for you at {rec.get('ollama_exe')}.")
        if ask_yes_no("Remove Ollama too? (its downloaded language models in ~/.ollama are kept)", False):
            if rec.get("ollama_unit"):
                subprocess.run(["systemctl", "--user", "disable", "--now", "ollama.service"], capture_output=True)
                Path(rec["ollama_unit"]).unlink(missing_ok=True)
            Path(rec["ollama_exe"]).unlink(missing_ok=True)
            shutil.rmtree(Path(rec["ollama_exe"]).parent.parent / "lib" / "ollama", ignore_errors=True)
    if rec.get("tailscale_installed_by_installer"):
        say("  Tailscale was installed for you. To remove it: sudo apt remove tailscale")
    system.RECORD.unlink(missing_ok=True)
    say("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
