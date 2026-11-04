"""The first-launch tutorial: what it says, who sees which steps, and that it cannot break the page's strict security rules.
(Its behaviour in a real browser is tested live; here the content and the code's rules are checked.)"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "veil" / "web"
JS = (WEB / "tutorial.js").read_text()
HTML = (WEB / "index.html").read_text()
node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def steps(**opts):
    out = subprocess.run(["node", "-e", "const t=require(process.argv[1]);console.log(JSON.stringify(t.stepsFor(JSON.parse(process.argv[2]))))",
                          str(WEB / "tutorial.js"), json.dumps(opts)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def links():
    out = subprocess.run(["node", "-e", "console.log(JSON.stringify(require(process.argv[1]).LINKS))", str(WEB / "tutorial.js")],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@node
def test_first_run_has_every_step_except_the_gpu_worker():
    ids = [s["id"] for s in steps(remote=False, replay=False)]
    assert "worker" not in ids
    assert ids == ["welcome", "photos", "look", "select", "adjust", "crop", "cleanup", "faces", "typed", "undo", "export", "watermarks",
                   "ts-pc", "ts-phone", "ts-pair", "done"]


@node
def test_help_replays_everything_including_the_gpu_worker():
    ids = [s["id"] for s in steps(remote=False, replay=True)]
    assert ids.index("worker") == ids.index("ts-pair") + 1 and ids[-1] == "done" and len(ids) == 17


@node
def test_a_phone_gets_a_shorter_tour_without_setup_or_management():
    for replay in (False, True):
        ids = [s["id"] for s in steps(remote=True, replay=replay)]
        assert not {"watermarks", "ts-pc", "ts-phone", "ts-pair", "worker"} & set(ids)
        assert ids[0] == "welcome" and ids[-1] == "done" and "export" in ids and len(ids) == 12


@node
def test_the_welcome_wording_is_the_approved_one():
    first = steps(remote=False, replay=False)[0]["body"][0]
    assert first == "Veil edits photos on your own computer. Nothing leaves your computer unless you choose."
    assert "uploaded anywhere" not in JS
    phone = steps(remote=True, replay=False)[0]["body"][0]
    assert "Nothing leaves your computer unless you choose." in phone


@node
def test_the_phone_export_step_says_presets_are_made_at_the_computer():
    export = [s for s in steps(remote=True, replay=False) if s["id"] == "export"][0]
    assert "Presets are made at the computer" in " ".join(export["body"])
    pc = [s for s in steps(remote=False, replay=False) if s["id"] == "export"][0]
    assert "only drawn on exports" in " ".join(pc["body"])


@node
def test_each_tailscale_step_has_links_and_a_what_you_should_see_line():
    by = {s["id"]: s for s in steps(remote=False, replay=False)}
    L = links()
    for sid in ("ts-pc", "ts-phone", "ts-pair"):
        assert by[sid]["see"].startswith("You should see:"), sid
    assert {"computer", "linux"} <= set(by["ts-pc"]["links"])
    assert set(by["ts-phone"]["links"]) == {"ios", "android"}
    assert L["ios"]["url"].startswith("https://apps.apple.com/") and L["android"]["url"].startswith("https://play.google.com/store/apps/")
    assert L["computer"]["url"] == "https://tailscale.com/download" and L["linux"]["url"] == "https://tailscale.com/download/linux"


@node
def test_links_are_https_and_only_to_the_expected_sites():
    for k, v in links().items():
        assert re.match(r"^https://(tailscale\.com|login\.tailscale\.com|apps\.apple\.com|play\.google\.com)/", v["url"]), k
    assert "target = '_blank'" in JS and "noopener noreferrer" in JS


@node
def test_every_step_points_at_something_that_exists_on_the_page():
    for s in steps(remote=False, replay=True):
        t = s["target"]
        if not t:
            continue
        if t.startswith("#"):
            assert f'id="{t[1:]}"' in HTML, (s["id"], t)
        elif t.startswith("."):
            assert re.search(rf'class="[^"]*\b{t[1:]}\b', HTML), (s["id"], t)


@node
def test_tool_names_in_the_tutorial_match_the_buttons():
    text = " ".join(" ".join(s["body"]) for s in steps(remote=False, replay=True))
    for word in ("Point", "Brush", "Box", "Erase", "Apply", "Reset", "Remove object", "Remove tattoo", "Find faces", "Before / after", "Show QR code", "Approve", "Allow phone access"):
        assert word in text, word
        # (the Help text names real buttons: each one exists in the page)
    for label in ("Remove object", "Remove tattoo", "Find faces", "Before / after", "Show QR code", "Apply", "Reset"):
        assert label in HTML, label


def test_it_obeys_the_pages_strict_security_rules():
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "eval(", "new Function", "fetch(", "XMLHttpRequest", "sendBeacon", "cssText", "setAttribute('style'",
                "document.write", "WebSocket"):
        assert bad not in JS, bad
    assert not re.search(r"\son\w+=", HTML) and 'style="' not in HTML


def test_the_page_loads_it_before_the_editor_and_has_a_help_button():
    assert HTML.index('src="tutorial.js"') < HTML.index('src="app.js"')
    assert 'id="btn-help"' in HTML
    app = (WEB / "app.js").read_text()
    assert "veilTutorial.maybe(remote)" in app and "veilTutorial.replay(remote)" in app


def test_the_tutorial_remembers_in_this_browser_only_and_never_nags_without_storage():
    assert "veil.tutorial.done" in JS and "window.localStorage" in JS
    assert "!s || s.getItem(KEY) === '1'" in JS                   # no storage = treated as done
    assert JS.count("markDone()") >= 2 and "function finish() {\n    markDone();" in JS     # skipping and finishing both count
    assert "skip.addEventListener('click', finish)" in JS and "e.key === 'Escape'" in JS
