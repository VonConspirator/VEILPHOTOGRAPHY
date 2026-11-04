"""Downloads with resume, and checksum checks. A file that does not match is deleted and never used."""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20


class DownloadError(Exception):
    """Something to tell the user in plain words."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha1_of(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def bar(done: int, total: int, started: float, label: str, out=sys.stderr) -> None:
    if not total:
        return
    pct = min(100, done * 100 // total)
    rate = done / max(0.1, time.monotonic() - started) / 1e6
    out.write(f"\r  {label[:46]:<46} {pct:3d}%  {done / 1e6:8.0f} / {total / 1e6:.0f} MB  {rate:5.1f} MB/s ")
    out.flush()


def fetch(url: str, dest: Path, size: int, sha256: str | None = None, git_sha1: str | None = None,
          label: str = "", retries: int = 4, show=bar) -> None:
    """Downloads url to dest (resuming a .part file), then checks size and checksum. Raises DownloadError on any problem."""
    if not (sha256 or git_sha1):
        raise DownloadError(f"No checksum is recorded for {label or dest.name}, so it will not be downloaded.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == size and _matches(dest, sha256, git_sha1):
        return
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink(); have = 0
        try:
            if have < size:
                req = urllib.request.Request(url, headers={"User-Agent": "veil-installer", **({"Range": f"bytes={have}-"} if have else {})})
                with urllib.request.urlopen(req, timeout=30) as r:
                    code = getattr(r, "status", 200)
                    if have and code != 206:                  # the server ignored the resume request: start again
                        have = 0
                    started = time.monotonic()
                    with open(part, "ab" if have else "wb") as out:
                        got = have
                        while True:
                            chunk = r.read(CHUNK)
                            if not chunk:
                                break
                            out.write(chunk); got += len(chunk)
                            show(got, size, started, label or dest.name)
                if show is bar:
                    sys.stderr.write("\n")
            break
        except (urllib.error.URLError, OSError, ConnectionError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise DownloadError(f"The download server refused {label or url} (error {e.code}).") from None
            time.sleep(min(2 * (attempt + 1), 8))
    else:
        raise DownloadError(f"Could not download {label or url}: {last}. Check your internet connection and run the installer again; it will continue where it stopped.")
    if part.stat().st_size != size:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{label or dest.name} arrived with the wrong size and was deleted.")
    if not _matches(part, sha256, git_sha1):
        part.unlink(missing_ok=True)
        raise DownloadError(f"{label or dest.name} did not match its recorded checksum and was deleted. It was not used.")
    part.replace(dest)


def _matches(path: Path, sha256: str | None, git_sha1: str | None) -> bool:
    return sha256_of(path) == sha256 if sha256 else git_sha1_of(path) == git_sha1


def fetch_model(reg, models_dir: Path, name: str, show=bar) -> None:
    """Every file of one registry model, with its recorded checksum."""
    for f in reg.MODELS[name]["files"]:
        fetch(f["url"], models_dir / f["path"], f["size"], f.get("sha256"), f.get("git_sha1"), label=f["path"], show=show)
    for f in reg.MODELS[name]["files"]:                     # a second, separate check of what is on disk
        if not reg.file_ok(models_dir, f, deep=True):
            raise DownloadError(f"{f['path']} failed its final checksum check.")
