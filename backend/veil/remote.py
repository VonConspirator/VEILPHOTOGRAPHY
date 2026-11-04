"""Using a GPU worker (another computer of yours, over Tailscale) for the heavy model jobs.

`Dispatcher` stands in for the local Engine. For each job it asks the worker first when one is switched on and healthy,
and otherwise (or if the worker cannot be reached) does the job here and says so. Photos only ever travel to the one
address the owner paired, only over Tailscale, only when the worker is switched on, and are never kept by the worker.
"""
from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import socket
import threading
import time
from pathlib import Path

import httpx
from PIL import Image

from . import netguard
from .vision.registry import ModelError
from .worker import DEFAULT_PORT, PROTOCOL

HEALTH_SECONDS = 20
DOWN_SECONDS = 15                       # after a failure, go straight to the fallback for this long
CONNECT_TIMEOUT = 4.0
READ_TIMEOUT = {"generative_fill": 900.0}
DEFAULT_READ_TIMEOUT = 240.0


class WorkerUnavailable(Exception):
    """The worker could not do this job (not reachable, busy, not paired any more...). The job is done here instead."""


class AddressError(Exception):
    """A worker address the owner typed that Veil refuses, in plain words."""


# ---------------------------------------------------------------- the address
def parse_address(text: str) -> tuple[str, int]:
    t = (text or "").strip()
    if not t or any(c in t for c in "/\\@ ?#") or "://" in t:
        raise AddressError("Type just the worker's Tailscale address, for example 100.101.102.103 or my-gpu.tail1234.ts.net (no http://).")
    host, port = t, DEFAULT_PORT
    if t.startswith("["):
        host, _, rest = t[1:].partition("]")
        if rest.startswith(":"):
            port = rest[1:]
    elif t.count(":") == 1:
        host, port = t.split(":")
    try:
        port = int(port)
    except ValueError:
        raise AddressError("The port must be a number.") from None
    if not 1 <= port <= 65535 or not host:
        raise AddressError("That address does not look right.")
    return host.lower(), port


def resolve_private(host: str, port: int) -> str:
    """The IP to connect to. Only a Tailscale address (or this very computer) is accepted, checked on every use."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise AddressError(f"Could not find {host}. Check the address, and that Tailscale is on here.") from None
    ips = []
    for info in infos:
        ip = info[4][0]
        if netguard.classify(ip) == "other":
            raise AddressError(f"{host} is not a Tailscale address. Veil only talks to a worker over Tailscale, never over your Wi-Fi or the internet.")
        ips.append(ip)
    if not ips:
        raise AddressError(f"Could not find {host}.")
    return ips[0]


# ---------------------------------------------------------------- saved setup
class WorkerConfig:
    """data/worker.json. It holds the pairing token, so the file is private (0600) and the token is never sent to the page."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "worker.json"
        self._lock = threading.RLock()

    def load(self) -> dict:
        try:
            d = json.loads(self.path.read_text())
        except Exception:
            d = {}
        return d if isinstance(d, dict) else {}

    def save(self, d: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=2))
            tmp.chmod(0o600)
            tmp.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


# ---------------------------------------------------------------- talking to the worker
class WorkerClient:
    def __init__(self, cfg: WorkerConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.transport = transport                 # tests plug a fake network in here
        self._health: tuple[float, dict | None, str] | None = None
        self._lock = threading.Lock()

    # -- state
    def settings(self) -> dict:
        return self.cfg.load()

    @property
    def enabled(self) -> bool:
        d = self.settings()
        return bool(d.get("enabled") and d.get("token") and d.get("host"))

    def name(self) -> str:
        return self.settings().get("name") or "GPU worker"

    # -- http
    def _request(self, method: str, path: str, *, host: str | None = None, port: int | None = None, token: str | None = None,
                 json_body: dict | None = None, content: bytes | None = None, read_timeout: float = DEFAULT_READ_TIMEOUT) -> httpx.Response:
        d = self.settings()
        host = host or d.get("host", "")
        port = port or d.get("port", DEFAULT_PORT)
        token = d.get("token") if token is None else token
        try:
            ip = resolve_private(host, port)
        except AddressError as e:
            raise WorkerUnavailable(str(e)) from None
        netloc = f"[{ip}]" if ":" in ip else ip
        headers = {"Host": f"{host}:{port}"}
        if token:
            headers["Authorization"] = "Bearer " + token
        timeout = httpx.Timeout(read_timeout, connect=CONNECT_TIMEOUT)
        try:
            # trust_env=False: never route a photo through a proxy set in the environment. No redirects are followed.
            with httpx.Client(transport=self.transport, timeout=timeout, trust_env=False, follow_redirects=False) as c:
                return c.request(method, f"http://{netloc}:{port}{path}", headers=headers, json=json_body, content=content)
        except httpx.TimeoutException:
            raise WorkerUnavailable("The worker took too long to answer.") from None
        except httpx.HTTPError:
            raise WorkerUnavailable("The worker could not be reached. Is it running, and is Tailscale on at both ends?") from None

    @staticmethod
    def _detail(r: httpx.Response) -> str:
        try:
            d = r.json().get("detail")
            return d if isinstance(d, str) else "The worker refused the job."
        except Exception:
            return "The worker gave an answer Veil could not read."

    # -- pairing (the code comes from `veil-worker pair` on the worker)
    def pair(self, address: str, code: str, my_name: str) -> dict:
        host, port = parse_address(address)
        try:
            r = self._request("POST", "/v1/pair", host=host, port=port, token="", json_body={"code": code, "name": my_name}, read_timeout=15)
        except WorkerUnavailable as e:
            raise AddressError(str(e)) from None
        if r.status_code != 200:
            raise AddressError(self._detail(r))
        try:
            d = r.json()
            ok = d["token"].startswith("vwk_") and d["protocol"] == PROTOCOL
        except Exception:
            raise AddressError("That did not look like a Veil worker.") from None
        if not ok:
            raise AddressError("That worker speaks a different version of Veil's worker protocol. Update Veil on both computers.")
        self.cfg.save({"enabled": True, "host": host, "port": port, "token": d["token"], "worker_id": d["worker_id"],
                       "name": str(d.get("name") or host)[:40], "client_id": d.get("client_id", "")})
        self._health = None
        return self.public()

    def forget(self) -> None:
        if self.settings().get("token"):
            try:
                self._request("POST", "/v1/unpair", read_timeout=8)          # tell the worker to stop trusting this token
            except WorkerUnavailable:
                pass
        self.cfg.clear()
        self._health = None

    def set_enabled(self, on: bool) -> None:
        d = self.settings()
        if not d.get("token"):
            raise AddressError("Pair a worker first.")
        d["enabled"] = bool(on)
        self.cfg.save(d)
        self._health = None

    # -- health
    def health(self, force: bool = False) -> dict:
        """The worker's capabilities. Raises WorkerUnavailable (remembered for a few seconds, so a dead worker costs one wait)."""
        with self._lock:
            now = time.monotonic()
            if self._health and not force and now - self._health[0] < (HEALTH_SECONDS if self._health[1] else DOWN_SECONDS):
                if self._health[1] is None:
                    raise WorkerUnavailable(self._health[2])
                return self._health[1]
            try:
                r = self._request("GET", "/v1/health", read_timeout=10)
                if r.status_code == 401:
                    raise WorkerUnavailable("The worker no longer trusts this computer. Pair it again.")
                if r.status_code != 200:
                    raise WorkerUnavailable(self._detail(r))
                h = r.json()
                if h.get("protocol") != PROTOCOL:
                    raise WorkerUnavailable("The worker runs a different version of Veil. Update Veil on both computers.")
                if h.get("worker_id") != self.settings().get("worker_id"):
                    raise WorkerUnavailable("This is not the worker that was paired (its identity changed). Pair again if you trust it.")
            except WorkerUnavailable as e:
                self._health = (now, None, str(e))
                raise
            except Exception:
                self._health = (now, None, "The worker gave an answer Veil could not read.")
                raise WorkerUnavailable("The worker gave an answer Veil could not read.") from None
            self._health = (now, h, "")
            return h

    def capable(self, cap: str) -> bool:
        return bool(self.health().get("capabilities", {}).get(cap))

    # -- jobs
    def _job(self, path: str, body: dict, blobs: dict[str, bytes], read_timeout: float) -> httpx.Response:
        """Posts a job. If the worker does not have an image yet (409), uploads just that image and tries once more."""
        for attempt in (0, 1):
            r = self._request("POST", path, json_body=body, read_timeout=read_timeout)
            if r.status_code == 409 and attempt == 0:
                try:
                    need = r.json()["detail"]["need"]
                    assert isinstance(need, list) and all(n in blobs for n in need)
                except Exception:
                    raise WorkerUnavailable("The worker asked for something unexpected.") from None
                for sha in need:
                    up = self._request("PUT", f"/v1/images/{sha}", content=blobs[sha], read_timeout=read_timeout)
                    if up.status_code != 200:
                        raise WorkerUnavailable(self._detail(up))
                continue
            break
        if r.status_code == 200:
            return r
        if r.status_code == 400:
            raise ModelError(self._detail(r))                    # a real problem with the request ("Mark the area first"): not a fallback case
        if r.status_code == 401:
            self._health = None
            raise WorkerUnavailable("The worker no longer trusts this computer. Pair it again.")
        raise WorkerUnavailable(self._detail(r) if r.status_code in (413, 429, 503) else "The worker could not finish this job.")

    @staticmethod
    def _sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def image_result(self, r: httpx.Response, mode: str) -> Image.Image:
        try:
            im = Image.open(io.BytesIO(r.content))
            im.load()
            return im.convert(mode)
        except Exception:
            raise WorkerUnavailable("The worker sent back something that is not a picture.") from None

    def job_segment(self, memo_png: tuple[str, bytes], points, box) -> Image.Image:
        sha, data = memo_png
        body = {"image": sha, "points": [list(p) for p in points], "box": list(box) if box else None}
        return self.image_result(self._job("/v1/segment", body, {sha: data}, DEFAULT_READ_TIMEOUT), "L")

    def job_matte(self, img: Image.Image) -> Image.Image:
        data = png_bytes(img.convert("RGB")); sha = self._sha(data)
        return self.image_result(self._job("/v1/matte", {"image": sha}, {sha: data}, DEFAULT_READ_TIMEOUT), "L")

    def job_inpaint(self, img: Image.Image, mask: Image.Image, grow) -> Image.Image:
        di, dm = png_bytes(img.convert("RGB")), png_bytes(mask.convert("L"))
        si, sm = self._sha(di), self._sha(dm)
        return self.image_result(self._job("/v1/inpaint", {"image": si, "mask": sm, "grow": grow}, {si: di, sm: dm}, DEFAULT_READ_TIMEOUT), "RGB")

    def job_fill(self, img, mask, prompt, negative, grow, opts: dict) -> Image.Image:
        di, dm = png_bytes(img.convert("RGB")), png_bytes(mask.convert("L"))
        si, sm = self._sha(di), self._sha(dm)
        body = {"image": si, "mask": sm, "prompt": prompt, "negative": negative or "", "grow": grow, **opts}
        return self.image_result(self._job("/v1/generative_fill", body, {si: di, sm: dm}, READ_TIMEOUT["generative_fill"]), "RGB")

    def job_faces(self, img: Image.Image, min_score: float) -> list[dict]:
        data = png_bytes(img.convert("RGB")); sha = self._sha(data)
        r = self._job("/v1/faces", {"image": sha, "min_score": min_score}, {sha: data}, DEFAULT_READ_TIMEOUT)
        try:
            return [{"box": tuple(float(v) for v in f["box"]), "score": float(f["score"])} for f in r.json()["faces"]]
        except Exception:
            raise WorkerUnavailable("The worker sent back something Veil could not read.") from None

    def job_landmarks(self, img: Image.Image, box):
        import numpy as np
        data = png_bytes(img.convert("RGB")); sha = self._sha(data)
        r = self._job("/v1/landmarks", {"image": sha, "box": [float(v) for v in box]}, {sha: data}, DEFAULT_READ_TIMEOUT)
        try:
            pts = r.json()["points"]
            return None if pts is None else np.array(pts, dtype=np.float64).reshape(478, 2)
        except Exception:
            raise WorkerUnavailable("The worker sent back something Veil could not read.") from None

    def public(self) -> dict:
        d = self.settings()
        return {"configured": bool(d.get("token")), "enabled": self.enabled, "name": d.get("name", ""), "address": f"{d['host']}:{d['port']}" if d.get("host") else ""}


# ---------------------------------------------------------------- the stand-in engine
class Dispatcher:
    """Looks like the Engine. Worker first when it is on and able; otherwise (or on trouble) here, with a visible note."""

    def __init__(self, local, client: WorkerClient):
        self.local = local
        self.client = client
        self.seq = 0
        self.last: dict = {}
        self._segmemo: dict[str, tuple[str, bytes]] = {}

    def __getattr__(self, name):                       # everything else (root, device, pick, release, ...) is the local engine's
        return getattr(self.local, name)

    # -- notes the page shows
    def _note(self, where: str, job: str, reason: str = "", fallback: bool = False) -> None:
        self.seq += 1
        self.last = {"seq": self.seq, "where": where, "job": job, "fallback": fallback, "reason": reason, "worker": self.client.name() if where == "worker" or fallback else "",
                     "at": time.time()}

    def _go(self, cap: str, job: str, remote, local):
        c = self.client
        if not c.enabled:
            return local()
        try:
            if not c.capable(cap):
                self._note("here", job, f"The worker does not have the {cap} tool set up, so this ran on this computer.", fallback=True)
                return local()
            result = remote()
            self._note("worker", job)
            return result
        except WorkerUnavailable as e:
            reason = str(e)
        self._note("here", job, reason + " This ran on this computer instead.", fallback=True)
        try:
            return local()
        except ModelError as err:
            raise ModelError(f"{reason} And this computer cannot do it either: {err}") from None

    # -- the jobs
    def segment(self, key, img, points, box):
        def remote():
            memo = self._segmemo.get(key)
            if memo is None:
                data = png_bytes(img.convert("RGB"))
                memo = (hashlib.sha256(data).hexdigest(), data)
                self._segmemo = {k: v for k, v in list(self._segmemo.items())[-1:]}
                self._segmemo[key] = memo
            return self.client.job_segment(memo, points, box)
        if not points and box is None:
            return self.local.segment(key, img, points, box)     # same plain message as ever, nothing sent
        return self._go("segment", "select", remote, lambda: self.local.segment(key, img, points, box))

    def matte(self, img):
        return self._go("matte", "find the person", lambda: self.client.job_matte(img), lambda: self.local.matte(img))

    def inpaint(self, img, mask, grow=None):
        return self._go("inpaint", "fill in", lambda: self.client.job_inpaint(img, mask, grow), lambda: self.local.inpaint(img, mask, grow))

    def generative_fill(self, img, mask, prompt, negative="", grow=None, *, strength=0.99, seed=None, context=0.4, steps=30, feather_px=3.0):
        opts = {"strength": strength, "seed": seed, "context": context, "steps": steps, "feather_px": feather_px}
        return self._go("generative", "generative fill", lambda: self.client.job_fill(img, mask, prompt, negative, grow, opts),
                        lambda: self.local.generative_fill(img, mask, prompt, negative, grow, strength=strength, seed=seed, context=context,
                                                           steps=steps, feather_px=feather_px))

    def detect_faces(self, img, min_score=0.6):
        return self._go("faces", "find faces", lambda: self.client.job_faces(img, min_score), lambda: self.local.detect_faces(img, min_score))

    def landmarks(self, img, box):
        return self._go("landmarks", "map the face", lambda: self.client.job_landmarks(img, box), lambda: self.local.landmarks(img, box))

    # -- what is available (this computer's models, or the worker's)
    def _worker_health(self) -> dict | None:
        if not self.client.enabled:
            return None
        try:
            return self.client.health()
        except WorkerUnavailable:
            return None

    def has(self, name: str) -> bool:
        if self.local.has(name):
            return True
        h = self._worker_health()
        return bool(h and name in h.get("models", []))

    def sdxl_ready(self) -> bool:
        if self.local.sdxl_ready():
            return True
        h = self._worker_health()
        return bool(h and h.get("capabilities", {}).get("generative"))

    def faces_ready(self) -> bool:
        if self.local.faces_ready():
            return True
        h = self._worker_health()
        return bool(h and h.get("capabilities", {}).get("landmarks") and h.get("capabilities", {}).get("faces"))

    def status(self) -> dict:
        st = self.local.status()
        h = self._worker_health()
        caps = (h or {}).get("capabilities", {})
        if h and all(caps.get(k) for k in ("segment", "matte", "inpaint", "faces")):
            st["ready"] = True
        by = {m["name"]: m["installed"] for m in st.get("models", [])}
        # what the page may offer: this computer can do it, or the (switched-on, reachable) worker can
        st["faces_available"] = bool((st.get("faces_libs") and by.get("yunet") and by.get("face-landmarker")) or (caps.get("faces") and caps.get("landmarks")))
        st["generative_available"] = bool((st.get("generative_libs") and by.get("sdxl-inpaint")) or caps.get("generative"))
        st["worker"] = {**self.client.public(), "reachable": bool(h), "capabilities": (h or {}).get("capabilities", {}),
                        "gpu": (h or {}).get("gpu", ""), "vram_gb": (h or {}).get("vram_gb", 0)}
        return st
