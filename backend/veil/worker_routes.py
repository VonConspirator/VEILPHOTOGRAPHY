"""Settings for using a GPU worker. Anyone signed in (a phone too) can see whether it is on and what happened to the last
job; only the owner at the computer can pair, switch it on or off, or forget it. The token is never sent to the page."""
from __future__ import annotations

import socket

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .remote import AddressError, Dispatcher, WorkerUnavailable


class PairRequest(BaseModel):
    address: str = Field(max_length=200)
    code: str = Field(max_length=40)


class EnableRequest(BaseModel):
    enabled: bool


def register(app: FastAPI, *, dispatcher: Dispatcher, require_ready, require_owner) -> None:
    client = dispatcher.client

    @app.get("/api/worker", dependencies=[Depends(require_ready)])
    def worker_status():
        pub = client.public()
        out = {"configured": pub["configured"], "enabled": pub["enabled"], "name": pub["name"], "last": dispatcher.last, "reachable": False,
               "capabilities": {}, "gpu": "", "vram_gb": 0, "problem": ""}
        if pub["enabled"]:
            try:
                h = client.health()
                out.update(reachable=True, capabilities=h.get("capabilities", {}), gpu=h.get("gpu", ""), vram_gb=h.get("vram_gb", 0), busy=h.get("busy", False))
            except WorkerUnavailable as e:
                out["problem"] = str(e)
        return out

    @app.get("/api/worker/address", dependencies=[Depends(require_owner)])
    def worker_address():
        return {"address": client.public()["address"]}

    @app.post("/api/worker/pair", dependencies=[Depends(require_owner)])
    def worker_pair(req: PairRequest):
        try:
            client.pair(req.address, req.code, socket.gethostname() or "Computer")
        except AddressError as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True, "name": client.name()}

    @app.post("/api/worker/enable", dependencies=[Depends(require_owner)])
    def worker_enable(req: EnableRequest):
        try:
            client.set_enabled(req.enabled)
        except AddressError as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True}

    @app.post("/api/worker/test", dependencies=[Depends(require_owner)])
    def worker_test():
        if not client.settings().get("token"):
            raise HTTPException(400, "Pair a worker first.")
        try:
            h = client.health(force=True)
        except WorkerUnavailable as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True, "name": h.get("name", ""), "gpu": h.get("gpu", ""), "vram_gb": h.get("vram_gb", 0), "device": h.get("device", ""),
                "capabilities": h.get("capabilities", {})}

    @app.delete("/api/worker", dependencies=[Depends(require_owner)])
    def worker_forget():
        client.forget()
        return {"ok": True}
