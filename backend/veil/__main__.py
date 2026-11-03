"""Run Veil:  python -m veil   (listens on this computer only)."""
import uvicorn

from . import config

if __name__ == "__main__":
    print(f"Veil is starting at http://localhost:{config.PORT}  (this computer only)")
    uvicorn.run("veil.main:app_factory", factory=True, host="127.0.0.1", port=config.PORT, log_level="warning")
