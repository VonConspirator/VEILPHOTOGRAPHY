"""Models Veil suggests, by hardware tier. Permissive licences only (see README for each one).
Only names listed here can be downloaded from inside Veil. Sizes are the download sizes
shown on ollama.com."""
from __future__ import annotations

TIERS = [
    {"tier": "Minimum", "hardware": "about 4 GB of video memory", "name": "qwen3.5:2b", "size": "2.7 GB",
     "license": "Apache-2.0", "note": "Small and quick. Reads simple commands."},
    {"tier": "Recommended", "hardware": "8-12 GB of video memory (RTX 3060 12 GB)", "name": "qwen3.5:9b", "size": "6.6 GB",
     "license": "Apache-2.0", "note": "Reads commands in about a second and can look at the photo."},
    {"tier": "High-end", "hardware": "24 GB of video memory (RTX 3090)", "name": "qwen3.5:27b", "size": "17 GB",
     "license": "Apache-2.0", "note": "Best at tricky wording and understanding the photo."},
]
DOWNLOADABLE = {t["name"] for t in TIERS}
