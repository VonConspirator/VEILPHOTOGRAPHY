"""The editing models Veil uses, with the exact size and checksum of every file.
Files are checked every time they are loaded; a file that does not match is never used.

required = part of the Recommended set.   optional = only downloaded when the owner asks.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path


SDXL_REV = "115134f363124c53c7d878647567d04daf26e41e"
SDXL_URL = "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/" + SDXL_REV + "/"

MODELS: dict[str, dict] = {
    "sam2": {
        "label": "Segment Anything 2.1 (base+)", "use": "Turns a click, brush or box into a precise outline.",
        "license": "Apache-2.0", "required": True, "tier": "Recommended", "group": "selection", "rank": 2,
        "sam_config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "files": [{"path": "sam2/sam2.1_hiera_base_plus.pt", "size": 323606802,
                   "sha256": "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5",
                   "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"}],
    },
    "sam2-tiny": {
        "label": "Segment Anything 2.1 (tiny)", "use": "Smaller, faster outline model for the Minimum tier.",
        "license": "Apache-2.0", "required": True, "tier": "Minimum", "group": "selection", "rank": 1,
        "sam_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        # Meta publishes no checksum for this file; the one below was recorded when it was first downloaded.
        "files": [{"path": "sam2/sam2.1_hiera_tiny.pt", "size": 156008466, "sha256": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
                   "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"}],
    },
    "sam2-large": {
        "label": "Segment Anything 2.1 (large)", "use": "Most precise outline model for the High-end tier.",
        "license": "Apache-2.0", "required": True, "tier": "High-end", "group": "selection", "rank": 3,
        "sam_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        # Meta publishes no checksum for this file; the one below was recorded when it was first downloaded.
        "files": [{"path": "sam2/sam2.1_hiera_large.pt", "size": 898083611, "sha256": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
                   "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"}],
    },
    "birefnet-lite": {
        "label": "BiRefNet lite", "use": "Smaller person finder for the Minimum tier.",
        "license": "MIT", "required": True, "tier": "Minimum", "group": "matte", "rank": 1, "backbone": 3,
        "files": [{"path": "birefnet-lite/model.safetensors", "size": 177634392,
                   "sha256": "4417d89795250e698c3cb0ae8df15743810065f646f48a694fdfa7ca052d0815",   # matches the checksum Hugging Face publishes for the file
                   "url": "https://huggingface.co/ZhengPeng7/BiRefNet_lite/resolve/aa62cd87eafb9cc43056d08ef3615a14628b831d/model.safetensors"}],
    },
    "lama": {
        "label": "LaMa big-lama", "use": "Removes objects and tattoos by filling in from the surroundings.",
        "license": "Apache-2.0", "required": True, "tier": "Recommended",
        # A third-party re-host of the official weights; the checksum below was recorded when first downloaded.
        "files": [{"path": "lama/big-lama.pt", "size": 205803670,
                   "sha256": "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c",
                   "url": "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"}],
    },
    "birefnet": {
        "label": "BiRefNet", "use": "Finds the person for background removal, blur and replacement.",
        "license": "MIT", "required": True, "tier": "Recommended", "group": "matte", "rank": 2, "backbone": 6,
        "files": [{"path": "birefnet/model.safetensors", "size": 444473596,
                   "sha256": "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154",   # matches the checksum Hugging Face publishes
                   "url": "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/model.safetensors"}],
    },
    "yunet": {
        "label": "YuNet face detector", "use": "Finds faces in a photo (box and five landmark points).",
        "license": "MIT", "required": True, "tier": "Recommended",
        "files": [{"path": "yunet/face_detection_yunet_2023mar.onnx", "size": 232589,
                   "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",   # matches the checksum OpenCV Zoo publishes
                   "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"}],
    },
    "face-landmarker": {
        "label": "MediaPipe Face Landmarker", "use": "Maps 478 points on a face: outline, eyes, irises, lips. Used by the look edits.",
        "license": "Apache-2.0 (the three model cards in the bundle each say so)", "required": True, "tier": "Recommended",
        "extra_packages": "absl-py flatbuffers matplotlib sounddevice cffi && pip install --no-deps mediapipe",
        # Google publishes no checksum for this file; the one below was recorded when it was first downloaded.
        "files": [{"path": "mediapipe/face_landmarker.task", "size": 3758596,
                   "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
                   "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"}],
    },
    "sdxl-inpaint": {
        "label": "Stable Diffusion XL Inpainting 0.1 (fp16)",
        "use": "Optional. Generative fill: better object and tattoo removal on large areas, and new backgrounds from a description.",
        "license": "CreativeML Open RAIL++-M (use restrictions apply, see README)", "required": False, "tier": "Optional",
        "extra_packages": "diffusers transformers accelerate",
        "files": [dict(f, url=SDXL_URL + f["path"], path="sdxl-inpaint/" + f["path"]) for f in [{'path': 'scheduler/scheduler_config.json', 'size': 479, 'git_sha1': '9e13de2afd2250977d01837deed2b1c35b75e552'}, {'path': 'text_encoder/config.json', 'size': 746, 'git_sha1': 'a34c80dc728ab0a61bc5f7e849fad01723f94a01'}, {'path': 'text_encoder/model.fp16.safetensors', 'size': 246144867, 'sha256': 'fc83cf401d930147807e7c44021c164bcc5508c9d4cc0ff35f4e354685ca9cd0'}, {'path': 'text_encoder_2/config.json', 'size': 758, 'git_sha1': '8f0f26a6056dd0aae0437184a1871eed6006e170'}, {'path': 'text_encoder_2/model.fp16.safetensors', 'size': 1389382884, 'sha256': 'a8622bd41f8d359df484fdf9de091edb9337a2fd747f0a5c9c320a62e24d1fd3'}, {'path': 'tokenizer/merges.txt', 'size': 524619, 'git_sha1': '76e821f1b6f0a9709293c3b6b51ed90980b3166b'}, {'path': 'tokenizer/special_tokens_map.json', 'size': 472, 'git_sha1': '2c2130b544c0c5a72d5d00da071ba130a9800fb2'}, {'path': 'tokenizer/tokenizer_config.json', 'size': 737, 'git_sha1': '2e8612a429492973fe60635b3f44a28b065cfac0'}, {'path': 'tokenizer/vocab.json', 'size': 1059962, 'git_sha1': '469be27c5c010538f845f518c4f5e8574c78f7c8'}, {'path': 'tokenizer_2/merges.txt', 'size': 524619, 'git_sha1': '76e821f1b6f0a9709293c3b6b51ed90980b3166b'}, {'path': 'tokenizer_2/special_tokens_map.json', 'size': 460, 'git_sha1': 'ae0c5be6f35217e51c4c000fd325d8de0294e99c'}, {'path': 'tokenizer_2/tokenizer_config.json', 'size': 725, 'git_sha1': 'a8438e020c4497a429240d6b89e0bf9a6e2ffa92'}, {'path': 'tokenizer_2/vocab.json', 'size': 1059962, 'git_sha1': '469be27c5c010538f845f518c4f5e8574c78f7c8'}, {'path': 'unet/config.json', 'size': 1932, 'git_sha1': '2fa48c31c3a64ba04fa8a76ac91f086d6309b406'}, {'path': 'unet/diffusion_pytorch_model.fp16.safetensors', 'size': 5135178560, 'sha256': '6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f'}, {'path': 'vae/config.json', 'size': 659, 'git_sha1': '4a3bb2e796869bdfa617fd28162f4087c4f771e3'}, {'path': 'vae/diffusion_pytorch_model.fp16.safetensors', 'size': 167335338, 'sha256': '4ad62825e5c8b31eefb77355ba6693785619bf7d669d9b1a6fd9f19dec6d65b3'}, {'path': 'model_index.json', 'size': 690, 'git_sha1': '8cab34157f46b4c35107a65e48a7236057ad6e46'}]],
    },
}


# ---------------------------------------------------------------- tiers (what the installer sets up)
# Language models run in Ollama. `weights` is the sha256 of the model's weights file as served on 2026-11-04 (Ollama tags can be
# re-pushed later, so the installer compares it after a fresh pull and tells the owner if it changed).
LLMS = {
    "qwen3.5:2b": {"size": 2683396812, "license": "Apache-2.0", "weights": "e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7"},
    "qwen3.5:9b": {"size": 6550825550, "license": "Apache-2.0", "weights": "02d45dc1cf451ba2475ac33b301c2dd8f985abe4c182ce04a1f2f5bf0260278d"},
    "qwen3.5:27b": {"size": 17478556591, "license": "Apache-2.0", "weights": "c7e0e920a0948849beb7074d8368cbd5ac6825fcd1f2450cf851276325d8a6a5"},
}
TIERS = {
    "Minimum": {"models": ["sam2-tiny", "birefnet-lite", "lama", "yunet", "face-landmarker"], "llm": "qwen3.5:2b",
                "about": "Small and quick. Works on 4-6 GB of video memory or a computer with no NVIDIA graphics card (slower)."},
    "Recommended": {"models": ["sam2", "birefnet", "lama", "yunet", "face-landmarker"], "llm": "qwen3.5:9b",
                    "about": "The best balance. Made for 8-12 GB of video memory."},
    "High-end": {"models": ["sam2-large", "birefnet", "lama", "yunet", "face-landmarker"], "llm": "qwen3.5:27b",
                 "about": "The most precise outlines and the smartest command model. Needs 16 GB or more of video memory."},
}
TIER_ORDER = ("Minimum", "Recommended", "High-end")


class ModelError(Exception):
    """A problem the user can be told about."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_ok(root: Path, f: dict, deep: bool = True) -> bool:
    p = root / f["path"]
    if not p.is_file() or p.stat().st_size != f["size"]:
        return False
    if not deep:
        return True
    return _sha256(p) == f["sha256"] if "sha256" in f else _git_sha1(p) == f["git_sha1"]


def installed(root: Path, name: str, deep: bool = False) -> bool:
    return all(file_ok(root, f, deep) for f in MODELS[name]["files"])


def verify(root: Path, name: str) -> None:
    """Full checksum check. Raises if any file is missing or has been changed."""
    for f in MODELS[name]["files"]:
        if not file_ok(root, f, deep=True):
            raise ModelError(f"{MODELS[name]['label']}: {f['path']} is missing or does not match its recorded checksum.")


class Downloader:
    """Fetches one model's files in the background, checking every file against its checksum."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self.state = {"running": False, "name": "", "file": "", "done": 0, "total": 0, "error": ""}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def start(self, name: str) -> None:
        if name not in MODELS:
            raise ModelError("Unknown model.")
        with self._lock:
            if self.state["running"]:
                raise ModelError("A download is already running.")
            self.state = {"running": True, "name": name, "file": "", "done": 0, "total": sum(f["size"] for f in MODELS[name]["files"]), "error": ""}
        threading.Thread(target=self._run, args=(name,), daemon=True).start()

    def _set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    def _run(self, name: str) -> None:
        import httpx           # imported here so the installer can read this file before httpx is installed
        try:
            done = 0
            for f in MODELS[name]["files"]:
                dest = self.root / f["path"]
                if file_ok(self.root, f, deep=True):
                    done += f["size"]; self._set(done=done); continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_name(dest.name + ".part")
                self._set(file=f["path"])
                with httpx.stream("GET", f["url"], follow_redirects=True, timeout=httpx.Timeout(60.0, connect=10.0)) as r:
                    if r.status_code != 200:
                        raise ModelError(f"The download server answered {r.status_code}.")
                    with open(part, "wb") as out:
                        for chunk in r.iter_bytes(1 << 20):
                            out.write(chunk); done += len(chunk); self._set(done=done)
                part.replace(dest)
                if not file_ok(self.root, f, deep=True):
                    dest.unlink(missing_ok=True)
                    raise ModelError(f"{f['path']} did not match its recorded checksum and was deleted.")
            self._set(running=False, file="")
        except ModelError as e:
            self._set(running=False, error=str(e))
        except (httpx.HTTPError, OSError):
            self._set(running=False, error="The download failed. Check your internet connection and disk space.")
