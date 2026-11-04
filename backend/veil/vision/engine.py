"""Loads the editing models on demand, runs them on the GPU, and lets go of the memory when idle
(so they do not fight the language model for video memory). Nothing here touches the internet."""
from __future__ import annotations

import gc
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from . import registry
from .registry import ModelError

IDLE_SECONDS = 300
LAMA_MAX_SIDE = 1536       # larger crops are shrunk for the model and the result is scaled back
SDXL_SIDE = 1024
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def _torch():
    try:
        import torch
    except ImportError:
        raise ModelError("The editing libraries are not installed (PyTorch is missing).") from None
    return torch


def binarize(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    m = mask.convert("L").resize(size, Image.Resampling.BILINEAR)
    return (np.asarray(m) > 127).astype(np.uint8)


def grow_mask(m: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return m
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m, k)


def feather(m: np.ndarray, px: float) -> Image.Image:
    img = Image.fromarray(m * 255, "L")
    return img.filter(ImageFilter.GaussianBlur(px)) if px > 0 else img


def crop_box(m: np.ndarray, margin_frac: float, min_margin: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        raise ModelError("Mark the area first.")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mg = int(max(min_margin, margin_frac * max(x1 - x0, y1 - y0)))
    return (max(0, x0 - mg), max(0, y0 - mg), min(size[0], x1 + mg), min(size[1], y1 + mg))


class Engine:
    def __init__(self, models_dir: Path):
        self.root = Path(models_dir)
        self._lock = threading.RLock()
        self._sam = self._lama = self._birefnet = self._sdxl = None
        self._sam_key = None
        self._last_use = 0.0
        self._timer: threading.Timer | None = None
        self._verified: set[str] = set()

    # ------------------------------------------------------------ status
    def device(self) -> str:
        try:
            t = _torch()
        except ModelError:
            return "none"
        return "cuda" if t.cuda.is_available() else "cpu"

    def libraries_ok(self) -> bool:
        try:
            _torch(); import cv2  # noqa: F401
            return True
        except (ModelError, ImportError):
            return False

    def status(self) -> dict:
        models = []
        for name, m in registry.MODELS.items():
            models.append({"name": name, "label": m["label"], "use": m["use"], "license": m["license"], "required": m["required"],
                           "tier": m["tier"], "size": sum(f["size"] for f in m["files"]),
                           "installed": registry.installed(self.root, name), "extra_packages": m.get("extra_packages", "")})
        base = all(x["installed"] for x in models if x["required"])
        try:
            import diffusers  # noqa: F401
            gen = True
        except ImportError:
            gen = False
        return {"libraries": self.libraries_ok(), "generative_libs": gen, "device": self.device(), "models": models,
                "ready": base and self.libraries_ok()}

    def has(self, name: str) -> bool:
        return registry.installed(self.root, name)

    # ------------------------------------------------------------ memory
    def _touch(self) -> None:
        self._last_use = time.monotonic()
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(IDLE_SECONDS + 1, self._idle_check)
        self._timer.daemon = True
        self._timer.start()

    def _idle_check(self) -> None:
        if time.monotonic() - self._last_use >= IDLE_SECONDS and self._lock.acquire(blocking=False):
            try:
                self.release()
            finally:
                self._lock.release()

    def release(self) -> None:
        """Let go of every loaded model and give the video memory back."""
        with self._lock:
            self._sam = self._lama = self._birefnet = self._sdxl = None
            self._sam_key = None
            gc.collect()
            try:
                t = _torch()
                if t.cuda.is_available():
                    t.cuda.empty_cache()
            except ModelError:
                pass

    def _need(self, name: str) -> None:
        if not self.has(name):
            raise ModelError(f"{registry.MODELS[name]['label']} is not downloaded yet. Open Model, then Editing models.")
        if name not in self._verified:           # full checksum check once per run, before the file is ever opened
            registry.verify(self.root, name)
            self._verified.add(name)

    # ------------------------------------------------------------ selection (SAM 2.1)
    def segment(self, key: str, img: Image.Image, points: list[tuple[float, float, int]], box: tuple[float, float, float, float] | None) -> Image.Image:
        """points are (x, y, 1 = inside / 0 = outside) and box is (x0, y0, x1, y1), all as fractions 0..1.
        Returns the outline as an 'L' image, white = selected, same size as img."""
        if not points and box is None:
            raise ModelError("Click on the thing you want to select, or draw a box around it.")
        with self._lock:
            self._need("sam2")
            torch = _torch()
            if self._sam is None:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                ckpt = str(self.root / registry.MODELS["sam2"]["files"][0]["path"])
                self._sam = SAM2ImagePredictor(build_sam2(SAM_CONFIG, ckpt, device=self.device()))
                self._sam_key = None
            if self._sam_key != key:
                self._sam.set_image(np.asarray(img.convert("RGB")))
                self._sam_key = key
            w, h = img.size
            pc = np.array([[x * w, y * h] for x, y, _ in points], dtype=np.float32) if points else None
            pl = np.array([l for _, _, l in points], dtype=np.int32) if points else None
            bx = np.array([box[0] * w, box[1] * h, box[2] * w, box[3] * h], dtype=np.float32) if box else None
            with torch.inference_mode():
                masks, scores, _ = self._sam.predict(point_coords=pc, point_labels=pl, box=bx,
                                                     multimask_output=bool(points) and len(points) == 1 and box is None)
            best = masks[int(np.argmax(scores))]
            self._touch()
            return Image.fromarray((best > 0).astype(np.uint8) * 255, "L")

    # ------------------------------------------------------------ filling in (LaMa)
    def _lama_model(self):
        if self._lama is None:
            torch = _torch()
            self._lama = torch.jit.load(str(self.root / registry.MODELS["lama"]["files"][0]["path"]), map_location=self.device()).eval()
        return self._lama

    def inpaint(self, img: Image.Image, mask: Image.Image, grow: int | None = None) -> Image.Image:
        """Fill the white area of `mask` from its surroundings. Pixels outside the (grown) mask are not changed."""
        with self._lock:
            self._need("lama")
            torch = _torch()
            rgb = img.convert("RGB")
            m = binarize(mask, rgb.size)
            g = max(4, round(0.004 * max(rgb.size))) if grow is None else grow
            m = grow_mask(m, g)
            box = crop_box(m, 0.5, 48, rgb.size)
            crop = rgb.crop(box); cm = m[box[1]:box[3], box[0]:box[2]]
            f = min(1.0, LAMA_MAX_SIDE / max(crop.size))
            sw, sh = max(8, round(crop.width * f)), max(8, round(crop.height * f))
            small = crop.resize((sw, sh), Image.Resampling.LANCZOS) if f < 1 else crop
            sm = np.asarray(Image.fromarray(cm * 255).resize((sw, sh), Image.Resampling.NEAREST)) > 127 if f < 1 else cm > 0
            pw, ph = (8 - sw % 8) % 8, (8 - sh % 8) % 8          # the model wants sizes divisible by 8
            x = torch.from_numpy(np.asarray(small)).permute(2, 0, 1).float().div(255).unsqueeze(0)
            mk = torch.from_numpy(sm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            if pw or ph:
                x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect"); mk = torch.nn.functional.pad(mk, (0, pw, 0, ph), mode="reflect")
            dev = self.device()
            with torch.inference_mode():
                out = self._lama_model()(x.to(dev), (mk > 0.5).float().to(dev))
            res = out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[:sh, :sw]
            res_img = Image.fromarray((res * 255).astype(np.uint8))
            if f < 1:
                res_img = res_img.resize(crop.size, Image.Resampling.LANCZOS)
            soft = feather(cm, 2.0)
            blended = Image.composite(res_img, crop, soft)
            out_img = rgb.copy(); out_img.paste(blended, box[:2])
            self._touch()
            return out_img

    # ------------------------------------------------------------ foreground (BiRefNet)
    def _birefnet_model(self):
        if self._birefnet is None:
            torch = _torch()
            from safetensors.torch import load_file
            from .vendor.birefnet import BiRefNet
            net = BiRefNet(bb_pretrained=False)
            net.load_state_dict(load_file(str(self.root / registry.MODELS["birefnet"]["files"][0]["path"])))
            net = net.to(self.device()).eval()
            if self.device() == "cuda":
                net = net.half()
            self._birefnet = net
        return self._birefnet

    def matte(self, img: Image.Image) -> Image.Image:
        """The person (or main subject) as a soft 'L' matte, white = foreground, same size as img."""
        with self._lock:
            self._need("birefnet")
            torch = _torch()
            rgb = img.convert("RGB")
            net = self._birefnet_model()
            arr = np.asarray(rgb.resize((1024, 1024), Image.Resampling.BICUBIC)).astype(np.float32) / 255
            arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            x = x.to(self.device()).half() if self.device() == "cuda" else x
            with torch.inference_mode():
                pred = net(x)[-1].sigmoid().float().cpu()[0, 0].numpy()
            m = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8), "L").resize(rgb.size, Image.Resampling.BICUBIC)
            self._touch()
            return m

    # ------------------------------------------------------------ optional: generative fill (SDXL inpainting)
    def sdxl_ready(self) -> bool:
        if self._sdxl is not None:
            return True
        try:
            import diffusers  # noqa: F401
        except ImportError:
            return False
        return self.has("sdxl-inpaint")

    def _sdxl_pipe(self):
        if self._sdxl is None:
            if not self.has("sdxl-inpaint"):
                raise ModelError("Stable Diffusion XL Inpainting is not downloaded. It is optional: Model, then Editing models.")
            try:
                from diffusers import StableDiffusionXLInpaintPipeline
            except ImportError:
                raise ModelError("Generative fill needs the optional libraries: pip install diffusers transformers accelerate") from None
            torch = _torch()
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                str(self.root / "sdxl-inpaint"), torch_dtype=torch.float16, variant="fp16", local_files_only=True, use_safetensors=True)
            pipe.enable_model_cpu_offload()
            self._sdxl = pipe
        return self._sdxl

    def generative_fill(self, img: Image.Image, mask: Image.Image, prompt: str, negative: str = "", grow: int | None = None) -> Image.Image:
        """Repaint the white area following `prompt`. Works on a crop around the area at about 1024 px."""
        with self._lock:
            self._need_sdxl()
            torch = _torch()
            rgb = img.convert("RGB")
            m = grow_mask(binarize(mask, rgb.size), max(4, round(0.004 * max(rgb.size))) if grow is None else grow)
            box = crop_box(m, 0.4, 64, rgb.size)
            crop = rgb.crop(box); cm = m[box[1]:box[3], box[0]:box[2]]
            f = SDXL_SIDE / max(crop.size)
            sw, sh = max(64, round(crop.width * f / 8) * 8), max(64, round(crop.height * f / 8) * 8)
            pipe = self._sdxl_pipe()
            res = pipe(prompt=prompt, negative_prompt=negative or None, image=crop.resize((sw, sh), Image.Resampling.LANCZOS),
                       mask_image=Image.fromarray(cm * 255).resize((sw, sh), Image.Resampling.NEAREST),
                       width=sw, height=sh, strength=0.99, num_inference_steps=30, guidance_scale=7.5).images[0]
            res = res.resize(crop.size, Image.Resampling.LANCZOS)
            blended = Image.composite(res, crop, feather(cm, 3.0))
            out = rgb.copy(); out.paste(blended, box[:2])
            self._touch()
            return out

    def _need_sdxl(self) -> None:
        if self._sdxl is None:
            if not self.has("sdxl-inpaint"):
                raise ModelError("Stable Diffusion XL Inpainting is not downloaded. It is optional: Model, then Editing models.")
            if "sdxl-inpaint" not in self._verified:
                registry.verify(self.root, "sdxl-inpaint")
                self._verified.add("sdxl-inpaint")
