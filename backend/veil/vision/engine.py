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
        self._yunet = self._landmarker = None
        self._sam_key = None
        self._sam_name = self._bn_name = None
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
                           "installed": registry.installed(self.root, name), "extra_packages": m.get("extra_packages", ""),
                           "libs_ok": {"sdxl-inpaint": None, "face-landmarker": None}.get(name, True)})
        # ready = every single-file model is there, and at least one variant of each choice (SAM size, BiRefNet size)
        groups = {m["group"] for m in registry.MODELS.values() if m.get("group")}
        base = all(x["installed"] for x in models if x["required"] and not registry.MODELS[x["name"]].get("group")) and \
            all(any(x["installed"] for x in models if registry.MODELS[x["name"]].get("group") == g) for g in groups)
        try:
            import diffusers  # noqa: F401
            gen = True
        except ImportError:
            gen = False
        try:
            import mediapipe  # noqa: F401
            faces_libs = True
        except ImportError:
            faces_libs = False
        for x in models:
            if x["name"] == "sdxl-inpaint":
                x["libs_ok"] = gen
            elif x["name"] == "face-landmarker":
                x["libs_ok"] = faces_libs
        return {"libraries": self.libraries_ok(), "generative_libs": gen, "faces_libs": faces_libs, "device": self.device(), "models": models,
                "ready": base and self.libraries_ok()}

    def has(self, name: str) -> bool:
        return registry.installed(self.root, name)

    def pick(self, group: str) -> str:
        """The best installed variant of a choice (largest SAM, full BiRefNet before lite)."""
        have = [n for n, m in registry.MODELS.items() if m.get("group") == group and self.has(n)]
        if not have:
            label = {"selection": "Segment Anything", "matte": "BiRefNet"}.get(group, group)
            raise ModelError(f"{label} is not downloaded yet. Open Model, then Editing models.")
        return max(have, key=lambda n: registry.MODELS[n]["rank"])

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
            self._yunet = self._landmarker = None
            self._sam_key = None
            self._sam_name = self._bn_name = None
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
            name = self.pick("selection")
            self._need(name)
            torch = _torch()
            if self._sam is None or self._sam_name != name:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                m = registry.MODELS[name]
                ckpt = str(self.root / m["files"][0]["path"])
                self._sam = SAM2ImagePredictor(build_sam2(m["sam_config"], ckpt, device=self.device()))
                self._sam_name, self._sam_key = name, None
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
    def _birefnet_model(self, name: str):
        if self._birefnet is None or self._bn_name != name:
            torch = _torch()
            from safetensors.torch import load_file
            from .vendor import birefnet as vb
            vb.Config.bb_index = registry.MODELS[name]["backbone"]          # 6 = Swin-L (full), 3 = Swin-T (lite)
            net = vb.BiRefNet(bb_pretrained=False)
            net.load_state_dict(load_file(str(self.root / registry.MODELS[name]["files"][0]["path"])))
            net = net.to(self.device()).eval()
            if self.device() == "cuda":
                net = net.half()
            self._birefnet, self._bn_name = net, name
        return self._birefnet

    def matte(self, img: Image.Image) -> Image.Image:
        """The person (or main subject) as a soft 'L' matte, white = foreground, same size as img."""
        with self._lock:
            name = self.pick("matte")
            self._need(name)
            torch = _torch()
            rgb = img.convert("RGB")
            net = self._birefnet_model(name)
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

    def generative_fill(self, img: Image.Image, mask: Image.Image, prompt: str, negative: str = "", grow: int | None = None, *,
                        strength: float = 0.99, seed: int | None = None, context: float = 0.4, steps: int = 30,
                        feather_px: float = 3.0) -> Image.Image:
        """Repaint the white area of `mask` following `prompt`. Works on a crop around the area at about 1024 px.
        `strength` 0..1: how far the model may depart from what is there (low = keep the structure, recolour or tidy).
        `context` is how much surrounding picture, as a fraction of the area's size, the model gets to look at.
        Nothing from any other picture is ever given to the model: only this photo, the mask and the words."""
        with self._lock:
            self._need_sdxl()
            torch = _torch()
            rgb = img.convert("RGB")
            m = grow_mask(binarize(mask, rgb.size), max(4, round(0.004 * max(rgb.size))) if grow is None else grow)
            box = crop_box(m, context, 64, rgb.size)
            crop = rgb.crop(box); cm = m[box[1]:box[3], box[0]:box[2]]
            f = SDXL_SIDE / max(crop.size)
            sw, sh = max(64, round(crop.width * f / 8) * 8), max(64, round(crop.height * f / 8) * 8)
            pipe = self._sdxl_pipe()
            gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
            res = pipe(prompt=prompt, negative_prompt=negative or None, image=crop.resize((sw, sh), Image.Resampling.LANCZOS),
                       mask_image=Image.fromarray(cm * 255).resize((sw, sh), Image.Resampling.NEAREST),
                       width=sw, height=sh, strength=float(strength), num_inference_steps=steps, guidance_scale=7.5,
                       generator=gen).images[0]
            res = res.resize(crop.size, Image.Resampling.LANCZOS)
            blended = Image.composite(res, crop, feather(cm, feather_px))
            out = rgb.copy(); out.paste(blended, box[:2])
            self._touch()
            return out

    # ------------------------------------------------------------ faces (YuNet finds them, MediaPipe maps them)
    def faces_ready(self) -> bool:
        try:
            import mediapipe  # noqa: F401
        except ImportError:
            return False
        return self.has("yunet") and self.has("face-landmarker")

    def detect_faces(self, img: Image.Image, min_score: float = 0.6) -> list[dict]:
        """Faces as [{'box': (x0, y0, x1, y1) in pixels, 'score': float}], biggest first."""
        import cv2
        with self._lock:
            self._need("yunet")
            rgb = np.asarray(img.convert("RGB"))
            h, w = rgb.shape[:2]
            f = min(1.0, 1600 / max(w, h))
            small = cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else rgb
            bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            if self._yunet is None:
                path = str(self.root / registry.MODELS["yunet"]["files"][0]["path"])
                self._yunet = cv2.FaceDetectorYN.create(path, "", (320, 320), min_score, 0.3, 50)
            self._yunet.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, rows = self._yunet.detect(bgr)
            out = []
            for r in (rows if rows is not None else []):
                x, y, bw, bh = (float(v) / f for v in r[:4])
                x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)
                if x1 - x0 >= 8 and y1 - y0 >= 8:
                    out.append({"box": (x0, y0, x1, y1), "score": float(r[14])})
            self._touch()
            return sorted(out, key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))

    def landmarks(self, img: Image.Image, box: tuple[float, float, float, float]) -> np.ndarray | None:
        """478 face landmark points in pixel coordinates for the face inside `box`, or None if the model cannot map it."""
        with self._lock:
            self._need("face-landmarker")
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            if self._landmarker is None:
                opts = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.root / registry.MODELS["face-landmarker"]["files"][0]["path"])),
                    num_faces=4, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
                self._landmarker = vision.FaceLandmarker.create_from_options(opts)
            rgb = img.convert("RGB")
            x0, y0, x1, y1 = box
            mg = 0.45 * max(x1 - x0, y1 - y0)
            cb = (max(0, int(x0 - mg)), max(0, int(y0 - mg)), min(rgb.width, int(x1 + mg)), min(rgb.height, int(y1 + mg)))
            crop = np.ascontiguousarray(np.asarray(rgb.crop(cb)))
            f = min(1.0, 1024 / max(crop.shape[:2]))
            if f < 1:
                import cv2
                crop = np.ascontiguousarray(cv2.resize(crop, (round(crop.shape[1] * f), round(crop.shape[0] * f)), interpolation=cv2.INTER_AREA))
            res = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            self._touch()
            if not res.face_landmarks:
                return None
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            best, best_d = None, 1e18
            for lm in res.face_landmarks:
                pts = np.array([[p.x, p.y] for p in lm], dtype=np.float64)
                pts[:, 0] = cb[0] + pts[:, 0] * (cb[2] - cb[0]); pts[:, 1] = cb[1] + pts[:, 1] * (cb[3] - cb[1])
                d = float(np.hypot(*(pts[1] - (cx, cy))))        # point 1 is the tip of the nose
                if d < best_d:
                    best, best_d = pts, d
            if best is None or best_d > 0.6 * max(x1 - x0, y1 - y0):
                return None                                      # it found some other face, not the one asked for
            return best

    def _need_sdxl(self) -> None:
        if self._sdxl is None:
            if not self.has("sdxl-inpaint"):
                raise ModelError("Stable Diffusion XL Inpainting is not downloaded. It is optional: Model, then Editing models.")
            if "sdxl-inpaint" not in self._verified:
                registry.verify(self.root, "sdxl-inpaint")
                self._verified.add("sdxl-inpaint")
