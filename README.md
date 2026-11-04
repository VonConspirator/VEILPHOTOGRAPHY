# Veil

A local, private, browser-based photo editor for professional boudoir and portrait
photographers, and the models they work with. Photos stay on your machine. No telemetry.

> Work in progress: phase 7 of 9. Screenshots, setup steps and hardware
> requirements land with the later phases.

Veil is for photographs of consenting adults. It never swaps in a face from an
uploaded image.

## Models and licenses

Veil's own code is MIT-licensed. The models it uses are separate projects with their own licenses.
Veil only suggests permissive licenses (Apache-2.0, MIT) by default. Every file Veil downloads
is checked against a recorded SHA-256 checksum and is refused if it does not match.

### Language models (read your typed commands; run in Ollama)

| Tier | Model | Download | License |
|---|---|---|---|
| Minimum | `qwen3.5:2b` | 2.7 GB | Apache-2.0 |
| Recommended | `qwen3.5:9b` | 6.6 GB | Apache-2.0 |
| High-end | `qwen3.5:27b` | 17 GB | Apache-2.0 |

Licenses read from each model's page on ollama.com. You can point Veil at any other Ollama,
OpenAI-compatible or Anthropic model; then that model's license and terms are yours to check.

### Editing models (selection, object and tattoo removal, background)

| Model | Used for | Download | License |
|---|---|---|---|
| Segment Anything 2.1, `base+` (Meta) | click / box to outline | 324 MB | Apache-2.0 |
| LaMa `big-lama` | fills in removed objects and tattoos | 206 MB | Apache-2.0 |
| BiRefNet | finds the person for background tools | 444 MB | MIT |

- LaMa: the code repository (advimman/lama) is Apache-2.0. Veil downloads the weights from a
  third-party re-host (enesmsahin/simple-lama-inpainting, release v0.1.0); its SHA-256 is recorded in
  `backend/veil/vision/registry.py`.
- BiRefNet: the code is vendored in `backend/veil/vision/vendor/` (MIT, notice in `LICENSE-BiRefNet`),
  pinned to one revision, with a small change that removes its dependency on the `transformers` library.
- Python libraries (PyTorch, OpenCV, numpy, timm, einops, kornia, safetensors, sam2 and its helpers)
  keep their own licenses; see each project. PyTorch's NVIDIA CUDA libraries are under NVIDIA's own license.

### Face models (find faces, hide faces, look edits)

| Model | Used for | Download | License |
|---|---|---|---|
| YuNet (OpenCV Zoo) `face_detection_yunet_2023mar` | finds faces | 233 KB | MIT |
| MediaPipe Face Landmarker (`face_landmarker.task`, Google) | maps 478 points on a face (outline, eyes, irises, lips) | 3.8 MB | Apache-2.0 |

- YuNet's SHA-256 matches the one OpenCV Zoo publishes.
- The Face Landmarker license was read from the three model cards that make up the bundle (BlazeFace
  detector, Face Mesh V2, Blendshape V2); each says "Licensed under Apache License, Version 2.0".
  Google publishes no checksum for the file, so the SHA-256 recorded in `registry.py` is the one taken
  when it was first downloaded.
- The `mediapipe` Python library (Apache-2.0) is installed with `pip install --no-deps mediapipe`
  plus `pip install absl-py flatbuffers matplotlib sounddevice cffi` (or `pip install veil-studio[faces]`
  for the second part). `--no-deps` matters: mediapipe asks for `opencv-contrib-python`, which would
  overwrite the `opencv-python-headless` Veil already uses.
- Veil never takes a face from another picture. Nothing in the face tools accepts an image, a reference or
  an identity of any kind; the generator is only ever given your own photo, a mask and fixed words
  (tests check this).

### Optional: generative model

| Model | Used for | Download | License |
|---|---|---|---|
| Stable Diffusion XL Inpainting 0.1 (fp16) | generated faces, makeup removal, hairstyle and facial hair changes; better removal of large areas; new backgrounds from a description | 6.9 GB | CreativeML Open RAIL++-M |

This model is not downloaded unless you ask for it (Model, then Editing models). It also needs
`pip install diffusers transformers accelerate`. Its license is **not** Apache/MIT: it
carries use restrictions that apply to you, and the license asks that they travel with the model if you
share it. In summary, you agree not to use it: in violation of any law; to exploit or harm
minors; to generate or spread verifiably false information meant to harm others; to generate personal
information that can be used to harm someone; to defame, disparage or harass; for fully automated
decisions that affect someone's legal rights; to discriminate against or harm people or groups; to
exploit vulnerabilities of a group because of age or social, physical or mental characteristics; to give
medical advice or interpret medical results; or for law-enforcement, justice or immigration profiling.
Read the full text: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md
(this inpainting model carries the same `openrail++` license). The editor for consenting adults only is
Veil's own rule on top of that.

### Watermarks and fonts

| Item | Used for | Download | License |
|---|---|---|---|
| Inter (variable font, google/fonts `ofl/inter` @ `0b58fb37`) | text watermarks | 0.9 MB, bundled with Veil (`backend/veil/assets/fonts`) | SIL Open Font License 1.1 |

The font and its license text (`Inter-OFL.txt`) ship inside Veil. Its SHA-256 is pinned in the code and checked
before it is used (`29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031`). Inter draws Latin,
Greek and Cyrillic letters, digits and common symbols; for anything else, use a logo.

**How watermarks work.** Make presets with the **Watermarks** button (at the computer only): text (words, weight,
colour, soft outline) or a logo (a PNG, JPEG or WebP you upload; SVG logos must be exported as PNG first),
placed in one of nine spots or repeated across the photo, with size, strength, distance from the edge and tilt.
A watermark is added **only when you export**, never to your working copy or its history. In the Export
window tick "Add a watermark" and choose a preset. A paired phone can switch the watermark on or off and
choose a saved preset, but presets can only be created, changed or deleted at the computer. Logos are stored as clean
PNGs (no hidden details) in the data folder and are only ever drawn on exported copies; they are never given to a
face or AI tool.

License: MIT (Veil itself).

