# Veil

A local, private, browser-based photo editor for professional boudoir and portrait
photographers, and the models they work with. Photos stay on your machine. No telemetry.

> Work in progress: phase 5 of 9. Screenshots, setup steps and hardware
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

### Optional: generative fill

| Model | Used for | Download | License |
|---|---|---|---|
| Stable Diffusion XL Inpainting 0.1 (fp16) | better removal of large areas; new backgrounds from a description | 6.9 GB | CreativeML Open RAIL++-M |

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

License: MIT (Veil itself).

