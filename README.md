# Veil

A local, private, browser-based photo editor for professional boudoir and portrait
photographers, and the models they work with. Photos stay on your machine. No telemetry.

> Work in progress: phase 9 of 9. Screenshots, setup steps and hardware
> requirements land with the later phases.

Veil is for photographs of consenting adults. It never swaps in a face from an
uploaded image.

## Install

Veil runs on Linux (64-bit Intel/AMD). The installer works on a computer that has never had Veil on it.

```
git clone <this repository> veil-studio
cd veil-studio
./install.sh --dry-run     # shows everything it would download or install, and changes nothing
./install.sh               # asks a few questions, shows each download with its size and license, then installs
```

What it does:

1. **Looks at your computer**: graphics card and video memory, memory, free disk space, Python, and any Ollama models you already have.
2. **Asks a few plain questions** (what you will mainly do, how typed commands should be understood, whether you want phone access, where to keep files)
   and **recommends Minimum, Recommended or High-end**. You can pick another.
3. **Shows every download** with its size and license, the total, and the free space needed, and waits for your yes.
4. **Checks every file** against a recorded SHA-256 and deletes any file that does not match. Interrupted downloads resume.
5. Asks before every step that needs your administrator password (`sudo`). It shows the exact command; you type your password in your own
   terminal. Run it with `--no-sudo` and it only prints those commands for you to run.
6. Installs Veil's own Python environment (nothing system-wide), the models, optionally Ollama and a language model, optionally Tailscale,
   a `veil` command and (if you want) a menu entry. Re-running it continues where it stopped and skips what is already right.
7. **Checks the finished install**: libraries load, every model matches its checksum, the editing engine is ready.

`./install.sh --uninstall` removes what it set up. It never touches your photos.

| Set | For | Models | Language model |
|---|---|---|---|
| Minimum | 4-6 GB video memory, or no NVIDIA card (slower) | SAM 2.1 tiny, BiRefNet lite, LaMa, face models | `qwen3.5:2b` (2.7 GB) |
| Recommended | 8-12 GB video memory | SAM 2.1 base+, BiRefNet, LaMa, face models | `qwen3.5:9b` (6.6 GB) |
| High-end | 16 GB or more video memory (and 32 GB memory) | SAM 2.1 large, BiRefNet, LaMa, face models | `qwen3.5:27b` (17.5 GB) |

The optional generative model (6.9 GB, needs an NVIDIA card) is offered on top of any set, with its license restrictions shown first.
Rough disk space: the Python libraries take about 6 GB with an NVIDIA card (3.2 GB to download), or about 1.3 GB on processor-only
computers. Ollama adds about 4 GB. The installer shows the exact figures for your choices and your free space.

**Graphics cards.** NVIDIA cards are supported. Without one, Veil installs the processor-only build and only the Minimum set
(slow, and no generative model). **AMD support is planned** (not in this version); AMD and Intel graphics are detected and
the installer says so, then uses the processor.

**Linux distributions.** Debian, Ubuntu and Mint (and derivatives that use `apt`) are fully supported: the installer can install missing
system packages and Tailscale for you. On **Fedora** and **Arch** it prints the commands to run yourself, then continues:

```
# Fedora
sudo dnf install -y python3 python3-pip curl zstd
# Arch
sudo pacman -S --needed python python-pip curl zstd
# Tailscale (any distribution): https://tailscale.com/download/linux
```

**Ollama without administrator rights.** If Ollama is not installed and you want a language model on your computer, the installer downloads
the official release (checked against the SHA-256 GitHub lists for the file), unpacks it into `~/.local` and, where your system has user
services, starts it at login. Models you already have in Ollama are reused. Ollama tags can be re-published, so the installer compares a fresh
download with the weights recorded in Veil and tells you if it is a newer build.

**Phone access (Tailscale)** is optional. The installer adds Tailscale's apt repository and package (with your approval), then walks you through
signing in on the computer and on the phone, step by step. It never types passwords or signs in for you.

**What is not verified by checksum.** Python libraries come from PyPI over HTTPS and pip checks each file against the hash the index lists; the
versions are not pinned to hashes in this version. Language models are checked by Ollama itself. Everything else is pinned in
`backend/veil/vision/registry.py`.

## First launch tutorial

The first time you open the editor, a short tour explains every tool and walks through phone and tablet setup over Tailscale. **Skip tutorial** (or Esc)
ends it, and once it is skipped or finished it does not appear again by itself. The **Help** button replays it, with a menu to jump to any step.
It is remembered in that browser only, makes no network requests, and a paired phone gets a shorter version.

## GPU worker (optional)

A GPU worker is a second computer with a strong NVIDIA card, at home or rented by the hour, that does Veil's heavy picture jobs
(selecting, finding the person, filling in, finding faces, generative fill) for the computer you edit on. It is useful when your own computer has
no NVIDIA card or a small one. Typed commands are **not** run on the worker in this version; they are read on your own computer.

How it behaves:

- **Tailscale only.** The worker listens on its Tailscale address and on itself, never on your Wi-Fi or the internet. Veil only talks to a
  Tailscale address (or a name that resolves to one), checked on every request, and ignores proxy settings.
- **One-time pairing.** On the worker you run `veil-worker pair`; it prints a code that works once, for 10 minutes. You type the code into Veil.
  After five wrong guesses every pending code is cancelled. The worker then trusts that one computer through a long random token; the worker
  keeps only a hash of it. Remove a computer with `veil-worker clients` and `veil-worker revoke ID`.
- **The worker keeps nothing.** A photo is held in memory for the job (at most 15 minutes, six pictures), never written to disk, and the
  worker has no accounts, no history and no telemetry. Jobs take only the photo being edited and its selection: there is no job that takes a
  second picture, so no face can be copied from one picture into another.
- **Automatic fallback.** If the worker is off, busy, unreachable or no longer trusts this computer, Veil does the job on your computer and shows
  a notice saying so (and "Ran on ..." after each job that used the worker). If your computer cannot do the job either, the message
  gives both reasons.
- **You decide per computer.** The "GPU worker" button (owner at the computer only) pairs, switches the worker on or off, tests it and forgets it.
  A paired phone can see where a job ran but cannot change the worker.

What you are trusting: a worker you pair sees your photos while it works on them. Pair only machines you control. On a rented machine,
the provider can in principle see its memory; do not use a rented worker for photos you would not hand to that provider.

### Set up a worker on your own second computer

1. On that computer, install Veil the normal way, but run `./install.sh --worker`. It sets up the libraries and the models for that
   card, asks no question about language models or phones, and installs Tailscale if needed.
2. Connect it to Tailscale (the installer prints these steps): `sudo tailscale up`, open the address it prints, sign in with the **same** account as your
   own computer, click **Connect**.
3. Start the worker: `veil-worker`. It prints the address to use, for example `100.101.102.103:5196`.
4. In a second terminal on the worker: `veil-worker pair`. It prints a code like `ABCDE-FGHJK`.
5. On your own computer, in Veil, click **GPU worker**, type the address and the code, click **Pair**. It should say
   "Connected to ..." with the card's name. Leave **Use the worker for heavy jobs** ticked.

### Set up a worker on a rented cloud GPU machine

Any provider works if it gives you a Linux machine (Ubuntu 22.04 or 24.04 recommended) with an NVIDIA card (8 GB or more; 12 GB or more if you
want generative fill), about 40 GB of disk, a terminal (SSH or a web terminal) and outbound internet. Menus differ between providers, so the
provider steps are described by what to look for. **You pay for the machine while it exists. Stop or delete it when you are done.**

1. **Make an auth key** (so the rented machine can join your Tailscale network without a browser).
   In your own browser go to <https://login.tailscale.com/admin/settings/keys> and sign in with the same account you use on your own computer.
   Click **Generate auth key...**. Type a description such as `veil-worker`. Switch **Ephemeral** on (the machine disappears from your network
   when it goes offline). Leave **Reusable** off. Set the expiry to 1 day. Click **Generate key**, copy the key (it starts with `tskey-`) and keep it
   private: treat it like a password. You will not see it again.
2. **Rent the machine.** At your provider choose a GPU, the Ubuntu image, and at least 40 GB of disk. Start it, then open its terminal.
3. **Get Veil onto it.** In that terminal: `git clone <the Veil repository address> veil-studio` and `cd veil-studio`.
   (If the machine has no `git`: `apt-get update && apt-get install -y git`, or copy the folder with `scp`.)
4. **Install the worker:** `./install.sh --worker`. On rented machines you are usually `root`; the installer says so and runs its administrator
   steps directly. Press Enter to accept the recommended set (it reads the card's memory), or pick another. Wait for it to finish; the models
   are downloaded each time on a fresh machine (about 1 GB for Recommended, plus 6.9 GB if you add generative fill).
5. **Join Tailscale** with the key from step 1 (replace the part after `=`):
   `tailscale up --auth-key=tskey-PASTE-YOUR-KEY-HERE`
   - If you get "tailscaled is not running" or no `/dev/net/tun` (common inside containers), start Tailscale in userspace mode first:
     `tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &`, run the `tailscale up` command above, and then once:
     `tailscale serve --bg --tcp 5196 tcp://localhost:5196`. The worker prints this help itself when it sees this situation.
   - Check: `tailscale ip -4` prints an address starting with `100.`.
6. **Start the worker:** `veil-worker` (if the machine says command not found, run `~/.local/bin/veil-worker`). Leave this terminal open (or run it inside `tmux` or `screen`). It prints the address to use.
7. **Pair:** open a second terminal on the machine and run `veil-worker pair`. Copy the code.
8. **On your own computer** (Tailscale switched on), open Veil, click **GPU worker**, type the worker's Tailscale address (from step 6, or
   `tailscale ip -4`), type the code, click **Pair**. The dialog should say "Connected to ..." with the rented card's name.
9. **When you are done:** in Veil click **GPU worker**, then **Forget this worker** (click it twice). Then stop or delete the machine at your provider.
   Optionally remove it from <https://login.tailscale.com/admin/machines> (the three dots next to it, then **Delete**). Ephemeral machines
   disappear by themselves after a while.

Tested with: `./install.sh --worker` as root on a clean Ubuntu 24.04 container (processor build; pairing and real jobs worked), and a worker with an
NVIDIA RTX 3060 listening on its Tailscale address and used by a second Veil on the same computer. It has **not** been tried on a real rented
machine, with a second physical computer, or in Tailscale's userspace mode; if a step above does not match what you see, please report it.

If something does not work:

| You see | What it means and what to do |
|---|---|
| "The worker could not be reached" | The worker is not running, or Tailscale is off at one end. On the worker: is `veil-worker` running, does `tailscale status` show it connected? On your computer: is Tailscale on? Jobs ran on your computer meanwhile. |
| "... is not a Tailscale address" | The address is a normal network or internet address. Use the one starting with `100.` or the `.ts.net` name. |
| "That code is not right, or it has expired" | Codes work once, for 10 minutes. Run `veil-worker pair` again. |
| "The worker no longer trusts this computer" | It was revoked, or its data folder was reset. Pair again. |
| "This is not the worker that was paired" | A different machine answers at that address (for example a new rental reusing it). Forget the worker and pair again if you trust it. |
| "The worker is busy" | It runs one job at a time and another is still running; Veil used your computer for this job. |

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
| Segment Anything 2.1, `base+` (Meta) | click / box to outline (Recommended) | 324 MB | Apache-2.0 |
| Segment Anything 2.1, `tiny` (Meta) | the same, smaller (Minimum) | 156 MB | Apache-2.0 |
| Segment Anything 2.1, `large` (Meta) | the same, most precise (High-end) | 898 MB | Apache-2.0 |
| LaMa `big-lama` | fills in removed objects and tattoos | 206 MB | Apache-2.0 |
| BiRefNet | finds the person for background tools (Recommended, High-end) | 444 MB | MIT |
| BiRefNet lite | the same, smaller (Minimum) | 178 MB | MIT |

- LaMa: the code repository (advimman/lama) is Apache-2.0. Veil downloads the weights from a
  third-party re-host (enesmsahin/simple-lama-inpainting, release v0.1.0); its SHA-256 is recorded in
  `backend/veil/vision/registry.py`.
- SAM tiny and large: Meta publishes no checksums, so the SHA-256 recorded in `registry.py` is from the first download. (`small` is not offered.)
- BiRefNet lite: its SHA-256 matches the one Hugging Face publishes. It runs the same vendored code with a smaller backbone (one index differs).
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

