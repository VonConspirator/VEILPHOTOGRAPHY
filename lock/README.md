# Hash-checked library lists

Every Python library Veil installs is pinned here to an exact version and to the SHA-256 of each file pip may install.

| File | What it is |
|---|---|
| `py312/gpu.lock` | NVIDIA build (PyTorch with CUDA from PyPI, OpenCV, MediaPipe, SAM 2.1 dependencies, web server) |
| `py312/gpu-sdxl.lock` | the same plus diffusers, transformers and accelerate for generative fill |
| `py312/cpu.lock` | processor-only build (PyTorch `+cpu` from download.pytorch.org, no CUDA packages) |
| `py312/source.lock` | `sam2`, `iopath`, `antlr4-python3-runtime`: published only as source, installed second from hash-checked source |
| `MANIFEST.json` | which Python versions have lists |
| `make_lock.py` | regenerates the lists |

The installer uses them with `pip install --require-hashes --no-deps --only-binary=:all: -r <list>` and then
`pip install --require-hashes --no-deps --no-build-isolation -r source.lock` (with `SAM2_BUILD_CUDA=0`), and finally points the environment at `backend/`
with a one-line `.pth` file. Only wheels for 64-bit Linux (glibc) and pure-Python wheels are listed, so a file for another platform is refused.

## Regenerating (when a library changes)

1. In a Veil environment with every extra installed, update the libraries, then run the whole test suite (`pytest`). The lists must describe a tested set.
2. `.venv/bin/python lock/make_lock.py`. It reads the installed versions, asks PyPI (and PyTorch's CPU index) for each file's SHA-256, and rewrites
   `lock/py3NN/*.lock` and `MANIFEST.json`. It downloads no package files.
3. `pytest backend/tests/test_lock.py` (shape, coverage and enforcement), then try an install on a clean machine or container.
4. Review the diff: a changed version or an unexpected new package is visible line by line.

To support another Python (say 3.13), run `make_lock.py` with that Python inside an environment built for it; it adds `py313/` to the manifest.
Pins are exact on purpose: there is no automatic upgrade path.
