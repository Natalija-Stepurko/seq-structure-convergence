# Python Package Management with uv

Use uv exclusively for Python package management in this project.

## Project location

Work in **`/ssc`** — a top-level path, deliberately kept separate from the materials
project at `/data/project`. `/ssc` is a **bind mount** of `/data/.ssc` (the backing
store on the 8 TB `/data` NVMe — the only disk large enough for embedding data). The
mount is persisted in `/etc/fstab` (`/data/.ssc /ssc none bind,nofail 0 0`), so it
survives reboots. Always refer to this project as `/ssc`; never touch `/data/project`
(materials) or `/data/.ssc` directly.

## ⚠️ Dedicated virtual environment (do NOT clobber the materials venv)

This machine's `~/.bashrc` sets a **global** `UV_PROJECT_ENVIRONMENT=/scratch/.venv`,
which points at the *material-science-ml* project's venv. Running `uv` here without
overriding it would install this project's dependencies into that venv.

**Always override it for this project** by exporting the dedicated path first:

```bash
export UV_PROJECT_ENVIRONMENT=/scratch/.venv-ssc   # this project's own venv
```

`UV_CACHE_DIR` (`/scratch/.uv-cache`), `HF_HOME` (`/scratch/.hf-cache`) and
`TORCH_HOME` (`/scratch/.torch-hub`, where fair-esm/ESM weights land) can stay
shared — they are keyed by package/model name and will not collide. **Set
`TORCH_HOME` before extracting embeddings** or the ~2.5 GB ESM-2 650M weights land
on the small root disk.

## Commands
- Install: `uv add <package>`
- Remove: `uv remove <package>`
- Sync from lockfile: `uv sync`
- Run scripts: `uv run <script>.py`
- Launch Jupyter: `cd notebooks && uv run jupyter lab`

## Hardware note
This instance is **CPU-only** (8 cores, ~165 GB RAM, no GPU). Model choices and
batch sizes assume CPU inference — see `paper/PROJECT_PLAN.md`.
