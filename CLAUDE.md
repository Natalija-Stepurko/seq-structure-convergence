# Python Package Management with uv

Use uv exclusively for Python package management in this project.

## ⚠️ Dedicated virtual environment (do NOT clobber the materials venv)

This machine's `~/.bashrc` sets a **global** `UV_PROJECT_ENVIRONMENT=/scratch/.venv`,
which points at the *material-science-ml* project's venv. Running `uv` here without
overriding it would install this project's dependencies into that venv.

**Always override it for this project** by exporting the dedicated path first:

```bash
export UV_PROJECT_ENVIRONMENT=/scratch/.venv-ssc   # this project's own venv
```

`UV_CACHE_DIR` (`/scratch/.uv-cache`) and `HF_HOME` (`/scratch/.hf-cache`) can stay
shared — they are keyed by package/model name and will not collide.

## Commands
- Install: `uv add <package>`
- Remove: `uv remove <package>`
- Sync from lockfile: `uv sync`
- Run scripts: `uv run <script>.py`
- Launch Jupyter: `cd notebooks && uv run jupyter lab`

## Hardware note
This instance is **CPU-only** (8 cores, ~165 GB RAM, no GPU). Model choices and
batch sizes assume CPU inference — see `paper/PROJECT_PLAN.md`.
