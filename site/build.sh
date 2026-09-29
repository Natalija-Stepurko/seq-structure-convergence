#!/usr/bin/env bash
# Rebuild the results page from results/ and audit every headline number against its source.
#   site/build.sh                  archived map images (the published page)
#   site/build.sh --fresh-figures  re-render the maps from results/umap6/coords.npz first
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

if [[ "${1:-}" == "--fresh-figures" ]]; then
  "$PY" make_figures.py
fi
"$PY" build_data.py
"$PY" build.py "$@"
"$PY" audit.py
