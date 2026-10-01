# Results page

Builds the results page from `results/` and checks it.

```bash
site/build.sh                  # needs numpy
site/build.sh --fresh-figures  # also re-renders the three map images (needs matplotlib)
```

| Step | File | Does |
|---|---|---|
| 1 | `build_data.py` | reads `results/` into `build/data.json`: the ladder, calibration, width series, layer grids, stitching grid, probes, dataset counts |
| 2 | `build.py` | renders `build/index.html`, the page GitHub Pages serves; checks tag nesting and anchors |
| 3 | `audit.py` | re-derives each headline number from `results/`, checks it appears in the page, and fails on any mismatch or banned phrase |
| 4 | `check_layout.py` | opens the page in a headless browser at desktop and phone width and fails if any chart label overlaps another or runs off the chart (needs `playwright`) |
| — | `charts.py` | the hand-written SVG charts |
| — | `make_figures.py` | renders the three map images from `results/umap6/coords.npz` into `build/figures.json` |

On every push to `main` that touches `site/` or `results/`, `.github/workflows/pages.yml` runs the
build, the audit and the layout check, and deploys `index.html` to GitHub Pages.

## What is archived

Most of the page is generated from `results/` on every build. Three inputs are carried over from
the earlier version of the page, in `archive/`:

- `pngs_opt.json`: the three map images, rendered by `make_figures.py` from
  `results/umap6/coords.npz` and re-encoded losslessly (pixel-identical, ~10% smaller).
  `--fresh-figures` swaps in a new render.
- `fragments.json`: nine HTML blocks lifted from the earlier page: the model cards and the
  reference-model table, the explainer diagrams for the three measures, the subtraction schematic,
  the layer × layer heatmaps, the win/loss chart and its companion panels, the 25-property battery
  and the layer curves. The numbers inside them come from the same result files and are covered by the audit
  where they are headline figures.
- `base_css.txt`: the page stylesheet.

## Sources

Every headline number has a file in `results/` behind it, and `audit.py` checks it on every build.
The scrambled-data calibration and the stitching result come from the reruns in
`results/ladder_ci_null/` and `results/stitch_grid_ci/`.
