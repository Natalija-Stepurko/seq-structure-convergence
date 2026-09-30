# Ladder with scrambled-residue nulls

`04_convergence.py ladder-ci --with-null --peaks-from results/ladder_ci/ladder_ci.csv` with the
same budget, subsample count and seed as `ladder_ci/`. Each row adds `null_mean`, `null_lo`,
`null_hi` and `null_share`: the same measure on the same subsample with the correspondence between
the two models' residues shuffled.

The scrambles draw from their own random stream and the peak layer pairs are read from
`ladder_ci/`, so the chain subsamples are identical to that run. 70 of 72 rows reproduce its
estimates exactly. The other two are mutual k-NN for the two seeded ESM-1v copies
(raw 0.8495 → 0.8499, partial 0.7960 → 0.7961; intervals shift by at most 0.001). Those two models
give near-identical distances, so which neighbour falls inside the k = 10 cut-off depends on the
last digits of the floating-point arithmetic, which vary with the number of threads.
