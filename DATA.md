# Data

## What is in this repository

| What | Where | Size | Tracked |
|---|---|---|---|
| Derived results: every agreement score, interval, probe score, stitching grid and map coordinate the results page uses | `results/` (index: [`results/README.md`](results/README.md)) | ~30 MB | yes |
| The chain set: 4,898 CATH S35 domains with PDB ID, chain, CATH code and length | `results/dataset/chains.csv` | 0.2 MB | yes |
| Archived page inputs: three map images and HTML fragments from the earlier page | `site/archive/` | 1.7 MB | yes |
| Structures: per-chain sequence, backbone coordinates and per-residue labels | rebuilt by `scripts/01_dataset.py` | 3.6 GB | no |
| Per-residue embeddings at every layer, nine model arms | rebuilt by `scripts/02_extract.py` | ~380 GB | no |
| The same embeddings on one fixed subsample of 35 whole chains (10,045 residues), with labels and reference scores | [Hugging Face](https://huggingface.co/datasets/NatalijaStepurko/seq-structure-convergence), written by `scripts/07_hf_dataset.py`; chain list in `results/dataset/hf_chains.csv` | 2.8 GB | published separately |
| One mean vector per protein for 4,894 proteins, every model and layer | same dataset (`07_hf_dataset.py --protein-level`); protein list in `results/dataset/hf_protein_ids.txt` | 1.4 GB | published separately |

The embeddings are the bulk: ESM-2 650M and the two ESM-1v checkpoints are ~112 GB each, ESM-2
35M and its untrained copy 16 GB each, CARP 11 GB, ESM-IF1 1.4 GB, ProteinMPNN and its untrained
copy 1.1 GB each. They are deterministic given the chain set and the model weights, so
`scripts/02_extract.py` regenerates them.

## Upstream sources and licences

| Source | Used for | Licence |
|---|---|---|
| [CATH](https://www.cathdb.info/) (S35 domain list) | the chain set; fold class, architecture and topology labels | CC BY 4.0 |
| [RCSB PDB](https://www.rcsb.org/) | backbone coordinates, sequences, B-factors | CC0 1.0 |
| [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/) (EMBL-EBI) | PDB chain → UniProt mapping | EMBL-EBI terms of use |
| [UniProt](https://www.uniprot.org/) | enzyme class, kingdom, protein class, binding, active and PTM sites | CC BY 4.0 |
| ESM-2, ESM-1v, ESM-IF1 ([fair-esm](https://github.com/facebookresearch/esm)) | sequence and structure models | MIT |
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | structure model; code vendored in `scripts/vendor/proteinmpnn/` | MIT |
| CARP-38M ([sequence-models](https://github.com/microsoft/protein-sequence-models)) | second sequence model | MIT |

The derived results in `results/` are released under this repository's MIT licence. When reusing
them, credit CATH (Sillitoe et al., 2021) and UniProt (The UniProt Consortium, 2023) for the labels
they were computed against, and SIFTS (Dana et al., 2019) for the chain mapping.
