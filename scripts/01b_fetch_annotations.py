"""
01b_fetch_annotations.py — enrich chains with function / localisation / family / PTM labels.

Maps each PDB+chain to a UniProt accession via the SIFTS flatfile (one download, no
per-PDB API calls), then queries UniProt in rate-limited batches for whole-protein labels
the models never trained on:

    ec_class       enzyme commission top class 1..7 (0 = non-enzyme)   -> function
    enzyme         1 if EC present else 0
    localisation   bucketed primary subcellular location                -> localisation
    kingdom        Bacteria / Archaea / Eukaryota / Viruses             -> taxonomy
    pfam           first Pfam family id                                 -> family
    is_phospho     1 if 'Phosphoprotein' keyword                        -> PTM (proxy)
    is_glyco       1 if 'Glycoprotein' keyword                          -> PTM (proxy)
    n_ptm          count of MOD_RES + CARBOHYD features                 -> PTM load

(CATH architecture/topology/homologous-superfamily are already in index.jsonl — no fetch
needed — and cover the fold/family/domain axis structurally.)

Rate limiting: UniProt is queried in batches of --batch accessions with --sleep seconds
between calls.

Outputs (under --structures-dir):
    pdb_chain_uniprot.tsv.gz   cached SIFTS mapping
    annotations.jsonl          one record per chain (id + the labels above)

Usage:
    uv run python scripts/01b_fetch_annotations.py --structures-dir /ssc/structures
"""

import argparse
import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

SIFTS_URL = ("https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/"
             "pdb_chain_uniprot.tsv.gz")
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,ec,cc_subcellular_location,lineage,xref_pfam,keyword,ft_mod_res,ft_carbohyd"

LOC_BUCKETS = [  # (substring to match in the location text, bucket label) — first match wins
    ("cell membrane", "membrane"), ("plasma membrane", "membrane"),
    ("membrane", "membrane"), ("secreted", "secreted"),
    ("cytoplasm", "cytoplasm"), ("cytosol", "cytoplasm"),
    ("nucleus", "nucleus"), ("mitochond", "mitochondrion"),
    ("endoplasmic reticulum", "ER"), ("golgi", "golgi"),
    ("periplasm", "periplasm"), ("cell wall", "cell_wall"),
    ("lysosome", "lysosome"), ("peroxisome", "peroxisome"),
    ("chloroplast", "plastid"), ("plastid", "plastid"),
]
KINGDOMS = ("Bacteria", "Archaea", "Eukaryota", "Viruses")

# Organism-agnostic functional "protein classes" from UniProt keywords (unlike the
# human-gene-centric Human Protein Atlas classes). Binary one-vs-rest flags:
KW_FLAGS = {
    "is_transport": ["Transport"],
    "is_dna_binding": ["DNA-binding"],
    "is_rna_binding": ["RNA-binding"],
    "is_kinase": ["Kinase"],
    "is_ribosomal": ["Ribosomal protein", "Ribonucleoprotein"],
    "is_metal_binding": ["Metal-binding"],
    "is_membrane_protein": ["Transmembrane"],
    "is_structural": ["Cytoskeleton", "Structural protein", "Muscle protein",
                      "Intermediate filament", "Keratin", "Actin-binding"],
    "is_immune": ["Immunity", "Innate immunity", "Adaptive immunity", "Immunoglobulin domain",
                  "Antimicrobial", "Complement pathway", "MHC", "Inflammatory response"],
}
# Single dominant class (priority order; enzyme EC as fallback):
PROTEIN_CLASS_PRIORITY = [
    ("gpcr", ["G-protein coupled receptor"]),
    ("ion_channel", ["Ion channel"]),
    ("receptor", ["Receptor"]),
    ("immune", KW_FLAGS["is_immune"]),
    ("structural", KW_FLAGS["is_structural"]),
    ("ribosomal", KW_FLAGS["is_ribosomal"]),
    ("chaperone", ["Chaperone"]),
    ("kinase", ["Kinase"]),
    ("dna_binding", ["DNA-binding"]),
    ("rna_binding", ["RNA-binding"]),
    ("transport", ["Transport"]),
]


def _download_sifts(cache: Path) -> Path:
    if cache.exists() and cache.stat().st_size > 0:
        return cache
    print(f"  downloading SIFTS -> {cache} ...")
    with urllib.request.urlopen(SIFTS_URL, timeout=180) as r:
        cache.write_bytes(r.read())
    return cache


def _load_sifts(cache: Path) -> dict:
    """(pdb_lower, chain) -> primary UniProt accession."""
    mapping = {}
    with gzip.open(cache, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("PDB\t"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            key = (p[0].lower(), p[1])
            mapping.setdefault(key, p[2])   # first segment's primary acc
    return mapping


def _bucket_loc(text: str) -> str:
    if not text:
        return "unknown"
    t = text.lower().split("subcellular location:", 1)[-1]
    for sub, lab in LOC_BUCKETS:
        if sub in t:
            return lab
    return "other"


def _parse_uniprot_tsv(tsv: str) -> dict:
    """Return acc -> label dict from a UniProt search TSV response."""
    out = {}
    lines = tsv.splitlines()
    if not lines:
        return out
    hdr = lines[0].split("\t")
    idx = {h: i for i, h in enumerate(hdr)}
    for line in lines[1:]:
        c = line.split("\t")
        if len(c) < len(hdr):
            c += [""] * (len(hdr) - len(c))
        acc = c[idx["Entry"]]
        ec = c[idx.get("EC number", -1)] if "EC number" in idx else ""
        ec_class = int(ec.strip()[0]) if ec.strip()[:1].isdigit() else 0
        loc = _bucket_loc(c[idx.get("Subcellular location [CC]", -1)] if "Subcellular location [CC]" in idx else "")
        lineage = c[idx.get("Taxonomic lineage", -1)] if "Taxonomic lineage" in idx else ""
        kingdom = next((k for k in KINGDOMS if k in lineage), "other")
        pfam_raw = c[idx.get("Pfam", -1)] if "Pfam" in idx else ""
        pfam = pfam_raw.split(";")[0] if pfam_raw else ""
        kw = c[idx.get("Keywords", -1)] if "Keywords" in idx else ""
        mod = c[idx.get("Modified residue", -1)] if "Modified residue" in idx else ""
        carb = c[idx.get("Glycosylation", -1)] if "Glycosylation" in idx else ""
        n_ptm = mod.count("MOD_RES") + carb.count("CARBOHYD")
        rec = {
            "uniprot": acc, "ec_class": ec_class, "enzyme": int(ec_class > 0),
            "localisation": loc, "kingdom": kingdom, "pfam": pfam,
            "is_phospho": int("Phosphoprotein" in kw), "is_glyco": int("Glycoprotein" in kw),
            "n_ptm": n_ptm,
        }
        for fname, subs in KW_FLAGS.items():
            rec[fname] = int(any(s in kw for s in subs))
        pc = next((lab for lab, subs in PROTEIN_CLASS_PRIORITY if any(s in kw for s in subs)), None)
        rec["protein_class"] = pc or (f"enzyme_ec{ec_class}" if ec_class > 0 else "other")
        out[acc] = rec
    return out


def _query_uniprot(accs: list[str], batch: int, sleep: float) -> dict:
    result = {}
    for i in range(0, len(accs), batch):
        chunk = accs[i:i + batch]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = f"{UNIPROT}?" + urllib.parse.urlencode(
            {"query": q, "fields": FIELDS, "format": "tsv", "size": batch})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    result.update(_parse_uniprot_tsv(r.read().decode()))
                break
            except Exception as exc:
                print(f"    batch {i//batch} attempt {attempt+1} failed: {exc}")
                time.sleep(sleep * (attempt + 2))
        print(f"  UniProt {min(i+batch, len(accs)):,}/{len(accs):,}")
        time.sleep(sleep)   # be polite between batches
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch UniProt function/localisation/family/PTM labels")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between UniProt calls")
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
    sifts = _load_sifts(_download_sifts(sdir / "pdb_chain_uniprot.tsv.gz"))
    print(f"  SIFTS: {len(sifts):,} (pdb,chain) mappings")

    recs = [json.loads(l) for l in (sdir / "index.jsonl").open() if l.strip()]
    id_to_acc = {}
    for r in recs:
        acc = sifts.get((r["pdb"].lower(), r["chain"]))
        if acc:
            id_to_acc[r["id"]] = acc
    accs = sorted(set(id_to_acc.values()))
    print(f"  {len(id_to_acc):,}/{len(recs):,} chains mapped to {len(accs):,} UniProt accessions")

    ann = _query_uniprot(accs, args.batch, args.sleep)
    print(f"  UniProt returned {len(ann):,} entries")

    out = sdir / "annotations.jsonl"
    n = 0
    with out.open("w") as f:
        for cid, acc in id_to_acc.items():
            a = ann.get(acc)
            if a:
                f.write(json.dumps({"id": cid, **a}) + "\n")
                n += 1
    print(f"  wrote {n:,} annotation records -> {out}")


if __name__ == "__main__":
    main()
