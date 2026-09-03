r"""
01_dataset.py -- Build the protein dataset: chains, chain-level labels, per-residue targets.

Subcommands (each keeps the exact flags it had as a standalone script):

    fetch            CATH S35 chains + structures -> npz (seq, coords, SSE, RSA) + manifest
    annotate         UniProt chain labels via the SIFTS flatfile -> annotations.jsonl
    targets          per-residue binding/active/PTM sites and B-factor

The subcommand token is removed from argv before the original parser runs, so every command
line that worked before still works, with the subcommand inserted after the script name:

    uv run python scripts/01_dataset.py fetch --help

Provenance: every subcommand writes params.json beside its outputs (see qc_common.record_params).

Merged from:
    fetch            was 01_fetch_proteins.py
    annotate         was 01b_fetch_annotations.py
    targets          was 01c_residue_targets.py
"""

import argparse
import json
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import qc_common as qc
from urllib.request import urlopen
import numpy as np
import gzip
import urllib.parse
import urllib.request
import re





# ======================================================================================
# fetch  --  from 01_fetch_proteins.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

CATH_DOMAIN_LIST_URL = (
    "https://download.cathdb.info/cath/releases/latest-release/"
    "cath-classification-data/cath-domain-list.txt"
)

MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

MIN_LEN = 30

MAX_LEN = 1000

MAX_UNKNOWN_FRAC = 0.10   # fraction of non-standard residues tolerated

MAX_MISSING_CA_FRAC = 0.10

def _download_cath_list(cache: Path) -> Path:
    if cache.exists() and cache.stat().st_size > 0:
        print(f"  CATH domain list cached: {cache} ({cache.stat().st_size/1e6:.1f} MB)")
        return cache
    print(f"  Downloading CATH domain list -> {cache} ...")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(CATH_DOMAIN_LIST_URL, timeout=120) as r:
        cache.write_bytes(r.read())
    print(f"  done ({cache.stat().st_size/1e6:.1f} MB)")
    return cache

def _select_from_cath(cache: Path, limit: int | None, seed: int) -> list[dict]:
    """Parse the CATH domain list, keep one chain per S35 cluster (non-redundant)."""
    seen_s35: set[tuple] = set()
    seen_chain: set[str] = set()
    chains: list[dict] = []
    with cache.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            dom = p[0]                       # e.g. 1oaiA00
            c, a, t, h, s35 = p[1], p[2], p[3], p[4], p[5]
            pdb, chain = dom[:4], dom[4]
            s35_key = (c, a, t, h, s35)
            if s35_key in seen_s35:          # non-redundancy: one per S35 cluster
                continue
            cid = f"{pdb}_{chain}"
            if cid in seen_chain:
                continue
            seen_s35.add(s35_key)
            seen_chain.add(cid)
            chains.append({
                "id": cid, "pdb": pdb, "chain": chain,
                "cath_class": int(c), "cath_arch": int(a),
                "cath_topol": int(t), "cath_homol": int(h),
                "cath_code": f"{c}.{a}.{t}.{h}",
            })
    print(f"  CATH: {len(chains):,} non-redundant chains (one per S35 cluster)")
    if limit is not None and limit < len(chains):
        rng = random.Random(seed)
        chains = rng.sample(chains, limit)
        print(f"  Subsampled to {limit:,} (seed={seed})")
    return chains

def _select_from_pdb_list(list_path: Path) -> list[dict]:
    chains: list[dict] = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.replace(",", "_").split("_")
        pdb = tok[0].lower()
        chain = tok[1] if len(tok) > 1 else "A"
        chains.append({"id": f"{pdb}_{chain}", "pdb": pdb, "chain": chain,
                       "cath_class": None, "cath_arch": None,
                       "cath_topol": None, "cath_homol": None, "cath_code": None})
    print(f"  pdb-list: {len(chains):,} chains")
    return chains

def _featurise_chain(pdb_id: str, chain_id: str, raw_dir: Path) -> dict | None:
    """Fetch a structure and extract seq, backbone coords, SSE, RSA for one chain."""
    import biotite.database.rcsb as rcsb
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    path = rcsb.fetch(pdb_id, "bcif", str(raw_dir))
    atoms = pdbx.get_structure(pdbx.BinaryCIFFile.read(path), model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if atoms.array_length() == 0:
        return None
    ch = atoms[atoms.chain_id == chain_id]
    if ch.array_length() == 0:                       # chain id mismatch -> first chain
        ch = atoms[atoms.chain_id == atoms.chain_id[0]]
    if ch.array_length() == 0:
        return None

    res_ids, res_names = struc.get_residues(ch)
    L = len(res_ids)
    seq = "".join((struc.info.one_letter_code(r) or "X") for r in res_names)

    # backbone N, CA, C, O -> [L, 4, 3], NaN where an atom is missing
    coords = np.full((L, 4, 3), np.nan, dtype=np.float32)
    id_to_idx = {int(r): i for i, r in enumerate(res_ids)}
    for name, k in (("N", 0), ("CA", 1), ("C", 2), ("O", 3)):
        sub = ch[ch.atom_name == name]
        for rid, xyz in zip(sub.res_id, sub.coord):
            j = id_to_idx.get(int(rid))
            if j is not None:
                coords[j, k] = xyz

    ss3 = struc.annotate_sse(ch)                      # 'a'/'b'/'c', per residue
    if len(ss3) != L:                                 # annotate_sse can drop residues
        ss3 = np.array((list(ss3) + ["c"] * L)[:L])

    sasa_atom = struc.sasa(ch, vdw_radii="Single")
    sasa_res = struc.apply_residue_wise(ch, sasa_atom, np.nansum)
    rsa = np.array([
        (float(s) / MAX_ASA[rn]) if (rn in MAX_ASA and np.isfinite(s)) else np.nan
        for s, rn in zip(sasa_res, res_names)
    ], dtype=np.float32)

    return {"seq": seq, "coords_bb": coords, "ss3": np.asarray(ss3, dtype="<U1"),
            "rsa": rsa, "res_ids": np.asarray(res_ids, dtype=np.int32),
            "length": L, "actual_chain": str(ch.chain_id[0])}

def _load_done_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done = set()
    for line in jsonl_path.open():
        line = line.strip()
        if line:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done

def _validate_manifest(jsonl_path: Path) -> None:
    print("\n-- Validation pass --")
    records = [json.loads(l) for l in jsonl_path.open() if l.strip()]
    n_flagged = 0
    for rec in records:
        reasons = []
        if not (MIN_LEN <= rec["length"] <= MAX_LEN):
            reasons.append(f"length {rec['length']} outside [{MIN_LEN},{MAX_LEN}]")
        if rec.get("unknown_frac", 0) > MAX_UNKNOWN_FRAC:
            reasons.append(f"unknown_frac {rec['unknown_frac']:.2f}")
        if rec.get("missing_ca_frac", 0) > MAX_MISSING_CA_FRAC:
            reasons.append(f"missing_ca_frac {rec['missing_ca_frac']:.2f}")
        rec["valid"] = not reasons
        rec["invalid_reasons"] = reasons
        n_flagged += bool(reasons)
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(jsonl_path)
    print(f"  {len(records)-n_flagged:,}/{len(records):,} valid, {n_flagged:,} flagged invalid")

def _main_fetch() -> None:
    ap = argparse.ArgumentParser(description="Fetch protein chains + structure labels")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--source", choices=["cath", "pdb-list"], default="cath")
    ap.add_argument("--pdb-list", type=Path, default=None,
                    help="file of PDB or PDB_CHAIN lines (with --source pdb-list)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of chains")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    struct_dir = Path(args.structures_dir)
    qc.record_params(struct_dir, args, filename="params_01.json")   # provenance: these outputs are shared by the 01* stages
    jsonl_path = struct_dir / "index.jsonl"
    prot_dir = struct_dir / "proteins"
    raw_dir = struct_dir / "raw"

    if args.validate_only:
        if not jsonl_path.exists():
            sys.exit(f"ERROR: {jsonl_path} not found.")
        _validate_manifest(jsonl_path)
        return

    struct_dir.mkdir(parents=True, exist_ok=True)
    prot_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    # --- select chains ---
    print("Selecting chains ...")
    if args.source == "cath":
        cache = _download_cath_list(struct_dir / "cath-domain-list.txt")
        chains = _select_from_cath(cache, args.limit, args.seed)
    else:
        if not args.pdb_list:
            sys.exit("ERROR: --source pdb-list requires --pdb-list FILE")
        chains = _select_from_pdb_list(args.pdb_list)
        if args.limit:
            chains = chains[: args.limit]

    done = _load_done_ids(jsonl_path)
    if done:
        print(f"Resuming: {len(done):,} chains already done — skipping.")
    todo = [c for c in chains if c["id"] not in done]
    print(f"To fetch this run: {len(todo):,}")
    if not todo:
        if not args.no_validate:
            _validate_manifest(jsonl_path)
        return

    (struct_dir / "meta.json").write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source, "target_count": args.limit, "seed": args.seed,
        "n_selected": len(chains),
    }, indent=2))

    # --- fetch + featurise ---
    t0 = time.time()
    n_ok = 0
    errors: list[tuple[str, str]] = []
    with jsonl_path.open("a") as mf:
        for i, c in enumerate(todo, 1):
            try:
                feat = _featurise_chain(c["pdb"], c["chain"], raw_dir)
                if feat is None:
                    errors.append((c["id"], "no protein atoms"))
                    continue
                npz = prot_dir / f"{c['id']}.npz"
                np.savez_compressed(
                    npz, seq=feat["seq"], coords_bb=feat["coords_bb"],
                    ss3=feat["ss3"], rsa=feat["rsa"], res_ids=feat["res_ids"],
                )
                unknown_frac = feat["seq"].count("X") / max(1, feat["length"])
                missing_ca = float(np.isnan(feat["coords_bb"][:, 1, 0]).mean())
                rec = {**c, "npz_file": str(npz), "length": feat["length"],
                       "actual_chain": feat["actual_chain"],
                       "unknown_frac": round(unknown_frac, 4),
                       "missing_ca_frac": round(missing_ca, 4)}
                mf.write(json.dumps(rec) + "\n"); mf.flush()
                n_ok += 1
            except Exception as exc:
                errors.append((c["id"], str(exc)))
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.time() - t0 + 1e-9)
                print(f"  {i:,}/{len(todo):,}  ok={n_ok:,} err={len(errors):,}  "
                      f"({rate:.1f} chain/s)")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min — ok={n_ok:,}, errors={len(errors):,}")
    if errors:
        (struct_dir / "fetch_errors.json").write_text(json.dumps(errors, indent=2))
        print(f"  errors -> {struct_dir/'fetch_errors.json'}")
    if not args.no_validate:
        _validate_manifest(jsonl_path)

# ======================================================================================
# annotate  --  from 01b_fetch_annotations.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

SIFTS_URL = ("https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/"
             "pdb_chain_uniprot.tsv.gz")

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"

FIELDS_annotate = "accession,ec,cc_subcellular_location,lineage,xref_pfam,keyword,ft_mod_res,ft_carbohyd"

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
            {"query": q, "fields": FIELDS_annotate, "format": "tsv", "size": batch})
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

def _main_annotate() -> None:
    ap = argparse.ArgumentParser(description="Fetch UniProt function/localisation/family/PTM labels")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between UniProt calls")
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
    qc.record_params(sdir, args, filename="params_01b.json")   # provenance: these outputs are shared by the 01* stages
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

# ======================================================================================
# targets  --  from 01c_residue_targets.py
# ======================================================================================

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

FIELDS_targets = "accession,sequence,ft_binding,ft_act_site,ft_mod_res"

def _positions(text: str, keyword: str) -> set[int]:
    """1-based residue positions for a UniProt feature keyword (expands ranges)."""
    pos = set()
    for m in re.finditer(rf"{keyword} (\d+)(?:\.\.(\d+))?", text or ""):
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        pos.update(range(a, b + 1))
    return pos

def _fetch_sites(accs, batch, sleep):
    out = {}
    for i in range(0, len(accs), batch):
        chunk = accs[i:i + batch]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = f"{UNIPROT}?" + urllib.parse.urlencode(
            {"query": q, "fields": FIELDS_targets, "format": "tsv", "size": batch})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    lines = r.read().decode().splitlines()
                break
            except Exception as exc:
                print(f"    batch {i//batch} attempt {attempt+1} failed: {exc}")
                time.sleep(sleep * (attempt + 2)); lines = []
        if lines:
            hdr = lines[0].split("\t"); idx = {h: j for j, h in enumerate(hdr)}
            for line in lines[1:]:
                c = line.split("\t")
                if len(c) < len(hdr):
                    c += [""] * (len(hdr) - len(c))
                acc = c[idx["Entry"]]
                out[acc] = {
                    "seq": c[idx.get("Sequence", -1)] if "Sequence" in idx else "",
                    "binding": _positions(c[idx.get("Binding site", -1)] if "Binding site" in idx else "", "BINDING"),
                    "active": _positions(c[idx.get("Active site", -1)] if "Active site" in idx else "", "ACT_SITE"),
                    "mod": _positions(c[idx.get("Modified residue", -1)] if "Modified residue" in idx else "", "MOD_RES"),
                }
        print(f"  UniProt {min(i+batch, len(accs)):,}/{len(accs):,}")
        time.sleep(sleep)
    return out

def _uni_to_chain_map(chain_seq: str, uni_seq: str) -> dict:
    """Map UniProt 1-based position -> chain residue index (0-based) via local alignment."""
    import biotite.sequence as bseq
    import biotite.sequence.align as balign
    san = lambda s: "".join(ch if ch in "ACDEFGHIKLMNPQRSTVWY" else "X" for ch in s)
    try:
        s1 = bseq.ProteinSequence(san(chain_seq))
        s2 = bseq.ProteinSequence(san(uni_seq))
    except Exception:
        return {}
    mat = balign.SubstitutionMatrix.std_protein_matrix()
    aln = balign.align_optimal(s1, s2, mat, gap_penalty=(-10, -1), local=True)[0]
    m = {}
    for c_idx, u_idx in aln.trace:
        if c_idx >= 0 and u_idx >= 0:
            m[u_idx + 1] = int(c_idx)   # uniprot 1-based -> chain 0-based
    return m

def _bfactor(pdb: str, chain: str, res_ids: np.ndarray, raw_dir: Path) -> np.ndarray:
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    out = np.full(len(res_ids), np.nan, dtype=np.float32)
    path = raw_dir / f"{pdb}.bcif"
    if not path.exists():
        return out
    a = pdbx.get_structure(pdbx.BinaryCIFFile.read(str(path)), model=1, extra_fields=["b_factor"])
    a = a[struc.filter_amino_acids(a)]
    ca = a[(a.chain_id == chain) & (a.atom_name == "CA")]
    if ca.array_length() == 0:
        ca = a[a.atom_name == "CA"]
    bf = {int(r): float(b) for r, b in zip(ca.res_id, ca.b_factor)}
    for i, r in enumerate(res_ids):
        if int(r) in bf:
            out[i] = bf[int(r)]
    return out

def _main_targets() -> None:
    ap = argparse.ArgumentParser(description="Per-residue intrinsic proxy labels")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
    qc.record_params(sdir, args, filename="params_01c.json")   # provenance: these outputs are shared by the 01* stages
    prot_dir = sdir / "proteins"; raw_dir = sdir / "raw"
    out_dir = sdir / "residue_targets"; out_dir.mkdir(exist_ok=True)

    manifest = {json.loads(l)["id"]: json.loads(l)
                for l in (sdir / "index.jsonl").open() if l.strip()}
    acc = {}
    if (sdir / "annotations.jsonl").exists():
        for l in (sdir / "annotations.jsonl").open():
            if l.strip():
                a = json.loads(l); acc[a["id"]] = a["uniprot"]
    ids = [i for i in manifest if manifest[i].get("valid", True)]
    if args.limit:
        ids = ids[: args.limit]

    print(f"Fetching UniProt sites for {len(set(acc.values())):,} accessions ...")
    sites = _fetch_sites(sorted(set(acc[i] for i in ids if i in acc)), args.batch, args.sleep)

    n_ok = 0
    for cid in ids:
        r = manifest[cid]
        npz = np.load(prot_dir / f"{cid}.npz", allow_pickle=True)
        res_ids = npz["res_ids"]; chain_seq = str(npz["seq"]); L = len(res_ids)
        bfac = _bfactor(r["pdb"], r.get("actual_chain", r["chain"]), res_ids, raw_dir)
        binding = np.zeros(L, np.int8); active = np.zeros(L, np.int8); ptm = np.zeros(L, np.int8)
        s = sites.get(acc.get(cid, ""))
        if s and s["seq"]:
            m = _uni_to_chain_map(chain_seq, s["seq"])
            for p, ci in m.items():
                if p in s["binding"]: binding[ci] = 1
                if p in s["active"]:  active[ci] = 1
                if p in s["mod"]:     ptm[ci] = 1
        np.savez_compressed(out_dir / f"{cid}.npz", bfactor=bfac,
                            binding_site=binding, active_site=active, ptm_site=ptm)
        n_ok += 1
        if n_ok % 250 == 0:
            print(f"  wrote {n_ok:,}/{len(ids):,}")
    print(f"Done — wrote {n_ok:,} residue-target files -> {out_dir}")


# ==========================================================================================
# dispatch
# ==========================================================================================

_SUBCOMMANDS = {
    "fetch": _main_fetch,
    "annotate": _main_annotate,
    "targets": _main_targets,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in _SUBCOMMANDS:
        print(__doc__)
        print("subcommands: " + ", ".join(_SUBCOMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    cmd = sys.argv.pop(1)                 # the original main() then sees its original argv
    _SUBCOMMANDS[cmd]()


if __name__ == "__main__":
    main()
