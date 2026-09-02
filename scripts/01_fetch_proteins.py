"""
01_fetch_proteins.py — build a non-redundant protein set with structure + labels.

For each selected protein chain this downloads the experimental structure from the
RCSB PDB (via biotite), then computes everything the downstream models and probes
need — entirely in Python, no external DSSP binary:

    - sequence                      (ESM-2 input)
    - backbone coordinates N,CA,C,O (ESM-IF1 / ProteinMPNN input)
    - per-residue 3-state SSE       (biotite annotate_sse; residue-local label)
    - per-residue relative SASA     (biotite sasa / Tien-2013 max ASA; burial label)
    - per-chain CATH C/A/T/H        (fold labels; whole-protein targets)

Sources (--source):
    cath      (default) download the CATH domain list, keep one representative per
              S35 sequence cluster (non-redundant), decode domain ids -> (pdb, chain),
              and label each chain with its CATH class/architecture/topology/homology.
    pdb-list  read a file of `PDB` or `PDB_CHAIN` lines (no CATH labels).

Outputs (under --structures-dir, default ./structures):
    meta.json                 run-level metadata
    index.jsonl               one JSON record per chain, appended as each is written
    proteins/<id>.npz         per-chain arrays (seq, coords_bb, ss3, rsa, res_ids)
    cath-domain-list.txt      cached CATH classification file (source=cath)

Resume: ids already in index.jsonl are skipped.
Validation: after fetching, flags chains outside length/quality bounds (--no-validate to skip).

Usage:
    uv run python scripts/01_fetch_proteins.py --structures-dir /ssc/structures --limit 5000
    uv run python scripts/01_fetch_proteins.py --structures-dir structures_test --limit 5
    uv run python scripts/01_fetch_proteins.py --pdb-list my_chains.txt --structures-dir /ssc/structures
"""

import argparse
import json
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np

warnings.filterwarnings("ignore")

CATH_DOMAIN_LIST_URL = (
    "https://download.cathdb.info/cath/releases/latest-release/"
    "cath-classification-data/cath-domain-list.txt"
)

# Tien et al. 2013 theoretical maximum accessible surface area (A^2), 3-letter codes.
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# Quality bounds applied in the validation pass.
MIN_LEN = 30
MAX_LEN = 1000
MAX_UNKNOWN_FRAC = 0.10   # fraction of non-standard residues tolerated
MAX_MISSING_CA_FRAC = 0.10


# ---------------------------------------------------------------------------
# Chain selection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-chain featurisation (biotite)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Manifest / validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
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


if __name__ == "__main__":
    main()
