"""
01c_residue_targets.py — per-residue intrinsic proxy labels on the common CATH set.

Adds per-residue targets to every chain so the "flexibility", "binding-site" and "PTM"
questions are answered on the SAME proteins as the structural/functional labels (no separate
benchmark sets):

    bfactor       Cα crystallographic B-factor (flexibility proxy)   — from the cached structure
    binding_site  1 if a UniProt BINDING-site residue                — UniProt ft_binding
    active_site   1 if a UniProt ACT_SITE residue                    — UniProt ft_act_site
    ptm_site      1 if a UniProt MOD_RES / modified residue          — UniProt ft_mod_res

UniProt features are in full-protein coordinates, so each chain's sequence (from its npz) is
locally aligned to the UniProt canonical sequence and feature positions are mapped onto the
chain's residues. UniProt is queried in rate-limited batches (reuses the accession from
annotations.jsonl).

Outputs (under --structures-dir):
    residue_targets/<id>.npz   {bfactor, binding_site, active_site, ptm_site} aligned to the
                               chain's residues (same order/length as proteins/<id>.npz)

Usage:
    uv run python scripts/01c_residue_targets.py --structures-dir /ssc/structures --sleep 3
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,sequence,ft_binding,ft_act_site,ft_mod_res"


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
            {"query": q, "fields": FIELDS, "format": "tsv", "size": batch})
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-residue intrinsic proxy labels")
    ap.add_argument("--structures-dir", default="structures")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sdir = Path(args.structures_dir)
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


if __name__ == "__main__":
    main()
