"""N3, part 1: curate a substitute protein-protein training set from RCSB.

The paper's 464 pairs (IntAct/STRING interactions verified in PDBe-PISA) are only in the Code Ocean
capsule, which is not reachable from this machine. This builds a documented substitute:
X-ray heterodimers (2 protein entities, no nucleic acid, resolution <= 2.5 Å), one chain per entity,
chains of 50-600 residues, >= 10 inter-chain Cα pairs under 7 Å, one entry per UniProt pair, no UniProt
accession used more than twice, and nothing with >= 30% sequence identity to a held-out protein.

    python scripts/n3_curate.py --n 200 --seed 0
Writes data/curation/candidates.json and data/curation/selected.csv.
"""

import argparse
import csv
import json
import random
import time
import urllib.request
from pathlib import Path

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL = "https://data.rcsb.org/graphql"
HELDOUT = {"6WZO": ["A", "B"], "6NQN": ["A", "B"], "1C29": ["A", "B"]}
OUT = Path("data/curation")


def post(url, payload, tries=4):
    data = json.dumps(payload).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                return json.loads(body) if body else {}
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def attr(a, op, v):
    return {"type": "terminal", "service": "text", "parameters": {"attribute": a, "operator": op, "value": v}}


def heterodimer_ids():
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
            attr("rcsb_entry_info.polymer_entity_count_protein", "equals", 2),
            attr("rcsb_entry_info.polymer_entity_count_nucleic_acid", "equals", 0),
            attr("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
            attr("rcsb_entry_info.resolution_combined", "less_or_equal", 2.5),
            attr("rcsb_entry_info.deposited_polymer_entity_instance_count", "less_or_equal", 4)]},
         "return_type": "entry", "request_options": {"return_all_hits": True}}
    return [h["identifier"] for h in post(SEARCH, q).get("result_set", [])]


def similar_to(seq, identity=0.3):
    q = {"query": {"type": "terminal", "service": "sequence",
                   "parameters": {"evalue_cutoff": 1, "identity_cutoff": identity, "sequence_type": "protein", "value": seq}},
         "return_type": "entry", "request_options": {"return_all_hits": True}}
    return {h["identifier"] for h in post(SEARCH, q).get("result_set", [])}


ENTITY_Q = """query($ids:[String!]!){ entries(entry_ids:$ids){ rcsb_id
  polymer_entities{ rcsb_polymer_entity_container_identifiers{ auth_asym_ids }
    entity_poly{ pdbx_seq_one_letter_code_can rcsb_sample_sequence_length type }
    rcsb_polymer_entity_container_identifiers{ uniprot_ids } } } }"""


def entity_info(ids):
    out = {}
    for i in range(0, len(ids), 200):
        res = post(GRAPHQL, {"query": ENTITY_Q, "variables": {"ids": ids[i:i + 200]}})
        for e in (res.get("data") or {}).get("entries") or []:
            ents = []
            for pe in e.get("polymer_entities") or []:
                ci = pe["rcsb_polymer_entity_container_identifiers"]
                ep = pe["entity_poly"]
                ents.append({"chains": ci.get("auth_asym_ids") or [], "uniprot": ci.get("uniprot_ids") or [],
                             "length": ep.get("rcsb_sample_sequence_length"), "type": ep.get("type"),
                             "seq": ep.get("pdbx_seq_one_letter_code_can")})
            out[e["rcsb_id"]] = ents
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ids = heterodimer_ids()
    print(f"{len(ids)} X-ray heterodimer entries")
    held = entity_info(list(HELDOUT))
    exclude = set(HELDOUT)
    for pid, ents in held.items():
        for e in ents:
            if e["seq"]:
                exclude |= similar_to(e["seq"])
    print(f"excluding {len(exclude)} entries with >= 30% identity to a held-out protein")

    rng = random.Random(a.seed)
    pool = [i for i in ids if i not in exclude]
    rng.shuffle(pool)
    pool = pool[: a.n * 6]
    info = entity_info(pool)
    (OUT / "candidates.json").write_text(json.dumps({"n_search_hits": len(ids), "n_excluded_homologs": len(exclude),
                                                     "pool": pool}, indent=1))
    rows, pairs_seen, acc_count = [], set(), {}
    for pid in pool:
        ents = [e for e in info.get(pid, []) if e["type"] == "polypeptide(L)"]
        if len(ents) != 2 or not all(e["chains"] and e["length"] and 50 <= e["length"] <= 600 for e in ents):
            continue
        acc = tuple(sorted((e["uniprot"] or [f"{pid}-{k}"])[0] for k, e in enumerate(ents)))
        if acc in pairs_seen or any(acc_count.get(x, 0) >= 2 for x in acc):
            continue
        pairs_seen.add(acc)
        for x in acc:
            acc_count[x] = acc_count.get(x, 0) + 1
        rows.append([pid, ents[0]["chains"], ents[1]["chains"], acc[0], acc[1]])
    with (OUT / "candidate_pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "chains_entity1", "chains_entity2", "uniprot1", "uniprot2"])
        for r in rows:
            w.writerow([r[0], " ".join(r[1]), " ".join(r[2]), r[3], r[4]])
    print(f"{len(rows)} candidate pairs after entity filters (contact filter is applied in n3_build.py)")


if __name__ == "__main__":
    main()
