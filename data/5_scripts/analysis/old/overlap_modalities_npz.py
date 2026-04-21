#!/usr/bin/env python3
import argparse, glob
from collections import Counter, defaultdict
import numpy as np

def iter_npz(glob_pat):
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"No NPZ matched: {glob_pat}")
    for p in paths:
        d = np.load(p, allow_pickle=True)
        yield p, d

def collect_counts(glob_pat):
    n = 0
    label_ctr = Counter()
    locus_ctr = Counter()
    example_id_set = set()
    variant_hash_set = set()

    for _, d in iter_npz(glob_pat):
        locus = d["locus"].astype(str)
        labels = d["label"].astype(np.int64)

        n += len(locus)
        label_ctr.update(labels.tolist())
        locus_ctr.update(locus.tolist())

        if "example_id" in d.files:
            example_id_set.update(d["example_id"].astype(str).tolist())
        if "variant_hash" in d.files:
            variant_hash_set.update(d["variant_hash"].astype(str).tolist())

    return {
        "N": n,
        "label_ctr": label_ctr,
        "locus_ctr": locus_ctr,
        "example_id_set": example_id_set,
        "variant_hash_set": variant_hash_set,
    }

def build_one_to_one_pairs(glob_ill, glob_ont, out_csv):
    # First pass: count candidates per locus
    ill = collect_counts(glob_ill)
    ont = collect_counts(glob_ont)

    common_loci = set(ill["locus_ctr"]) & set(ont["locus_ctr"])

    # loci where exactly one example in each modality
    loci_11 = [L for L in common_loci if ill["locus_ctr"][L] == 1 and ont["locus_ctr"][L] == 1]

    # Second pass: map locus -> (npz_path, local_index, label, example_id)
    def map_singletons(glob_pat, keep_loci):
        m = {}
        for path, d in iter_npz(glob_pat):
            locus = d["locus"].astype(str)
            labels = d["label"].astype(np.int64)
            exid = d["example_id"].astype(str) if "example_id" in d.files else None
            for i, L in enumerate(locus):
                if L in keep_loci:
                    m[L] = (path, i, int(labels[i]), (exid[i] if exid is not None else ""))
        return m

    ill_m = map_singletons(glob_ill, set(loci_11))
    ont_m = map_singletons(glob_ont, set(loci_11))

    # Write pairs
    rows = []
    agree = 0
    for L in loci_11:
        pi, ii, yi, ei = ill_m[L]
        po, io, yo, eo = ont_m[L]
        if yi == yo:
            agree += 1
        rows.append((L, yi, yo, pi, ii, po, io, ei, eo))

    header = "locus,label_illumina,label_ont,ill_npz,ill_idx,ont_npz,ont_idx,ill_example_id,ont_example_id"
    with open(out_csv, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    return {
        "common_loci": len(common_loci),
        "loci_11": len(loci_11),
        "label_agreement_11": agree / max(1, len(loci_11)),
        "out_csv": out_csv,
        "ill": ill,
        "ont": ont,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ill_glob", required=True)
    ap.add_argument("--ont_glob", required=True)
    ap.add_argument("--out_csv", default="pairs_one_to_one.csv")
    args = ap.parse_args()

    stats = build_one_to_one_pairs(args.ill_glob, args.ont_glob, args.out_csv)

    ill = stats["ill"]; ont = stats["ont"]
    print("\n=== Illumina ===")
    print("N examples:", ill["N"])
    print("Unique loci:", len(ill["locus_ctr"]))
    print("Label dist:", dict(sorted(ill["label_ctr"].items())))
    print("Unique example_id:", len(ill["example_id_set"]) if ill["example_id_set"] else 0)
    print("Unique variant_hash:", len(ill["variant_hash_set"]) if ill["variant_hash_set"] else 0)

    print("\n=== ONT ===")
    print("N examples:", ont["N"])
    print("Unique loci:", len(ont["locus_ctr"]))
    print("Label dist:", dict(sorted(ont["label_ctr"].items())))
    print("Unique example_id:", len(ont["example_id_set"]) if ont["example_id_set"] else 0)
    print("Unique variant_hash:", len(ont["variant_hash_set"]) if ont["variant_hash_set"] else 0)

    # overlaps for sanity
    if ill["example_id_set"] and ont["example_id_set"]:
        print("\nOverlap example_id:", len(ill["example_id_set"] & ont["example_id_set"]))
    if ill["variant_hash_set"] and ont["variant_hash_set"]:
        print("Overlap variant_hash:", len(ill["variant_hash_set"] & ont["variant_hash_set"]))

    print("\n=== Locus overlap ===")
    print("Common loci:", stats["common_loci"])
    print("1↔1 loci:", stats["loci_11"])
    print("Label agreement on 1↔1:", f"{100*stats['label_agreement_11']:.2f}%")
    print("Pairs CSV:", stats["out_csv"])

if __name__ == "__main__":
    main()