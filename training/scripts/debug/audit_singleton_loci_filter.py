#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def variant_class_from_ref_alt(ref, alt):
    ref = "" if ref is None else str(ref)
    alt = "" if alt is None else str(alt)

    if not ref or not alt:
        return "MISSING"

    alts = [a for a in alt.split(",") if a]

    if not alts:
        return "MISSING"

    if any(a.startswith("<") or a == "*" for a in alts):
        return "SYMBOLIC_OR_STAR"

    if all(len(ref) == 1 and len(a) == 1 for a in alts):
        return "SNP"

    if any(len(ref) != len(a) for a in alts):
        return "INDEL"

    if all(len(ref) == len(a) and len(ref) > 1 for a in alts):
        return "MNV_OR_COMPLEX_SUB"

    return "COMPLEX"


def load_rows(dataset_name, path):
    rows = []

    files = sorted(Path(path).glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found: {path}")

    for p in files:
        d = np.load(p, allow_pickle=False)
        n = len(d["label"])

        locus = d["locus"].astype(str)
        label = d["label"]
        variant_type = d["variant_type"] if "variant_type" in d.files else np.full(n, -1)

        vcf_ref = d["vcf_ref"].astype(str) if "vcf_ref" in d.files else np.asarray([""] * n)
        vcf_alt = d["vcf_alt"].astype(str) if "vcf_alt" in d.files else np.asarray([""] * n)
        vcf_alt_full = d["vcf_alt_full"].astype(str) if "vcf_alt_full" in d.files else np.asarray([""] * n)
        variant_key = d["variant_key"].astype(str) if "variant_key" in d.files else np.asarray([""] * n)

        for i in range(n):
            ref = str(vcf_ref[i])
            alt = str(vcf_alt[i])
            alt_full = str(vcf_alt_full[i])

            rows.append(
                {
                    "dataset": dataset_name,
                    "locus": str(locus[i]),
                    "label": int(label[i]),
                    "truth_variant_binary": int(int(label[i]) > 0),
                    "variant_type": int(variant_type[i]),
                    "class_from_vcf": variant_class_from_ref_alt(ref, alt),
                    "multi_selected_alt": int("," in alt),
                    "multi_full_alt": int("," in alt_full),
                    "variant_key": str(variant_key[i]),
                }
            )

        d.close()

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--illumina", required=True)
    ap.add_argument("--ont", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Load] Illumina")
    ill = load_rows("illumina", args.illumina)

    print("[Load] ONT")
    ont = load_rows("ont", args.ont)

    # Count loci within each modality.
    ill_locus_counts = ill["locus"].value_counts()
    ont_locus_counts = ont["locus"].value_counts()

    ill["within_modality_locus_count"] = ill["locus"].map(ill_locus_counts).astype(int)
    ont["within_modality_locus_count"] = ont["locus"].map(ont_locus_counts).astype(int)

    ill["singleton_within_modality"] = (ill["within_modality_locus_count"] == 1).astype(int)
    ont["singleton_within_modality"] = (ont["within_modality_locus_count"] == 1).astype(int)

    # Reproduce outer singleton rule:
    # keep locus if illumina count <= 1 and ont count <= 1 and present in at least one.
    all_loci = set(ill_locus_counts.index) | set(ont_locus_counts.index)

    keep_loci = set()
    excluded_loci = set()

    for L in all_loci:
        ic = int(ill_locus_counts.get(L, 0))
        oc = int(ont_locus_counts.get(L, 0))

        if ic > 1 or oc > 1:
            excluded_loci.add(L)
            continue

        if ic == 1 or oc == 1:
            keep_loci.add(L)

    ill["kept_by_outer_singleton_rule"] = ill["locus"].isin(keep_loci).astype(int)
    ont["kept_by_outer_singleton_rule"] = ont["locus"].isin(keep_loci).astype(int)

    all_rows = pd.concat([ill, ont], ignore_index=True)

    all_rows.to_csv(out_dir / "singleton_loci_row_level.csv", index=False)

    summary = []

    for dataset, df in all_rows.groupby("dataset"):
        for kept_value, sub in df.groupby("kept_by_outer_singleton_rule"):
            summary.append(
                {
                    "dataset": dataset,
                    "kept_by_outer_singleton_rule": int(kept_value),
                    "rows": len(sub),
                    "truth_positive": int((sub["label"] > 0).sum()),
                    "truth_positive_snp_vcf": int(((sub["label"] > 0) & (sub["class_from_vcf"] == "SNP")).sum()),
                    "truth_positive_indel_vcf": int(((sub["label"] > 0) & (sub["class_from_vcf"] == "INDEL")).sum()),
                    "truth_positive_variant_type_1": int(((sub["label"] > 0) & (sub["variant_type"] == 1)).sum()),
                    "truth_positive_variant_type_2": int(((sub["label"] > 0) & (sub["variant_type"] == 2)).sum()),
                    "multi_selected_alt_rows": int(sub["multi_selected_alt"].sum()),
                    "multi_full_alt_rows": int(sub["multi_full_alt"].sum()),
                    "unique_loci": int(sub["locus"].nunique()),
                }
            )

    summary_df = pd.DataFrame(summary).sort_values(["dataset", "kept_by_outer_singleton_rule"])
    summary_df.to_csv(out_dir / "singleton_loci_summary.csv", index=False)

    by_class = (
        all_rows
        .groupby(
            [
                "dataset",
                "kept_by_outer_singleton_rule",
                "variant_type",
                "class_from_vcf",
                "label",
                "truth_variant_binary",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    by_class.to_csv(out_dir / "singleton_loci_counts_by_class.csv", index=False)

    locus_summary = pd.DataFrame(
        [
            {
                "all_loci": len(all_loci),
                "keep_loci": len(keep_loci),
                "excluded_loci": len(excluded_loci),
                "illumina_loci": len(ill_locus_counts),
                "ont_loci": len(ont_locus_counts),
                "illumina_rows": len(ill),
                "ont_rows": len(ont),
            }
        ]
    )
    locus_summary.to_csv(out_dir / "singleton_loci_global_summary.csv", index=False)

    print("\n=== Global ===")
    print(locus_summary.to_string(index=False))

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    print("\nWrote:")
    print(out_dir / "singleton_loci_global_summary.csv")
    print(out_dir / "singleton_loci_summary.csv")
    print(out_dir / "singleton_loci_counts_by_class.csv")
    print(out_dir / "singleton_loci_row_level.csv")


if __name__ == "__main__":
    main()
