#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze ambiguous group11 examples in OUTER meta.

Goal
----
Characterize how often group11 examples present:
- vt_mismatch    : variant_type_ill != variant_type_ont
- label_mismatch : label_ill != label_ont

This helps decide whether vt_mismatch examples should be:
- kept,
- excluded,
- or only flagged for later post-processing.

Expected input CSV
------------------
A meta CSV with at least these columns:
- locus
- group
- label
- label_ill
- label_ont
- variant_type
- variant_type_ill
- variant_type_ont
- vt_mismatch

Typical usage
-------------
python training/scripts/hybrid_dv/analysis/analyze_group11_mismatches.py \
  --meta_csv training/out/meta/hg003_chr20_outer.meta_with_vt.csv \
  --out_json training/out/reports/outer_chr20_v1/group11_mismatch_report.json \
  --out_csv training/out/reports/outer_chr20_v1/group11_mismatch_report.csv
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any


LABEL_NAME = {
    -1: "missing",
    0: "hom-ref",
    1: "het",
    2: "hom-alt",
}

VT_NAME = {
    0: "UNK",
    1: "SNP",
    2: "INDEL",
}


def as_int(x, default=-1):
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def combo_key(label_ill: int, label_ont: int) -> str:
    return f"{label_ill}->{label_ont}"


def vt_combo_key(vt_ill: int, vt_ont: int) -> str:
    return f"{VT_NAME.get(vt_ill, str(vt_ill))}->{VT_NAME.get(vt_ont, str(vt_ont))}"


def pct(n: int, d: int) -> float:
    return 100.0 * float(n) / max(float(d), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    meta_csv = Path(args.meta_csv)
    if not meta_csv.exists():
        raise SystemExit(f"Missing meta_csv: {meta_csv}")

    n_total = 0
    n_group11 = 0

    # Core counters
    label_pair_counts = Counter()
    vt_pair_counts = Counter()

    n_label_mismatch = 0
    n_vt_mismatch = 0
    n_both_mismatch = 0
    n_neither_mismatch = 0

    # Cross tabs
    vtmm_by_labelpair = Counter()
    labelmm_by_vtpair = Counter()

    # More detailed breakdowns
    buckets = {
        "group11_all": [],
        "vt_mismatch_only": [],
        "label_mismatch_only": [],
        "both_mismatch": [],
        "neither_mismatch": [],
    }

    # Summary by final joined label
    final_join_label_counts = Counter()
    final_join_label_counts_vtmm = Counter()
    final_join_label_counts_labelmm = Counter()

    # By final variant type (from meta variant_type)
    final_vt_counts = Counter()
    final_vt_counts_vtmm = Counter()
    final_vt_counts_labelmm = Counter()

    with open(meta_csv, "r", newline="") as f:
        r = csv.DictReader(f)

        required = {
            "locus", "group", "label",
            "label_ill", "label_ont",
            "variant_type", "variant_type_ill", "variant_type_ont",
            "vt_mismatch",
        }
        missing = required - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in meta CSV: {sorted(missing)}")

        for row in r:
            n_total += 1

            group = as_int(row.get("group"), default=-1)
            if group != 11:
                continue

            n_group11 += 1

            locus = row["locus"]
            label_join = as_int(row.get("label"), default=-1)
            label_ill = as_int(row.get("label_ill"), default=-1)
            label_ont = as_int(row.get("label_ont"), default=-1)

            vt_final = as_int(row.get("variant_type"), default=0)
            vt_ill = as_int(row.get("variant_type_ill"), default=0)
            vt_ont = as_int(row.get("variant_type_ont"), default=0)

            vt_mismatch = as_int(row.get("vt_mismatch"), default=0)
            label_mismatch = int(label_ill >= 0 and label_ont >= 0 and label_ill != label_ont)

            lp = combo_key(label_ill, label_ont)
            vp = vt_combo_key(vt_ill, vt_ont)

            label_pair_counts[lp] += 1
            vt_pair_counts[vp] += 1

            final_join_label_counts[label_join] += 1
            final_vt_counts[vt_final] += 1

            if vt_mismatch:
                n_vt_mismatch += 1
                vtmm_by_labelpair[lp] += 1
                final_join_label_counts_vtmm[label_join] += 1
                final_vt_counts_vtmm[vt_final] += 1

            if label_mismatch:
                n_label_mismatch += 1
                labelmm_by_vtpair[vp] += 1
                final_join_label_counts_labelmm[label_join] += 1
                final_vt_counts_labelmm[vt_final] += 1

            if vt_mismatch and label_mismatch:
                n_both_mismatch += 1
                bucket_name = "both_mismatch"
            elif vt_mismatch and not label_mismatch:
                bucket_name = "vt_mismatch_only"
            elif label_mismatch and not vt_mismatch:
                bucket_name = "label_mismatch_only"
            else:
                n_neither_mismatch += 1
                bucket_name = "neither_mismatch"

            rec = {
                "locus": locus,
                "label_join": label_join,
                "label_ill": label_ill,
                "label_ont": label_ont,
                "variant_type": vt_final,
                "variant_type_ill": vt_ill,
                "variant_type_ont": vt_ont,
                "vt_mismatch": vt_mismatch,
                "label_mismatch": label_mismatch,
            }
            buckets["group11_all"].append(rec)
            buckets[bucket_name].append(rec)

    summary: Dict[str, Any] = {
        "n_total_meta": n_total,
        "n_group11": n_group11,

        "core_counts": {
            "vt_mismatch": n_vt_mismatch,
            "label_mismatch": n_label_mismatch,
            "both_mismatch": n_both_mismatch,
            "neither_mismatch": n_neither_mismatch,
        },

        "core_rates_pct_over_group11": {
            "vt_mismatch_pct": pct(n_vt_mismatch, n_group11),
            "label_mismatch_pct": pct(n_label_mismatch, n_group11),
            "both_mismatch_pct": pct(n_both_mismatch, n_group11),
            "neither_mismatch_pct": pct(n_neither_mismatch, n_group11),
        },

        "label_pair_counts": dict(label_pair_counts),
        "vt_pair_counts": dict(vt_pair_counts),

        "vt_mismatch_by_label_pair": dict(vtmm_by_labelpair),
        "label_mismatch_by_vt_pair": dict(labelmm_by_vtpair),

        "final_join_label_counts_group11": {
            str(k): int(v) for k, v in sorted(final_join_label_counts.items())
        },
        "final_join_label_counts_among_vt_mismatch": {
            str(k): int(v) for k, v in sorted(final_join_label_counts_vtmm.items())
        },
        "final_join_label_counts_among_label_mismatch": {
            str(k): int(v) for k, v in sorted(final_join_label_counts_labelmm.items())
        },

        "final_variant_type_counts_group11": {
            VT_NAME.get(k, str(k)): int(v) for k, v in sorted(final_vt_counts.items())
        },
        "final_variant_type_counts_among_vt_mismatch": {
            VT_NAME.get(k, str(k)): int(v) for k, v in sorted(final_vt_counts_vtmm.items())
        },
        "final_variant_type_counts_among_label_mismatch": {
            VT_NAME.get(k, str(k)): int(v) for k, v in sorted(final_vt_counts_labelmm.items())
        },

        "interpretation_hints": [
            "If vt_mismatch is common but label_mismatch is rare, then vt_mismatch may be mostly harmless for the current 3-class task.",
            "If label_mismatch is concentrated inside vt_mismatch, then vt_mismatch is a good proxy for ambiguous loci.",
            "If many vt_mismatch cases still have label_ill == label_ont, they may be valid to keep for 0/1/2 classification and only problematic later for VCF reconstruction.",
            "If both_mismatch is tiny, a simple heuristic for hybrid benchmarking is to drop only label_mismatch in group11.",
        ],
    }

    # Console summary
    print("\n" + "=" * 80)
    print("GROUP11 MISMATCH REPORT")
    print("=" * 80)
    print(f"Total rows in meta: {n_total:,}")
    print(f"group11 rows:       {n_group11:,}")
    print()
    print("Core counts:")
    print(f"  vt_mismatch:      {n_vt_mismatch:,} ({pct(n_vt_mismatch, n_group11):.3f}%)")
    print(f"  label_mismatch:   {n_label_mismatch:,} ({pct(n_label_mismatch, n_group11):.3f}%)")
    print(f"  both_mismatch:    {n_both_mismatch:,} ({pct(n_both_mismatch, n_group11):.3f}%)")
    print(f"  neither_mismatch: {n_neither_mismatch:,} ({pct(n_neither_mismatch, n_group11):.3f}%)")
    print()

    print("Top label pairs (label_ill -> label_ont):")
    for k, v in label_pair_counts.most_common(10):
        print(f"  {k:>6s}: {v:,}")

    print("\nTop vt pairs (variant_type_ill -> variant_type_ont):")
    for k, v in vt_pair_counts.most_common(10):
        print(f"  {k:>12s}: {v:,}")

    print("\nFinal joined label distribution in group11:")
    for k, v in sorted(final_join_label_counts.items()):
        print(f"  {k} ({LABEL_NAME.get(k, str(k))}): {v:,}")

    print("\nFinal joined label distribution among vt_mismatch:")
    for k, v in sorted(final_join_label_counts_vtmm.items()):
        print(f"  {k} ({LABEL_NAME.get(k, str(k))}): {v:,}")

    print("\nFinal joined label distribution among label_mismatch:")
    for k, v in sorted(final_join_label_counts_labelmm.items()):
        print(f"  {k} ({LABEL_NAME.get(k, str(k))}): {v:,}")

    print("=" * 80 + "\n")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[Saved] JSON summary: {out_json}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for rec in buckets["group11_all"]:
            rows.append({
                "locus": rec["locus"],
                "label_join": rec["label_join"],
                "label_ill": rec["label_ill"],
                "label_ont": rec["label_ont"],
                "label_mismatch": rec["label_mismatch"],
                "variant_type": rec["variant_type"],
                "variant_type_ill": rec["variant_type_ill"],
                "variant_type_ont": rec["variant_type_ont"],
                "vt_mismatch": rec["vt_mismatch"],
            })

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
                "locus", "label_join", "label_ill", "label_ont", "label_mismatch",
                "variant_type", "variant_type_ill", "variant_type_ont", "vt_mismatch"
            ])
            w.writeheader()
            w.writerows(rows)

        print(f"[Saved] Detailed CSV: {out_csv}")


if __name__ == "__main__":
    main()