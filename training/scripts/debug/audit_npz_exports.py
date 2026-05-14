#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def variant_class_from_ref_alt(ref: str, alt: str) -> str:
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


def safe_get(arrs, key, n=None, default=None):
    if key in arrs:
        return arrs[key]
    if n is None:
        return None
    return np.asarray([default] * n)


def audit_dataset(name: str, path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found for {name}: {path}")

    summary = []
    by_class_label = Counter()
    by_variant_type_label = Counter()
    teacher_binary = defaultdict(lambda: Counter())

    total_rows = 0
    empty_variant_key = 0
    empty_ref_alt = 0
    multi_selected_alt = 0
    multi_full_alt = 0
    duplicate_key_rows = 0

    seen_keys = Counter()

    all_keys = set()

    for p in files:
        arrs = np.load(p, allow_pickle=False)
        all_keys.update(arrs.files)

        if "label" not in arrs:
            print(f"[WARN] {name}: missing label in {p}")
            continue

        labels = arrs["label"]
        n = len(labels)
        total_rows += n

        variant_type = safe_get(arrs, "variant_type", n=n, default=-1)
        vcf_ref = safe_get(arrs, "vcf_ref", n=n, default="")
        vcf_alt = safe_get(arrs, "vcf_alt", n=n, default="")
        vcf_alt_full = safe_get(arrs, "vcf_alt_full", n=n, default="")
        variant_key = safe_get(arrs, "variant_key", n=n, default="")

        teacher_probs = arrs["teacher_probs"] if "teacher_probs" in arrs else None

        for i in range(n):
            y = int(labels[i])
            vt_code = int(variant_type[i])

            ref = str(vcf_ref[i])
            alt = str(vcf_alt[i])
            alt_full = str(vcf_alt_full[i])
            key = str(variant_key[i])

            cls = variant_class_from_ref_alt(ref, alt)
            pos = int(y > 0)

            by_class_label[(cls, y, pos)] += 1
            by_variant_type_label[(vt_code, y, pos)] += 1

            if not key:
                empty_variant_key += 1
            else:
                seen_keys[key] += 1

            if not ref or not alt:
                empty_ref_alt += 1

            if "," in alt:
                multi_selected_alt += 1

            if "," in alt_full:
                multi_full_alt += 1

            if teacher_probs is not None and y >= 0:
                pred = int(np.argmax(teacher_probs[i]))
                pred_pos = int(pred > 0)
                truth_pos = int(y > 0)

                c = teacher_binary[cls]
                if pred_pos and truth_pos:
                    c["tp"] += 1
                elif pred_pos and not truth_pos:
                    c["fp"] += 1
                elif not pred_pos and truth_pos:
                    c["fn"] += 1
                else:
                    c["tn"] += 1

    duplicate_key_rows = sum(v for v in seen_keys.values() if v > 1)

    summary.append(
        {
            "dataset": name,
            "path": str(path),
            "npz_files": len(files),
            "rows": total_rows,
            "unique_variant_keys": len(seen_keys),
            "duplicate_key_rows": duplicate_key_rows,
            "empty_variant_key_rows": empty_variant_key,
            "empty_ref_alt_rows": empty_ref_alt,
            "multi_selected_alt_rows": multi_selected_alt,
            "multi_full_alt_rows": multi_full_alt,
            "available_keys": "|".join(sorted(all_keys)),
        }
    )

    class_rows = []
    for (cls, y, pos), count in sorted(by_class_label.items()):
        class_rows.append(
            {
                "dataset": name,
                "class_from_vcf": cls,
                "label": y,
                "truth_variant_binary": pos,
                "count": count,
            }
        )

    vt_rows = []
    for (vt_code, y, pos), count in sorted(by_variant_type_label.items()):
        vt_rows.append(
            {
                "dataset": name,
                "variant_type_code": vt_code,
                "label": y,
                "truth_variant_binary": pos,
                "count": count,
            }
        )

    teacher_rows = []
    for cls, c in sorted(teacher_binary.items()):
        tp = c["tp"]
        fp = c["fp"]
        fn = c["fn"]
        tn = c["tn"]

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) else np.nan

        teacher_rows.append(
            {
                "dataset": name,
                "class_from_vcf": cls,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "truth_positive": tp + fn,
                "pred_positive": tp + fp,
            }
        )

    return summary, class_rows + vt_rows, teacher_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec NAME=PATH. Can be repeated.",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_counts = []
    all_teacher = []

    for spec in args.dataset:
        if "=" not in spec:
            raise SystemExit(f"Bad --dataset spec: {spec}. Expected NAME=PATH")
        name, path_s = spec.split("=", 1)
        path = Path(path_s)

        print(f"\n=== Auditing {name}: {path} ===")
        summary, counts, teacher = audit_dataset(name, path)
        all_summary.extend(summary)
        all_counts.extend(counts)
        all_teacher.extend(teacher)

    summary_df = pd.DataFrame(all_summary)
    counts_df = pd.DataFrame(all_counts)
    teacher_df = pd.DataFrame(all_teacher)

    summary_df.to_csv(out_dir / "npz_audit_summary.csv", index=False)
    counts_df.to_csv(out_dir / "npz_audit_counts.csv", index=False)
    teacher_df.to_csv(out_dir / "npz_teacher_binary_metrics_by_vcf_class.csv", index=False)

    print("\n=== Summary ===")
    print(summary_df[["dataset", "npz_files", "rows", "unique_variant_keys", "empty_variant_key_rows", "multi_selected_alt_rows", "multi_full_alt_rows"]].to_string(index=False))

    if not teacher_df.empty:
        print("\n=== Teacher binary metrics by VCF-derived class ===")
        print(teacher_df.to_string(index=False))

    print("\nWrote:")
    print(out_dir / "npz_audit_summary.csv")
    print(out_dir / "npz_audit_counts.csv")
    print(out_dir / "npz_teacher_binary_metrics_by_vcf_class.csv")


if __name__ == "__main__":
    main()
