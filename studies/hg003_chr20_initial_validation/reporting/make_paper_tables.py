#!/usr/bin/env python3
"""
Tables 1 and 5 of the classical paper, from the HG003 chr20 exploratory study.

  Table 1: representation-level classification; DeepVariant internal reference versus
           linear and MLP probes trained on the DeepVariant representations.
  Table 5: direct three-class classification versus two-stage classification
           (variant versus non-variant, then genotype), hybrid heads.

All metrics are recomputed from the 3x3 confusion matrices of the evaluation reports
(rows = truth, columns = prediction; 0 hom-ref, 1 het, 2 hom-alt). For Table 5, variant
recall, false positives (FP) and false negatives (FN) refer to the binary variant /
non-variant decision: FN = true variants predicted hom-ref; FP = true hom-ref predicted
as a variant.

These are exploratory results from a single training run. The output files hold the rows
between \\midrule and \\bottomrule of each table.

    python studies/hg003_chr20_initial_validation/reporting/make_paper_tables.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
EVALUATION = STUDY / "evaluation"
TWO_STAGE = STUDY / "historical_outputs" / "two_stage"

TABLE1 = [  # (technology, classifier, report)
    ("Illumina", "DeepVariant internal", EVALUATION / "dv_illumina_test.json"),
    ("Illumina", "Linear Probe", EVALUATION / "unimodal_illumina_linear_test.json"),
    ("Illumina", "MLP Probe", EVALUATION / "unimodal_illumina_mlp_test.json"),
    ("ONT", "DeepVariant internal", EVALUATION / "dv_ont_test.json"),
    ("ONT", "Linear Probe", EVALUATION / "unimodal_ont_linear_test.json"),
    ("ONT", "MLP Probe", EVALUATION / "unimodal_ont_mlp_test.json"),
]

TABLE5 = [  # (head, calibration, report)
    ("Hybrid-S, direct (3 classes)", "--", EVALUATION / "hybrid_simple_mlp_test.json"),
    ("Hybrid-S, two-stage", "none", TWO_STAGE / "two_stage_shared_raw_test.json"),
    ("Hybrid-S, two-stage", "stage 1", TWO_STAGE / "two_stage_shared_calib_test.json"),
    ("Hybrid-S, two-stage", "3-class", TWO_STAGE / "two_stage_shared_3c_calib_test.json"),
    ("Hybrid-G, direct (3 classes)", "--", EVALUATION / "hybrid_groupwise_mlp_test.json"),
    ("Hybrid-G, two-stage", "none", TWO_STAGE / "two_stage_groupwise_raw_test.json"),
    ("Hybrid-G, two-stage", "stage 1", TWO_STAGE / "two_stage_groupwise_calib_test.json"),
    ("Hybrid-G, two-stage", "3-class", TWO_STAGE / "two_stage_groupwise_3c_calib_test.json"),
]


def find_cm(obj):
    """Return the first 3x3 confusion matrix over all groups found in a report."""
    if isinstance(obj, dict):
        for key in ("overall_3c", "metrics"):
            if key in obj and isinstance(obj[key], dict) and "cm" in obj[key]:
                return np.array(obj[key]["cm"])
        if "cm" in obj and np.array(obj["cm"]).shape == (3, 3):
            return np.array(obj["cm"])
        for value in obj.values():
            cm = find_cm(value)
            if cm is not None:
                return cm
    return None


def load_cm(path: Path) -> np.ndarray:
    cm = find_cm(json.loads(path.read_text(encoding="utf-8")))
    if cm is None:
        raise SystemExit(f"[ERROR] no 3x3 confusion matrix in {path}")
    return cm


def per_class(cm):
    tp = np.diag(cm).astype(float)
    precision = tp / cm.sum(0)
    recall = tp / cm.sum(1)
    f1 = 2 * precision * recall / (precision + recall)
    return tp, precision, recall, f1


def table1():
    lines = []
    for i, (tech, clf, path) in enumerate(TABLE1):
        cm = load_cm(path)
        tp, p, r, f1 = per_class(cm)
        lines.append(f"        {tech:<8} & {clf:<20} & {tp.sum() / cm.sum():.4f} & "
                     f"{p.mean():.4f} & {r.mean():.4f} & {f1.mean():.4f} \\\\")
        if i == 2:
            lines.append("        \\midrule")
    return lines


def table5():
    lines = []
    for i, (head, cal, path) in enumerate(TABLE5):
        cm = load_cm(path)
        _, _, _, f1 = per_class(cm)
        variants = cm[1:].sum()
        fn = int(cm[1:, 0].sum())
        fp = int(cm[0, 1:].sum())
        lines.append(f"        {head:<30} & {cal:<8} & {f1.mean():.5f} & "
                     f"{(variants - fn) / variants:.5f} & {fp} & {fn} \\\\")
        if i == 3:
            lines.append("        \\addlinespace")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(REPO / "paper" / "tables"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stem, fn in [("tab1_hg003_probes", table1), ("tab5_two_stage", table5)]:
        path = out / f"{stem}.tex"
        path.write_text("\n".join(fn()) + "\n", encoding="utf-8")
        print("Saved", path.relative_to(REPO) if path.is_relative_to(REPO) else path)


if __name__ == "__main__":
    main()
