#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot simple binary variant-vs-no-variant summaries from mixed OUTER report JSONs.

Supported inputs
----------------
1) eval_outer_ckpt_with_vt.py / eval_outer_teacher.py outputs
   - uses:
       overall_binary_variant_vs_no
       by_group[*].binary_variant_vs_no
       by_variant_type[*].binary_variant_vs_no

2) stage1 binary JSONs
   - best_metrics.json
   - calib_global.json
   - calib_per_group*.json
   - uses:
       test / val
       test.by_group[*]
       test.by_variant_type[*]

Usage pattern
-------------
Repeat --item as:
  --item "Label::path/to/report.json"

Slices
------
- overall
- groups:1,11
- groups:10,11
- groups:1,10,11
- vt:SNP
- vt:INDEL

Examples
--------
python training/scripts/hybrid_dv/analysis/plot_outer_variant_detection_summary.py \
  --item "DV ONT::training/out/reports/outer_chr20_v1/dv_ont_test.json" \
  --item "ONT Linear::training/out/reports/outer_chr20_v1/ont_only_linear_test.json" \
  --item "ONT MLP::training/out/reports/outer_chr20_v1/ont_only_mlp_test.json" \
  --item "Hybrid Linear::training/out/reports/outer_chr20_v1/hybrid_linear_ontvis_test.json" \
  --item "Hybrid MLP::training/out/reports/outer_chr20_v1/hybrid_mlp_ontvis_test.json" \
  --item "Stage1 Shared::training/out/models/stage1/hybrid_stage1_shared_mlp/best_metrics.json" \
  --item "Stage1 Groupwise global::training/out/models/stage1/hybrid_stage1_groupwise_mlp/calib_global.json" \
  --item "Stage1 Groupwise tuned::training/out/models/stage1/hybrid_stage1_groupwise_mlp/calib_per_group_g1_minrec090.json" \
  --slice groups:1,11 \
  --title "Variant vs no-variant — ONT-visible subset (groups 1+11)" \
  --out_png training/out/reports/outer_chr20_v1/variant_vs_no_ont_visible.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = [
    ("acc", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("specificity", "Specificity"),
]


def load_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def binary_metrics_from_cm(cm: List[List[int]]) -> Dict[str, float]:
    arr = np.array(cm, dtype=np.int64)
    if arr.shape != (2, 2):
        raise ValueError(f"Expected binary 2x2 cm, got shape={arr.shape}")

    tn, fp = int(arr[0, 0]), int(arr[0, 1])
    fn, tp = int(arr[1, 0]), int(arr[1, 1])

    eps = 1e-12
    acc = float((tp + tn) / max(tp + tn + fp + fn, 1))
    precision = float(tp / max(tp + fp, eps))
    recall = float(tp / max(tp + fn, eps))
    specificity = float(tn / max(tn + fp, eps))
    f1 = float(2 * precision * recall / max(precision + recall, eps))
    fpr = float(fp / max(fp + tn, eps))

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "cm": [[tn, fp], [fn, tp]],
        "n": int(tp + tn + fp + fn),
    }


def _is_eval3_json(j: Dict) -> bool:
    return "overall_binary_variant_vs_no" in j


def _is_stage1_json(j: Dict) -> bool:
    return (
        "test" in j
        and isinstance(j["test"], dict)
        and "precision" in j["test"]
        and "recall" in j["test"]
        and "cm" in j["test"]
    )


def _get_stage1_section(j: Dict, split: str) -> Dict:
    if split not in j:
        raise KeyError(f"Stage1-like JSON missing split='{split}'")
    return j[split]


def extract_overall_binary(j: Dict, split: str) -> Dict[str, float]:
    if _is_eval3_json(j):
        out = dict(j["overall_binary_variant_vs_no"])
        out["n"] = int(out.get("n", 0))
        return out

    if _is_stage1_json(j):
        sec = _get_stage1_section(j, split)
        out = {
            "acc": float(sec["acc"]),
            "precision": float(sec["precision"]),
            "recall": float(sec["recall"]),
            "f1": float(sec["f1"]),
            "specificity": float(sec["specificity"]),
            "cm": sec["cm"],
            "n": int(sec["n"]),
        }
        return out

    raise ValueError("Unknown JSON format for overall binary extraction")


def extract_group_binary_cm(j: Dict, split: str, group: str) -> List[List[int]]:
    if _is_eval3_json(j):
        bg = j.get("by_group", {})
        if group not in bg:
            raise KeyError(f"group={group} not found")
        return bg[group]["binary_variant_vs_no"]["cm"]

    if _is_stage1_json(j):
        sec = _get_stage1_section(j, split)
        bg = sec.get("by_group", {})
        if group not in bg:
            raise KeyError(f"group={group} not found")
        return bg[group]["cm"]

    raise ValueError("Unknown JSON format for by-group extraction")


def extract_vt_binary_cm(j: Dict, split: str, vt_name: str) -> List[List[int]]:
    if _is_eval3_json(j):
        bv = j.get("by_variant_type", {})
        if vt_name not in bv:
            raise KeyError(f"variant_type={vt_name} not found")
        return bv[vt_name]["binary_variant_vs_no"]["cm"]

    if _is_stage1_json(j):
        sec = _get_stage1_section(j, split)
        bv = sec.get("by_variant_type", {})
        if vt_name not in bv:
            raise KeyError(f"variant_type={vt_name} not found")
        return bv[vt_name]["cm"]

    raise ValueError("Unknown JSON format for by-variant-type extraction")


def sum_binary_cms(cms: List[List[List[int]]]) -> List[List[int]]:
    total = np.zeros((2, 2), dtype=np.int64)
    for cm in cms:
        total += np.array(cm, dtype=np.int64)
    return total.tolist()


def extract_slice_metrics(j: Dict, slice_spec: str, split: str) -> Dict[str, float]:
    if slice_spec == "overall":
        return extract_overall_binary(j, split)

    if slice_spec.startswith("groups:"):
        groups = [x.strip() for x in slice_spec.split(":", 1)[1].split(",") if x.strip()]
        cms = [extract_group_binary_cm(j, split, g) for g in groups]
        return binary_metrics_from_cm(sum_binary_cms(cms))

    if slice_spec.startswith("vt:"):
        vt = slice_spec.split(":", 1)[1].strip()
        cm = extract_vt_binary_cm(j, split, vt)
        return binary_metrics_from_cm(cm)

    raise ValueError(f"Unsupported slice_spec: {slice_spec}")


def parse_item(s: str) -> Tuple[str, Path]:
    if "::" not in s:
        raise ValueError(f"Invalid --item '{s}'. Expected format: Label::path/to/file.json")
    label, path = s.split("::", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise ValueError(f"Empty label in --item '{s}'")
    return label, path


def make_bar_plot(
    title: str,
    series: Dict[str, Dict[str, float]],
    metric_keys: List[Tuple[str, str]],
    out_png: Path,
):
    labels = [lab for _, lab in metric_keys]
    model_names = list(series.keys())
    x = np.arange(len(labels))
    width = 0.8 / max(len(model_names), 1)

    fig_w = max(10, 1.25 * len(model_names) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    all_vals = []

    for i, name in enumerate(model_names):
        vals = [float(series[name][k]) for k, _ in metric_keys]
        all_vals.extend(vals)
        xpos = x + (i - (len(model_names) - 1) / 2) * width
        ax.bar(xpos, vals, width=width, label=name)

    ymin = max(0.0, min(all_vals) - 0.02) if all_vals else 0.0
    ymax = min(1.001, max(all_vals) + 0.01) if all_vals else 1.0

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--item",
        action="append",
        required=True,
        help='Repeat as: --item "Label::path/to/report.json"',
    )
    ap.add_argument(
        "--slice",
        type=str,
        default="overall",
        help="overall | groups:1,11 | groups:10,11 | groups:1,10,11 | vt:SNP | vt:INDEL",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Used for stage1-like JSONs. Ignored for eval_outer_* JSONs.",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="acc,precision,recall,f1,specificity",
        help="Comma-separated metric keys from: acc,precision,recall,f1,specificity,fpr",
    )
    ap.add_argument("--title", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    metric_map = {
        "acc": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "specificity": "Specificity",
        "fpr": "FPR",
    }
    metric_keys = []
    for k in [x.strip() for x in args.metrics.split(",") if x.strip()]:
        if k not in metric_map:
            raise SystemExit(f"Unsupported metric key: {k}")
        metric_keys.append((k, metric_map[k]))

    series: Dict[str, Dict[str, float]] = {}
    summary: Dict[str, Dict] = {}

    for raw_item in args.item:
        label, path = parse_item(raw_item)
        if not path.exists():
            raise SystemExit(f"File not found for item '{label}': {path}")

        j = load_json(path)
        m = extract_slice_metrics(j, slice_spec=args.slice, split=args.split)
        series[label] = m
        summary[label] = {
            "path": str(path),
            "slice": args.slice,
            "split": args.split,
            "metrics": m,
        }

    make_bar_plot(
        title=args.title,
        series=series,
        metric_keys=metric_keys,
        out_png=Path(args.out_png),
    )

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[Saved] {args.out_png}")
    if args.out_json:
        print(f"[Saved] {args.out_json}")


if __name__ == "__main__":
    main()