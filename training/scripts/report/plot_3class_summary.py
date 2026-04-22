#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flexible 3-class OUTER summary plotting.

Supported inputs
----------------
Main target:
- JSONs from:
    * eval_outer_ckpt_with_vt.py
    * eval_outer_teacher.py

Those provide:
- overall_3c
- by_group[*]
- by_variant_type[*]

Also supports training metrics JSONs that expose:
- val / test with keys:
    acc, macro_precision, macro_recall, macro_f1, weighted_f1
    optionally by_group

Typical use
-----------
Repeat:
  --item "Label::path/to/report.json"

Slices
------
- overall
- groups:1,11
- groups:10,11
- groups:1,10,11
- vt:SNP
- vt:INDEL

Metrics
-------
Default:
- acc
- macro_precision
- macro_recall
- macro_f1
- weighted_f1

Examples
--------
python training/scripts/hybrid_dv/analysis/plot_outer_3class_summary.py \
  --item "Hybrid Linear::training/out/reports/outer_chr20_v1/hybrid_linear_test.json" \
  --item "Hybrid MLP::training/out/reports/outer_chr20_v1/hybrid_mlp_test.json" \
  --item "Hybrid Groupwise Linear::training/out/reports/outer_chr20_v1/hybrid_groupwise_linear_test.json" \
  --item "Hybrid Groupwise MLP::training/out/reports/outer_chr20_v1/hybrid_groupwise_mlp_test.json" \
  --slice groups:1,10,11 \
  --title "Hybrid 3-class ablation — shared vs groupwise" \
  --out_png training/out/reports/outer_chr20_v1/hybrid_3class_ablation.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = [
    ("acc", "Accuracy"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
]


def load_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def metrics_from_cm_3(cm: List[List[int]]) -> Dict[str, float]:
    arr = np.array(cm, dtype=np.int64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 confusion matrix, got shape={arr.shape}")

    eps = 1e-12
    total = int(arr.sum())
    acc = float(np.trace(arr) / max(total, 1))

    f1s = []
    recs = []
    precs = []
    supports = []

    for c in range(3):
        tp = float(arr[c, c])
        fp = float(arr[:, c].sum() - arr[c, c])
        fn = float(arr[c, :].sum() - arr[c, c])
        support = float(arr[c, :].sum())

        prec = float(tp / (tp + fp + eps))
        rec = float(tp / (tp + fn + eps))
        f1 = float(2 * prec * rec / (prec + rec + eps))

        f1s.append(f1)
        recs.append(rec)
        precs.append(prec)
        supports.append(support)

    macro_f1 = float(np.mean(f1s))
    macro_recall = float(np.mean(recs))
    macro_precision = float(np.mean(precs))
    wsum = float(np.sum(supports)) + eps
    weighted_f1 = float(np.sum([f1s[i] * supports[i] for i in range(3)]) / wsum)

    return {
        "acc": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cm": arr.tolist(),
        "n": total,
    }


def sum_cm3(cms: List[List[List[int]]]) -> List[List[int]]:
    total = np.zeros((3, 3), dtype=np.int64)
    for cm in cms:
        total += np.array(cm, dtype=np.int64)
    return total.tolist()


def _is_eval_json(j: Dict) -> bool:
    return "overall_3c" in j


def _is_train_metrics_json(j: Dict) -> bool:
    return (
        "test" in j
        and isinstance(j["test"], dict)
        and "acc" in j["test"]
        and "macro_f1" in j["test"]
        and "weighted_f1" in j["test"]
    )


def _get_train_section(j: Dict, split: str) -> Dict:
    if split not in j:
        raise KeyError(f"Training-metrics JSON missing split='{split}'")
    return j[split]


def extract_overall_3c(j: Dict, split: str) -> Dict[str, float]:
    if _is_eval_json(j):
        out = dict(j["overall_3c"])
        out["n"] = int(out.get("n", 0))
        return out

    if _is_train_metrics_json(j):
        sec = _get_train_section(j, split)
        return {
            "acc": float(sec["acc"]),
            "macro_precision": float(sec["macro_precision"]),
            "macro_recall": float(sec["macro_recall"]),
            "macro_f1": float(sec["macro_f1"]),
            "weighted_f1": float(sec["weighted_f1"]),
            "n": int(sec["n"]),
            "cm": sec["cm"],
        }

    raise ValueError("Unknown JSON format for overall_3c extraction")


def extract_group_cm3(j: Dict, split: str, group: str) -> List[List[int]]:
    if _is_eval_json(j):
        bg = j.get("by_group", {})
        if group not in bg:
            raise KeyError(f"group={group} not found")
        return bg[group]["cm"]

    if _is_train_metrics_json(j):
        sec = _get_train_section(j, split)
        bg = sec.get("by_group", {})
        if group not in bg:
            raise KeyError(f"group={group} not found")
        return bg[group]["cm"]

    raise ValueError("Unknown JSON format for by-group extraction")


def extract_vt_cm3(j: Dict, vt_name: str) -> List[List[int]]:
    if _is_eval_json(j):
        bv = j.get("by_variant_type", {})
        if vt_name not in bv:
            raise KeyError(f"variant_type={vt_name} not found")
        return bv[vt_name]["cm"]

    raise ValueError("This JSON format does not support by_variant_type slices")


def extract_slice_metrics(j: Dict, slice_spec: str, split: str) -> Dict[str, float]:
    if slice_spec == "overall":
        return extract_overall_3c(j, split)

    if slice_spec.startswith("groups:"):
        groups = [x.strip() for x in slice_spec.split(":", 1)[1].split(",") if x.strip()]
        cms = [extract_group_cm3(j, split, g) for g in groups]
        return metrics_from_cm_3(sum_cm3(cms))

    if slice_spec.startswith("vt:"):
        vt = slice_spec.split(":", 1)[1].strip()
        return metrics_from_cm_3(extract_vt_cm3(j, vt))

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
    y_pad_frac: float = 0.15,
    y_pad_abs: float = 0.0015,
    min_y_window: float = 0.015,
    ymin_override: float = None,
    ymax_override: float = None,
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

    if not all_vals:
        ymin, ymax = 0.0, 1.0
    else:
        vmin = min(all_vals)
        vmax = max(all_vals)
        span = vmax - vmin

        effective_span = max(span, min_y_window)
        pad = max(y_pad_abs, effective_span * y_pad_frac)

        ymin = max(0.0, vmin - pad)
        ymax = min(1.0, vmax + pad)

        # Garantiza una ventana mínima aunque todos los valores estén pegadísimos
        if (ymax - ymin) < min_y_window:
            center = 0.5 * (ymin + ymax)
            half = 0.5 * min_y_window
            ymin = max(0.0, center - half)
            ymax = min(1.0, center + half)

    if ymin_override is not None:
        ymin = float(ymin_override)
    if ymax_override is not None:
        ymax = float(ymax_override)

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
        help="Used only for train-metrics style JSONs.",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="acc,macro_precision,macro_recall,macro_f1,weighted_f1",
        help="Comma-separated metric keys from: acc,macro_precision,macro_recall,macro_f1,weighted_f1",
    )
    ap.add_argument("--title", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_json", type=str, default=None)

    ap.add_argument("--y_pad_frac", type=float, default=0.15)
    ap.add_argument("--y_pad_abs", type=float, default=0.0015)
    ap.add_argument("--min_y_window", type=float, default=0.015)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    args = ap.parse_args()

    metric_map = {
        "acc": "Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
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
        y_pad_frac=args.y_pad_frac,
        y_pad_abs=args.y_pad_abs,
        min_y_window=args.min_y_window,
        ymin_override=args.ymin,
        ymax_override=args.ymax,
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