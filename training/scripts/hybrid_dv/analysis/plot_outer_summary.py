#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create simple matplotlib bar plots for OUTER reporting.

Expected inputs:
- JSONs from eval_outer_ckpt_with_vt.py (models)
- JSONs from eval_outer_teacher.py (DeepVariant teachers)

We plot:
1) Architecture ablations:
   - Illumina-only: linear vs MLP
   - ONT-only: linear vs MLP
   - Hybrid: linear vs MLP

2) DeepVariant Illumina vs Illumina baseline vs Hybrid
   (all evaluated on groups 10+11)

3) DeepVariant ONT vs ONT baseline vs Hybrid
   (all evaluated on groups 1+11)
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


METRIC_KEYS = [
    ("acc", "Accuracy"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("macro_f1", "Macro F1"),
]


def load_eval_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def metrics_from_eval(j: Dict) -> Dict[str, float]:
    o = j["overall_3c"]
    return {
        "acc": float(o["acc"]),
        "macro_precision": float(o["macro_precision"]),
        "macro_recall": float(o["macro_recall"]),
        "macro_f1": float(o["macro_f1"]),
    }


def make_bar_plot(title: str, series: Dict[str, Dict[str, float]], out_png: Path):
    labels = [lab for _, lab in METRIC_KEYS]
    model_names = list(series.keys())
    x = list(range(len(labels)))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    all_vals = []

    for i, name in enumerate(model_names):
        vals = [series[name][k] for k, _ in METRIC_KEYS]
        all_vals.extend(vals)
        xpos = [xx + (i - (len(model_names) - 1) / 2) * width for xx in x]
        ax.bar(xpos, vals, width=width, label=name)

    ymin = max(0.0, min(all_vals) - 0.01)
    ymax = min(1.001, max(all_vals) + 0.002)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    # Architecture ablations
    ap.add_argument("--ill_linear", type=str, required=True)
    ap.add_argument("--ill_mlp", type=str, required=True)
    ap.add_argument("--ont_linear", type=str, required=True)
    ap.add_argument("--ont_mlp", type=str, required=True)
    ap.add_argument("--hyb_linear_all", type=str, required=True)
    ap.add_argument("--hyb_mlp_all", type=str, required=True)

    # DV comparisons
    ap.add_argument("--dv_ill", type=str, required=True)
    ap.add_argument("--dv_ont", type=str, required=True)
    ap.add_argument("--hyb_linear_illvis", type=str, required=True)
    ap.add_argument("--hyb_mlp_illvis", type=str, required=True)
    ap.add_argument("--hyb_linear_ontvis", type=str, required=True)
    ap.add_argument("--hyb_mlp_ontvis", type=str, required=True)
    ap.add_argument("--dv_ill_group11", type=str, required=True)
    ap.add_argument("--dv_ont_group11", type=str, required=True)
    ap.add_argument("--hyb_mlp_group11", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load everything
    ill_linear = metrics_from_eval(load_eval_json(Path(args.ill_linear)))
    ill_mlp = metrics_from_eval(load_eval_json(Path(args.ill_mlp)))
    ont_linear = metrics_from_eval(load_eval_json(Path(args.ont_linear)))
    ont_mlp = metrics_from_eval(load_eval_json(Path(args.ont_mlp)))
    hyb_linear_all = metrics_from_eval(load_eval_json(Path(args.hyb_linear_all)))
    hyb_mlp_all = metrics_from_eval(load_eval_json(Path(args.hyb_mlp_all)))

    dv_ill = metrics_from_eval(load_eval_json(Path(args.dv_ill)))
    dv_ont = metrics_from_eval(load_eval_json(Path(args.dv_ont)))

    hyb_linear_illvis = metrics_from_eval(load_eval_json(Path(args.hyb_linear_illvis)))
    hyb_mlp_illvis = metrics_from_eval(load_eval_json(Path(args.hyb_mlp_illvis)))
    hyb_linear_ontvis = metrics_from_eval(load_eval_json(Path(args.hyb_linear_ontvis)))
    hyb_mlp_ontvis = metrics_from_eval(load_eval_json(Path(args.hyb_mlp_ontvis)))

    dv_ill_group11 = metrics_from_eval(load_eval_json(Path(args.dv_ill_group11)))
    dv_ont_group11 = metrics_from_eval(load_eval_json(Path(args.dv_ont_group11)))
    hyb_mlp_group11 = metrics_from_eval(load_eval_json(Path(args.hyb_mlp_group11)))

    # 1) Architecture plots
    make_bar_plot(
        "Illumina-only: linear vs MLP",
        {
            "Linear": ill_linear,
            "MLP": ill_mlp,
        },
        out_dir / "ill_only_linear_vs_mlp.png",
    )

    make_bar_plot(
        "ONT-only: linear vs MLP",
        {
            "Linear": ont_linear,
            "MLP": ont_mlp,
        },
        out_dir / "ont_only_linear_vs_mlp.png",
    )

    make_bar_plot(
        "Hybrid simple: linear vs MLP",
        {
            "Linear": hyb_linear_all,
            "MLP": hyb_mlp_all,
        },
        out_dir / "hybrid_linear_vs_mlp.png",
    )

    # 2) DV Illumina-visible comparison
    make_bar_plot(
        "Illumina-visible subset (groups 10+11)",
        {
            "DV Illumina": dv_ill,
            "Illumina-only Linear": ill_linear,
            "Hybrid Linear": hyb_linear_illvis,
            "Hybrid MLP": hyb_mlp_illvis,
        },
        out_dir / "dv_illumina_vs_models.png",
    )

    # 3) DV ONT-visible comparison
    make_bar_plot(
        "ONT-visible subset (groups 1+11)",
        {
            "DV ONT": dv_ont,
            "ONT-only Linear": ont_linear,
            "Hybrid Linear": hyb_linear_ontvis,
            "Hybrid MLP": hyb_mlp_ontvis,
        },
        out_dir / "dv_ont_vs_models.png",
    )

    # 4) DV ONT and Illumina vs Hybrid in Group11
    make_bar_plot(
        "Common multimodal subset (group11 only)",
        {
            "DV Illumina": dv_ill_group11,
            "DV ONT": dv_ont_group11,
            "Hybrid MLP": hyb_mlp_group11,
        },
        out_dir / "group11_dv_vs_hybrid_mlp.png",
    )

    print(f"[Saved] plots in {out_dir}")


if __name__ == "__main__":
    main()