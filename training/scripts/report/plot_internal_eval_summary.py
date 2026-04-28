#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot internal evaluation summary from eval_checkpoint.py long CSV.

Main plots
----------
1. illumina_view:
   compares DV Illumina, Unimodal Illumina, Hybrid Simple, Hybrid Groupwise

2. ont_view:
   compares DV ONT, Unimodal ONT, Hybrid Simple, Hybrid Groupwise

Optional plots
--------------
If the CSV contains combined view+variant scopes, such as:

- illumina_view_SNP
- illumina_view_INDEL
- ont_view_SNP
- ont_view_INDEL

the script also generates:

3. illumina_variant_best_vs_dv:
   DV Illumina vs best non-teacher model on SNP and INDEL

4. ont_variant_best_vs_dv:
   DV ONT vs best non-teacher model on SNP and INDEL
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["acc", "macro_f1", "macro_recall", "macro_precision"]

METRIC_LABELS = {
    "acc": "Accuracy",
    "macro_f1": "Macro F1",
    "macro_recall": "Macro recall",
    "macro_precision": "Macro precision",
}

DISPLAY_NAMES = {
    "dv_illumina": "DV Illumina",
    "dv_ont": "DV ONT",
    "unimodal_illumina": "Unimodal Illumina",
    "unimodal_ont": "Unimodal ONT",
    "hybrid_simple": "Hybrid Simple",
    "hybrid_groupwise": "Hybrid Groupwise",
}

VIEW_MODEL_ORDER = {
    "illumina_view": [
        "dv_illumina",
        "unimodal_illumina",
        "hybrid_simple",
        "hybrid_groupwise",
    ],
    "ont_view": [
        "dv_ont",
        "unimodal_ont",
        "hybrid_simple",
        "hybrid_groupwise",
    ],
}

BASELINE_BY_VIEW = {
    "illumina_view": "dv_illumina",
    "ont_view": "dv_ont",
}


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(str(name), str(name).replace("_", " ").title())


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["name", "partition", "scope", "scope_type", "comparable_scope"]
    existing = [k for k in keys if k in df.columns]
    if not existing:
        return df
    return df.drop_duplicates(subset=existing, keep="last").copy()


def compute_zoomed_ylim(values, *, full_scale: bool = False) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0 or full_scale:
        return 0.0, 1.02

    vmin = float(vals.min())
    vmax = float(vals.max())

    if abs(vmax - vmin) < 1e-9:
        pad = max(0.0005, abs(vmax) * 0.0005)
    else:
        pad = max((vmax - vmin) * 0.35, 0.0005)

    ymin = max(0.0, vmin - pad)
    ymax = min(1.0008, vmax + pad)

    if ymax <= ymin:
        ymax = min(1.0008, ymin + 0.001)

    return ymin, ymax


def collect_metric_values(df: pd.DataFrame, metrics: List[str]) -> List[float]:
    vals = []
    for _, row in df.iterrows():
        for m in metrics:
            if m in row and pd.notna(row[m]):
                vals.append(float(row[m]))
    return vals


def ordered_view_df(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    sub = df[
        (df["scope"] == scope)
        & (df["scope_type"] == "view")
    ].copy()

    order = VIEW_MODEL_ORDER[scope]
    sub = sub[sub["name"].isin(order)].copy()
    sub["model_order"] = sub["name"].map({name: i for i, name in enumerate(order)})
    sub = sub.sort_values("model_order")

    return sub


def improvement_pct(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return 100.0 * (float(value) - float(baseline)) / abs(float(baseline))


def annotate_view_improvements(
    ax,
    *,
    bars_by_model: Dict[str, List],
    df: pd.DataFrame,
    metrics: List[str],
    baseline_name: str,
    y_limits: Tuple[float, float],
) -> None:
    base_rows = df[df["name"] == baseline_name]
    if base_rows.empty:
        print(f"[WARN] Baseline {baseline_name} not found. Skipping improvement annotations.")
        return

    base = base_rows.iloc[0]
    ymin, ymax = y_limits
    yrange = max(ymax - ymin, 1e-9)
    dy = yrange * 0.025

    for _, row in df.iterrows():
        name = row["name"]
        if name not in bars_by_model:
            continue

        bars = bars_by_model[name]
        for j, metric in enumerate(metrics):
            value = float(row[metric])
            baseline = float(base[metric])
            imp = improvement_pct(value, baseline)

            if name == baseline_name:
                label = "0%"
            else:
                label = f"{imp:+.3f}%"

            bar = bars[j]
            x = bar.get_x() + bar.get_width() / 2.0

            # Put label inside the visible axis if the bar is very close to ymax.
            y = min(value + dy, ymax - dy * 0.5)

            ax.text(
                x,
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )


def plot_grouped_metric_bars(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    metrics: List[str] = METRICS,
    y_limits: Optional[Tuple[float, float]] = None,
    annotate_improvement: bool = False,
    baseline_name: Optional[str] = None,
) -> None:
    if df.empty:
        print(f"[WARN] Empty dataframe for plot: {title}")
        return

    x = np.arange(len(metrics))
    n_models = len(df)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_by_model: Dict[str, List] = {}

    for i, (_, row) in enumerate(df.iterrows()):
        values = [float(row[m]) for m in metrics]
        offset = (i - (n_models - 1) / 2.0) * width
        bars = ax.bar(x + offset, values, width, label=display_name(row["name"]))
        bars_by_model[str(row["name"])] = list(bars)

    if y_limits is None:
        y_limits = compute_zoomed_ylim(collect_metric_values(df, metrics))

    ax.set_ylim(*y_limits)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    if annotate_improvement and baseline_name is not None:
        annotate_view_improvements(
            ax,
            bars_by_model=bars_by_model,
            df=df,
            metrics=metrics,
            baseline_name=baseline_name,
            y_limits=y_limits,
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[Saved] {out_path}")


def plot_view_comparisons(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    annotate_improvement: bool,
) -> None:
    illumina = ordered_view_df(df, "illumina_view")
    ont = ordered_view_df(df, "ont_view")

    # One shared zoomed Y scale for both view plots.
    all_values = []
    all_values.extend(collect_metric_values(illumina, METRICS))
    all_values.extend(collect_metric_values(ont, METRICS))
    shared_ylim = compute_zoomed_ylim(all_values)

    plot_grouped_metric_bars(
        illumina,
        title="Internal test — Illumina view (HG004 chr20)",
        out_path=out_dir / "illumina_view_metrics.png",
        y_limits=shared_ylim,
        annotate_improvement=annotate_improvement,
        baseline_name="dv_illumina",
    )

    plot_grouped_metric_bars(
        ont,
        title="Internal test — ONT view (HG004 chr20)",
        out_path=out_dir / "ont_view_metrics.png",
        y_limits=shared_ylim,
        annotate_improvement=annotate_improvement,
        baseline_name="dv_ont",
    )


def has_combined_variant_scopes(df: pd.DataFrame, view: str) -> bool:
    needed = {f"{view}_SNP", f"{view}_INDEL"}
    available = set(df["scope"].astype(str).unique())
    return needed.issubset(available)


def best_non_teacher_for_scope(
    df: pd.DataFrame,
    *,
    scope: str,
    best_metric: str,
) -> Optional[pd.Series]:
    sub = df[
        (df["scope"] == scope)
        & (df["model_kind"] != "teacher")
    ].copy()

    if sub.empty:
        return None

    sub[best_metric] = pd.to_numeric(sub[best_metric], errors="coerce")
    sub = sub.dropna(subset=[best_metric])
    if sub.empty:
        return None

    return sub.sort_values(best_metric, ascending=False).iloc[0]


def teacher_for_scope(
    df: pd.DataFrame,
    *,
    teacher_name: str,
    scope: str,
) -> Optional[pd.Series]:
    sub = df[
        (df["name"] == teacher_name)
        & (df["scope"] == scope)
    ].copy()

    if sub.empty:
        return None

    return sub.iloc[0]


def collect_variant_best_rows(
    df: pd.DataFrame,
    *,
    view: str,
    teacher_name: str,
    best_metric: str,
) -> List[Tuple[str, str, pd.Series]]:
    if not has_combined_variant_scopes(df, view):
        print(
            f"[WARN] CSV does not contain combined scopes "
            f"{view}_SNP and {view}_INDEL."
        )
        return []

    rows = []
    for vt in ["SNP", "INDEL"]:
        scope = f"{view}_{vt}"

        dv_row = teacher_for_scope(df, teacher_name=teacher_name, scope=scope)
        best_row = best_non_teacher_for_scope(df, scope=scope, best_metric=best_metric)

        if dv_row is None:
            print(f"[WARN] Missing teacher row for scope={scope}, teacher={teacher_name}")
            continue
        if best_row is None:
            print(f"[WARN] Missing non-teacher model rows for scope={scope}")
            continue

        rows.append((vt, "DV", dv_row))
        rows.append((vt, f"Best: {display_name(best_row['name'])}", best_row))

    return rows


def collect_values_from_variant_rows(
    rows: List[Tuple[str, str, pd.Series]],
    metrics: List[str],
) -> List[float]:
    vals = []
    for _, _, row in rows:
        for m in metrics:
            if m in row and pd.notna(row[m]):
                vals.append(float(row[m]))
    return vals


def plot_variant_best_vs_dv(
    *,
    rows: List[Tuple[str, str, pd.Series]],
    title: str,
    out_path: Path,
    y_limits: Tuple[float, float],
    metrics: List[str] = METRICS,
) -> None:
    if not rows:
        return

    variant_types = ["SNP", "INDEL"]
    x = np.arange(len(metrics))
    width = 0.36

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(variant_types),
        figsize=(13, 5.5),
        sharey=True,
    )

    if len(variant_types) == 1:
        axes = [axes]

    for ax, vt in zip(axes, variant_types):
        vt_rows = [(label, row) for row_vt, label, row in rows if row_vt == vt]

        for i, (label, row) in enumerate(vt_rows):
            values = [float(row[m]) for m in metrics]
            offset = (i - (len(vt_rows) - 1) / 2.0) * width
            ax.bar(x + offset, values, width, label=label)

        ax.set_title(vt)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=20, ha="right")
        ax.set_ylim(*y_limits)
        ax.grid(axis="y", alpha=0.25)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="lower right", frameon=True)
    fig.suptitle(title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[Saved] {out_path}")


def plot_optional_variant_comparisons(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    best_metric: str,
) -> None:
    illumina_rows = collect_variant_best_rows(
        df,
        view="illumina_view",
        teacher_name="dv_illumina",
        best_metric=best_metric,
    )

    ont_rows = collect_variant_best_rows(
        df,
        view="ont_view",
        teacher_name="dv_ont",
        best_metric=best_metric,
    )

    # One shared zoomed Y scale for both Illumina/ONT SNP/INDEL plots.
    all_values = []
    all_values.extend(collect_values_from_variant_rows(illumina_rows, METRICS))
    all_values.extend(collect_values_from_variant_rows(ont_rows, METRICS))
    shared_ylim = compute_zoomed_ylim(all_values)

    plot_variant_best_vs_dv(
        rows=illumina_rows,
        title=f"Internal test — DV Illumina vs best model by SNP/INDEL ({best_metric})",
        out_path=out_dir / "illumina_variant_best_vs_dv.png",
        y_limits=shared_ylim,
    )

    plot_variant_best_vs_dv(
        rows=ont_rows,
        title=f"Internal test — DV ONT vs best model by SNP/INDEL ({best_metric})",
        out_path=out_dir / "ont_variant_best_vs_dv.png",
        y_limits=shared_ylim,
    )


def row_for_name_scope(
    df: pd.DataFrame,
    *,
    name: str,
    scope: str,
) -> Optional[pd.Series]:
    sub = df[
        (df["name"] == name)
        & (df["scope"] == scope)
    ].copy()

    if sub.empty:
        return None

    return sub.iloc[-1]


def row_for_name_scope_candidates(
    df: pd.DataFrame,
    *,
    name: str,
    scopes: List[str],
) -> Optional[pd.Series]:
    for scope in scopes:
        row = row_for_name_scope(df, name=name, scope=scope)
        if row is not None:
            return row
    print(f"[WARN] Missing row for name={name} with any of scopes={scopes}")
    return None


def build_full_potential_df(
    df: pd.DataFrame,
    *,
    hybrid_name: str,
) -> pd.DataFrame:
    """
    Practical comparison:
    - DV Illumina on illumina_view
    - DV ONT on ont_view
    - Best multimodal model on hybrid_all
    """
    specs = [
        ("dv_illumina", "illumina_view"),
        ("dv_ont", "ont_view"),
        (hybrid_name, "hybrid_all"),
    ]

    rows = []
    for name, scope in specs:
        row = row_for_name_scope(df, name=name, scope=scope)
        if row is None:
            print(f"[WARN] Missing row for full-potential plot: name={name}, scope={scope}")
            continue
        rows.append(row.to_dict())

    if not rows:
        return pd.DataFrame(columns=df.columns)

    out = pd.DataFrame(rows)
    order = {
        "dv_illumina": 0,
        "dv_ont": 1,
        hybrid_name: 2,
    }
    out["model_order"] = out["name"].map(order)
    out = out.sort_values("model_order")

    return out


def plot_full_potential_overall(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    hybrid_name: str,
) -> None:
    sub = build_full_potential_df(df, hybrid_name=hybrid_name)

    if sub.empty:
        print("[WARN] Skipping full-potential overall plot: no rows found.")
        return

    y_limits = compute_zoomed_ylim(collect_metric_values(sub, METRICS))

    plot_grouped_metric_bars(
        sub,
        title="Internal test — Practical comparison (DV Illumina vs DV ONT vs Hybrid)",
        out_path=out_dir / "full_potential_overall.png",
        y_limits=y_limits,
        annotate_improvement=False,
        baseline_name=None,
    )


def collect_full_potential_variant_rows(
    df: pd.DataFrame,
    *,
    hybrid_name: str,
) -> List[Tuple[str, str, pd.Series]]:
    """
    Returns rows as:
      (variant_type, model_name, row)

    Preferred scopes:
      - dv_illumina -> illumina_view_SNP / illumina_view_INDEL
      - dv_ont      -> ont_view_SNP / ont_view_INDEL
      - hybrid      -> hybrid_all_SNP / hybrid_all_INDEL

    Fallbacks:
      - variant_SNP / variant_INDEL
    """
    rows: List[Tuple[str, str, pd.Series]] = []

    specs = {
        "SNP": [
            ("dv_illumina", ["illumina_view_SNP", "variant_SNP"]),
            ("dv_ont", ["ont_view_SNP", "variant_SNP"]),
            (hybrid_name, ["hybrid_all_SNP", "variant_SNP"]),
        ],
        "INDEL": [
            ("dv_illumina", ["illumina_view_INDEL", "variant_INDEL"]),
            ("dv_ont", ["ont_view_INDEL", "variant_INDEL"]),
            (hybrid_name, ["hybrid_all_INDEL", "variant_INDEL"]),
        ],
    }

    for vt, vt_specs in specs.items():
        for model_name, scope_candidates in vt_specs:
            row = row_for_name_scope_candidates(
                df,
                name=model_name,
                scopes=scope_candidates,
            )
            if row is None:
                continue
            rows.append((vt, model_name, row))

    return rows


def plot_full_potential_variant(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    hybrid_name: str,
    metrics: List[str] = METRICS,
) -> None:
    rows = collect_full_potential_variant_rows(df, hybrid_name=hybrid_name)

    if not rows:
        print("[WARN] Skipping full-potential variant plot: no rows found.")
        return

    variant_types = ["SNP", "INDEL"]
    x = np.arange(len(metrics))
    width = 0.24

    all_values = collect_values_from_variant_rows(rows, metrics)
    shared_ylim = compute_zoomed_ylim(all_values)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13.5, 5.5),
        sharey=True,
    )

    for ax, vt in zip(axes, variant_types):
        vt_rows = [(model_name, row) for row_vt, model_name, row in rows if row_vt == vt]

        for i, (model_name, row) in enumerate(vt_rows):
            values = [float(row[m]) for m in metrics]
            offset = (i - (len(vt_rows) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=display_name(model_name),
            )

        ax.set_title(vt)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=20, ha="right")
        ax.set_ylim(*shared_ylim)
        ax.grid(axis="y", alpha=0.25)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="lower right", frameon=True)

    fig.suptitle(
        "Internal test — Practical SNP/INDEL comparison (DV Illumina vs DV ONT vs Hybrid)"
    )
    fig.tight_layout()

    out_path = out_dir / "full_potential_variant.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[Saved] {out_path}")


def plot_full_potential(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    hybrid_name: str,
) -> None:
    plot_full_potential_overall(
        df,
        out_dir=out_dir,
        hybrid_name=hybrid_name,
    )

    plot_full_potential_variant(
        df,
        out_dir=out_dir,
        hybrid_name=hybrid_name,
    )    


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Long CSV from eval_checkpoint.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--partition", default="test")
    ap.add_argument(
        "--best_metric",
        default="macro_f1",
        choices=[
            "acc",
            "macro_f1",
            "macro_recall",
            "macro_precision",
            "weighted_f1",
            "variant_recall",
            "variant_f1",
            "binary_f1",
        ],
    )
    ap.add_argument(
        "--skip_variant_best",
        action="store_true",
        help="Skip optional DV vs best-model SNP/INDEL plots.",
    )
    ap.add_argument(
        "--no_annotate_improvement",
        action="store_true",
        help="Do not annotate relative improvement vs DV in view plots.",
    )
    ap.add_argument(
        "--skip_full_potential",
        action="store_true",
        help="Skip practical full-potential plots (DV Illumina vs DV ONT vs best multimodal).",
    )
    ap.add_argument(
        "--full_potential_model",
        default="hybrid_groupwise",
        choices=["hybrid_groupwise", "hybrid_simple"],
        help="Multimodal model to use for the practical full-potential plots.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = dedupe_rows(df)

    required = {
        "name",
        "partition",
        "scope",
        "scope_type",
        "model_kind",
        "modality",
        "model_family",
        *METRICS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    df = df[df["partition"] == args.partition].copy()
    df = ensure_numeric(
        df,
        [
            "acc",
            "macro_f1",
            "macro_recall",
            "macro_precision",
            "weighted_f1",
            "variant_recall",
            "variant_f1",
            "binary_f1",
        ],
    )

    if df.empty:
        raise SystemExit(f"No rows found for partition={args.partition}")

    plot_view_comparisons(
        df,
        out_dir,
        annotate_improvement=not args.no_annotate_improvement,
    )

    if not args.skip_variant_best:
        plot_optional_variant_comparisons(
            df,
            out_dir=out_dir,
            best_metric=args.best_metric,
        )

    if not args.skip_full_potential:
        plot_full_potential(
            df,
            out_dir=out_dir,
            hybrid_name=args.full_potential_model,
        )

    print("\n[Done]")


if __name__ == "__main__":
    main()