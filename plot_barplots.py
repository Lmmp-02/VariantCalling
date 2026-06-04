from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------

METRICS = ["acc", "macro_precision", "macro_recall", "macro_f1"]
COVERAGES = ["5x", "10x", "20x", "40x"]
VARIANT_TYPES = ["SNP", "INDEL"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title_es: str
    title_en: str
    csv_relpath: str

    def title(self, language: str) -> str:
        return self.title_es if language == "es" else self.title_en


DATASETS = {
    "HG003": DatasetSpec(
        key="HG003",
        title_es="HG003 chr20",
        title_en="HG003 chr20",
        csv_relpath="HG003_ch20_initial_test/hg003_summary_internal_test_long.csv",
    ),
    "HG004": DatasetSpec(
        key="HG004",
        title_es="HG004 chr20",
        title_en="HG004 chr20",
        csv_relpath="HG004_chr20/summary_internal_test_long.csv",
    ),
    "HG005": DatasetSpec(
        key="HG005",
        title_es="HG005 chr20+chr21, 40x",
        title_en="HG005 chr20+chr21, 40x",
        csv_relpath="HG005_chr20_chr21_40x/summary_hg005_trainmode_40x_long.csv",
    ),
}

HG005_COVERAGE_DATASETS = {
    "5x": DatasetSpec(
        key="HG005_5x",
        title_es="HG005 chr20+chr21, 5x",
        title_en="HG005 chr20+chr21, 5x",
        csv_relpath="HG005_chr20_chr21_5x/summary_hg005_trainmode_5x_long.csv",
    ),
    "10x": DatasetSpec(
        key="HG005_10x",
        title_es="HG005 chr20+chr21, 10x",
        title_en="HG005 chr20+chr21, 10x",
        csv_relpath="HG005_chr20_chr21_10x/summary_hg005_trainmode_10x_long.csv",
    ),
    "20x": DatasetSpec(
        key="HG005_20x",
        title_es="HG005 chr20+chr21, 20x",
        title_en="HG005 chr20+chr21, 20x",
        csv_relpath="HG005_chr20_chr21_20x/summary_hg005_trainmode_20x_long.csv",
    ),
    "40x": DatasetSpec(
        key="HG005_40x",
        title_es="HG005 chr20+chr21, 40x",
        title_en="HG005 chr20+chr21, 40x",
        csv_relpath="HG005_chr20_chr21_40x/summary_hg005_trainmode_40x_long.csv",
    ),
}


# -----------------------------------------------------------------------------
# Labels and visual style
# -----------------------------------------------------------------------------

TEXT = {
    "es": {
        "score": "Score",
        "count": "Número",
        "coverage": "Cobertura",
        "model": "Modelo",
        "model_variant": "Modelo y tipo de variante",
        "metric": "Métrica",
        "macro_f1": "Macro F1",
        "f1": "F1",
        "precision": "Precisión",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "variant_recall": "Variant recall",
        "variant_fp": "FP variantes",
        "variant_fn": "FN variantes",
        "illumina_view": "Illumina",
        "ont_view": "ONT",
        "scatter_title": "Mapa precision-recall",
        "scatter_xlabel": "Recall",
        "scatter_ylabel": "Precisión",
        "coverage_overview": "Evolución por cobertura (macro F1)",
        "coverage_variant": "Cobertura por tipo de variante (F1)",
        "two_stage": "HG003 chr20 - ablación one-stage vs two-stage",
        "variant_fp_fn": "FP/FN de variantes",
        "snp_indel": "SNP/INDEL",
    },
    "en": {
        "score": "Score",
        "count": "Count",
        "coverage": "Coverage",
        "model": "Model",
        "model_variant": "Model and variant type",
        "metric": "Metric",
        "macro_f1": "Macro F1",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "variant_recall": "Variant recall",
        "variant_fp": "Variant FP",
        "variant_fn": "Variant FN",
        "illumina_view": "Illumina view",
        "ont_view": "ONT view",
        "scatter_title": "Precision-recall map",
        "scatter_xlabel": "Recall",
        "scatter_ylabel": "Precision",
        "coverage_overview": "Coverage trend (macro F1)",
        "coverage_variant": "Coverage by variant type (F1)",
        "two_stage": "HG003 chr20 - one-stage vs two-stage ablation",
        "variant_fp_fn": "Variant FP/FN",
        "snp_indel": "SNP/INDEL",
    },
}

METRIC_LABELS = {
    "es": {
        "acc": "Accuracy",
        "macro_precision": "Precisión",
        "macro_recall": "Recall",
        "macro_f1": "F1",
        "variant_recall": "Variant recall",
        "variant_fp": "FP",
        "variant_fn": "FN",
    },
    "en": {
        "acc": "Accuracy",
        "macro_precision": "Precision",
        "macro_recall": "Recall",
        "macro_f1": "F1",
        "variant_recall": "Variant recall",
        "variant_fp": "FP",
        "variant_fn": "FN",
    },
}

# Keep model labels mostly in English because these were the cleaner names used in
# the original Chapter 6 figures and are shorter than literal Spanish translations.
MODEL_LABELS = {
    "es": {
        "dv_illumina": "DV Illumina",
        "dv_ont": "DV ONT",
        "unimodal_illumina_linear": "Unimodal linear",
        "unimodal_illumina_mlp": "Unimodal MLP",
        "unimodal_ont_linear": "Unimodal linear",
        "unimodal_ont_mlp": "Unimodal MLP",
        "unimodal_illumina": "Unimodal",
        "unimodal_ont": "Unimodal",
        "hybrid_simple_linear": "Hybrid simple linear",
        "hybrid_simple_mlp": "Hybrid simple MLP",
        "hybrid_groupwise_linear": "Hybrid groupwise linear",
        "hybrid_groupwise_mlp": "Hybrid groupwise MLP",
        "hybrid_simple": "Hybrid simple",
        "hybrid_groupwise": "Hybrid groupwise",
        "Hybrid simple direct 3c": "Simple direct 3c",
        "Hybrid groupwise direct 3c": "Groupwise direct 3c",
        "Hybrid simple 2-stage raw": "Simple 2-stage raw",
        "Hybrid simple 2-stage 3c calib": "Simple 2-stage calib.",
        "Hybrid groupwise 2-stage raw": "Groupwise 2-stage raw",
        "Hybrid groupwise 2-stage 3c calib": "Groupwise 2-stage calib.",
    },
    "en": {
        "dv_illumina": "DV Illumina",
        "dv_ont": "DV ONT",
        "unimodal_illumina_linear": "Unimodal linear",
        "unimodal_illumina_mlp": "Unimodal MLP",
        "unimodal_ont_linear": "Unimodal linear",
        "unimodal_ont_mlp": "Unimodal MLP",
        "unimodal_illumina": "Unimodal",
        "unimodal_ont": "Unimodal",
        "hybrid_simple_linear": "Hybrid simple linear",
        "hybrid_simple_mlp": "Hybrid simple MLP",
        "hybrid_groupwise_linear": "Hybrid groupwise linear",
        "hybrid_groupwise_mlp": "Hybrid groupwise MLP",
        "hybrid_simple": "Hybrid simple",
        "hybrid_groupwise": "Hybrid groupwise",
        "Hybrid simple direct 3c": "Simple direct 3c",
        "Hybrid groupwise direct 3c": "Groupwise direct 3c",
        "Hybrid simple 2-stage raw": "Simple 2-stage raw",
        "Hybrid simple 2-stage 3c calib": "Simple 2-stage calib.",
        "Hybrid groupwise 2-stage raw": "Groupwise 2-stage raw",
        "Hybrid groupwise 2-stage 3c calib": "Groupwise 2-stage calib.",
    },
}

VIEW_INFO = {
    "illumina_view": {
        "dv": "dv_illumina",
        "unimodal": "unimodal_illumina",
        "unimodal_linear": "unimodal_illumina_linear",
        "unimodal_mlp": "unimodal_illumina_mlp",
    },
    "ont_view": {
        "dv": "dv_ont",
        "unimodal": "unimodal_ont",
        "unimodal_linear": "unimodal_ont_linear",
        "unimodal_mlp": "unimodal_ont_mlp",
    },
}

# Muted/pastel palette derived from the original script's navy/teal/amber/wine family.
# The colors remain clearly distinguishable, but avoid the more electric look of v1.
MODEL_COLORS = {
    "dv_illumina": "#5F7FA3",
    "dv_ont": "#5F7FA3",
    "unimodal_illumina_linear": "#D7A461",
    "unimodal_ont_linear": "#D7A461",
    "unimodal_illumina_mlp": "#7EAE91",
    "unimodal_ont_mlp": "#7EAE91",
    "unimodal_illumina": "#7EAE91",
    "unimodal_ont": "#7EAE91",
    "hybrid_simple_linear": "#C77C7C",
    "hybrid_simple_mlp": "#C77C7C",
    "hybrid_groupwise_linear": "#8D89B8",
    "hybrid_groupwise_mlp": "#8D89B8",
    "hybrid_simple": "#C77C7C",
    "hybrid_groupwise": "#8D89B8",
    "Hybrid simple direct 3c": "#C77C7C",
    "Hybrid groupwise direct 3c": "#8D89B8",
    "Hybrid simple 2-stage raw": "#D7A461",
    "Hybrid simple 2-stage 3c calib": "#7EAE91",
    "Hybrid groupwise 2-stage raw": "#D7A461",
    "Hybrid groupwise 2-stage 3c calib": "#7EAE91",
}

FALLBACK_COLORS = [
    "#5F7FA3", "#D7A461", "#7EAE91", "#C77C7C", "#8D89B8",
    "#6CA6A3", "#B9926B", "#8A94A3", "#B07A91", "#6E8F79",
]

EDGE_COLOR = "#111827"
GRID_ALPHA = 0.22
LABEL_PADDING = 5


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def t(language: str, key: str) -> str:
    return TEXT[language].get(key, key)


def metric_label(metric: str, language: str) -> str:
    return METRIC_LABELS[language].get(metric, metric)


def view_label(scope: str, language: str) -> str:
    return t(language, scope)


def friendly_model_name(name: str, language: str) -> str:
    labels = MODEL_LABELS[language]
    if name in labels:
        return labels[name]
    return str(name).replace("_", " ").replace("-", " ").title()


def color_for_model(name: str) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    return FALLBACK_COLORS[abs(hash(name)) % len(FALLBACK_COLORS)]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, out_path: Path, *, save_svg: bool = False) -> None:
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "n", "acc", "macro_f1", "macro_recall", "macro_precision", "weighted_f1",
        "variant_precision", "variant_recall", "variant_specificity", "variant_balanced_acc",
        "variant_f1", "variant_f2", "variant_tp", "variant_tn", "variant_fp", "variant_fn",
        "binary_acc", "binary_balanced_acc", "binary_precision", "binary_recall",
        "binary_specificity", "binary_f1", "binary_f2",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_summary(data_root: Path, spec: DatasetSpec, *, required: bool = False) -> pd.DataFrame | None:
    path = data_root / spec.csv_relpath
    if not path.exists():
        msg = f"[AVISO] No existe el CSV esperado: {path}"
        if required:
            raise FileNotFoundError(msg)
        print(msg)
        return None
    return clean_numeric(pd.read_csv(path))


def filter_rows(df: pd.DataFrame, scope: str, names: Sequence[str], variant_type: str = "ALL") -> pd.DataFrame:
    if "scope" not in df.columns or "name" not in df.columns:
        return pd.DataFrame()
    sub = df[(df["scope"] == scope) & (df["name"].isin(names))].copy()
    if "variant_type" in sub.columns:
        sub = sub[sub["variant_type"] == variant_type].copy()
    order = {name: i for i, name in enumerate(names)}
    sub["_requested_order"] = sub["name"].map(order)
    sub = sub.sort_values(["_requested_order", "name"]).drop(columns=["_requested_order"])
    return sub


def sort_names_by_f1(df: pd.DataFrame, names: Sequence[str], *, descending: bool = True) -> list[str]:
    if df.empty or "macro_f1" not in df.columns:
        return list(names)
    sub = df[df["name"].isin(names)][["name", "macro_f1"]].drop_duplicates("name")
    f1 = dict(zip(sub["name"], sub["macro_f1"]))
    if descending:
        return sorted(names, key=lambda n: (-f1.get(n, -math.inf), n))
    return sorted(names, key=lambda n: (f1.get(n, math.inf), n))


def ordered_models(
    df: pd.DataFrame,
    names: Sequence[str],
    *,
    baseline_name: str | None = None,
    sort_by_f1: bool = True,
) -> list[str]:
    """Return a readable order while keeping the DV baseline on the right.

    The non-baseline models can still be sorted by F1, but the matching DV
    baseline is appended at the end so DV Illumina / DV ONT appear on the
    right consistently across barplots.
    """
    requested = list(names)
    if baseline_name and baseline_name in requested:
        non_baseline = [n for n in requested if n != baseline_name]
        ordered = sort_names_by_f1(df, non_baseline) if sort_by_f1 else non_baseline
        return ordered + [baseline_name]
    return sort_names_by_f1(df, requested) if sort_by_f1 else requested


def finite_values(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(v)]


def nice_step_at_least(required: float) -> float:
    required = max(float(required), 1e-7)
    exponent = math.floor(math.log10(required))
    base = 10 ** exponent
    for mult in [1, 2, 2.5, 5, 10]:
        step = mult * base
        if step >= required:
            return step
    return 10 * base


def score_axis_spec(values: Iterable[float], *, n_ticks_below_one: int = 5) -> tuple[float, float, list[float], float]:
    """Zoomed score axis with 1.0 as the top grid tick and label space above."""
    vals = finite_values(values)
    if not vals:
        step = 0.001
    else:
        min_value = min(vals)
        required = (1.0 - min_value) / max(n_ticks_below_one, 1)
        step = nice_step_at_least(required * 1.12)
    ticks = [1.0 - i * step for i in range(n_ticks_below_one + 1)]
    bottom = ticks[-1]
    top = 1.0 + max(step * 0.95, 0.0010)
    return bottom, top, ticks, step


def score_tick_formatter(step: float) -> FuncFormatter:
    if step < 0.001:
        fmt = "{:.4f}"
    elif step < 0.01:
        fmt = "{:.3f}"
    else:
        fmt = "{:.2f}"
    return FuncFormatter(lambda y, _: fmt.format(y))


def apply_score_axis(ax: plt.Axes, axis_spec: tuple[float, float, list[float], float]) -> None:
    bottom, top, ticks, step = axis_spec
    ax.set_ylim(bottom, top)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(score_tick_formatter(step))
    ax.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)


def set_zoom_axis_from_xy(ax: plt.Axes, xs: Sequence[float], ys: Sequence[float]) -> None:
    vals_x = finite_values(xs)
    vals_y = finite_values(ys)
    if not vals_x or not vals_y:
        return
    min_x, max_x = min(vals_x), max(vals_x)
    min_y, max_y = min(vals_y), max(vals_y)
    pad_x = max((max_x - min_x) * 0.18, 0.0007)
    pad_y = max((max_y - min_y) * 0.18, 0.0007)
    ax.set_xlim(max(0.0, min_x - pad_x), min(1.0005, max_x + pad_x))
    ax.set_ylim(max(0.0, min_y - pad_y), min(1.0005, max_y + pad_y))
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)


def format_pct(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return ""
    # Baselines should read as 0%, not 0.00%, which is cleaner and easier to scan.
    if abs(value) < 0.5 * (10 ** -decimals):
        return "0%"
    sign = "+" if value > 0 else ""
    if decimals <= 0:
        return f"{sign}{value:.0f}%"
    return f"{sign}{value:.{decimals}f}%"


def rel_pct_change(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0:
        return np.nan
    return ((value - baseline) / abs(baseline)) * 100.0


def add_bar_labels(
    ax: plt.Axes,
    containers: Sequence,
    labels_by_container: Sequence[Sequence[str]],
    *,
    fontsize: int = 7,
    rotation: int = 90,
) -> None:
    for container, labels in zip(containers, labels_by_container):
        ax.bar_label(container, labels=list(labels), fontsize=fontsize, rotation=rotation, padding=LABEL_PADDING)


def add_bar_labels_manual(
    ax: plt.Axes,
    containers: Sequence,
    labels_by_container: Sequence[Sequence[str]],
    *,
    fontsize: int = 7,
    rotation: int = 0,
    y_offset_points: int = 7,
) -> None:
    """Place labels manually so tight y-ranges do not hide them.

    This is especially useful for Figure 6.3, where several values sit very close
    to 1.0 and the normal bar_label call can be clipped or visually lost.
    """
    for container, labels in zip(containers, labels_by_container):
        for bar, label in zip(container.patches, labels):
            if not label:
                continue
            height = bar.get_height()
            if not np.isfinite(height):
                continue
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, y_offset_points),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                rotation=rotation,
                clip_on=False,
            )


def external_legend(ax: plt.Axes, *, title: str, language: str, ncol: int = 1) -> None:
    ax.legend(
        title=title,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        ncol=ncol,
        borderaxespad=0.0,
    )


def collect_metric_values(df: pd.DataFrame, scope: str, names: Sequence[str], metrics: Sequence[str]) -> list[float]:
    values: list[float] = []
    sub = filter_rows(df, scope, list(names), "ALL")
    for metric in metrics:
        if metric in sub.columns:
            values.extend(pd.to_numeric(sub[metric], errors="coerce").dropna().tolist())
    return values


# -----------------------------------------------------------------------------
# Plotting primitives
# -----------------------------------------------------------------------------


def plot_grouped_metric_bars_single_view(
    df: pd.DataFrame,
    *,
    scope: str,
    model_order: Sequence[str],
    metrics: Sequence[str],
    title: str,
    out_path: Path,
    language: str,
    baseline_name: str | None,
    pct_decimals: int,
    save_svg: bool,
    label_mode: str = "pct_vs_baseline",
    sort_by_f1: bool = True,
    shared_axis_spec: tuple[float, float, list[float], float] | None = None,
) -> pd.DataFrame:
    requested = list(model_order)
    sub = filter_rows(df, scope, requested, "ALL")
    if sub.empty:
        print(f"[AVISO] Sin datos para {out_path.name}")
        return pd.DataFrame()

    order = ordered_models(sub, requested, baseline_name=baseline_name, sort_by_f1=sort_by_f1)
    axis_spec = shared_axis_spec if shared_axis_spec is not None else score_axis_spec(collect_metric_values(df, scope, order, metrics))
    rows = sub.drop_duplicates("name").set_index("name")

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x = np.arange(len(metrics))
    n_models = len(order)
    width = min(0.82 / max(n_models, 1), 0.18)

    baseline_values: dict[str, float] = {}
    if baseline_name and baseline_name in rows.index:
        baseline_values = {metric: float(rows.loc[baseline_name, metric]) for metric in metrics if metric in rows.columns}

    containers = []
    labels_by_container: list[list[str]] = []
    manifest = []

    for i, name in enumerate(order):
        values = [float(rows.loc[name, metric]) if name in rows.index and metric in rows.columns else np.nan for metric in metrics]
        offsets = x + (i - (n_models - 1) / 2) * width
        container = ax.bar(
            offsets,
            values,
            width=width,
            label=friendly_model_name(name, language),
            color=color_for_model(name),
            edgecolor=EDGE_COLOR,
            linewidth=0.55,
        )
        containers.append(container)

        labels = []
        for metric, value in zip(metrics, values):
            relative_change = np.nan
            baseline_value = baseline_values.get(metric, np.nan)
            if label_mode == "pct_vs_baseline" and metric in baseline_values:
                relative_change = 0.0 if name == baseline_name else rel_pct_change(value, baseline_values[metric])
                labels.append(format_pct(relative_change, pct_decimals))
            elif label_mode == "value":
                labels.append(f"{value:.4f}" if np.isfinite(value) else "")
            else:
                labels.append("")

            manifest.append({
                "figure": out_path.stem,
                "scope": scope,
                "variant_type": "ALL",
                "model": name,
                "metric": metric,
                "value": value,
                "baseline_model": baseline_name or "",
                "baseline_value": baseline_value,
                "relative_change_pct": relative_change,
            })
        labels_by_container.append(labels)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_label(m, language) for m in metrics])
    ax.set_xlabel(t(language, "metric"))
    ax.set_ylabel(t(language, "score"))
    apply_score_axis(ax, axis_spec)
    external_legend(ax, title=t(language, "model"), language=language)
    add_bar_labels(ax, containers, labels_by_container, fontsize=7, rotation=90)
    fig.tight_layout()
    savefig(fig, out_path, save_svg=save_svg)
    return pd.DataFrame(manifest)


def plot_variant_breakdown_single_view(
    df: pd.DataFrame,
    *,
    scope: str,
    model_order: Sequence[str],
    title: str,
    out_path: Path,
    language: str,
    metrics: Sequence[str] = ("macro_f1", "macro_recall"),
    baseline_name: str | None = None,
    labels: str = "pct_vs_baseline",
    pct_decimals: int = 2,
    save_svg: bool = False,
    sort_by_overall_f1: bool = True,
    shared_axis_spec: tuple[float, float, list[float], float] | None = None,
) -> pd.DataFrame:
    requested = list(model_order)
    overall = filter_rows(df, scope, requested, "ALL")
    if overall.empty:
        print(f"[AVISO] Sin datos para {out_path.name}")
        return pd.DataFrame()

    order = ordered_models(overall, requested, baseline_name=baseline_name, sort_by_f1=sort_by_overall_f1)
    all_values = []
    for vt in VARIANT_TYPES:
        rows_vt = df[(df.get("scope") == f"{scope}_{vt}") & (df.get("name").isin(order))]
        if "variant_type" in rows_vt.columns:
            rows_vt = rows_vt[rows_vt["variant_type"] == vt]
        for metric in metrics:
            if metric in rows_vt.columns:
                all_values.extend(pd.to_numeric(rows_vt[metric], errors="coerce").dropna().tolist())

    if not all_values:
        print(f"[AVISO] Sin datos SNP/INDEL para {out_path.name}")
        return pd.DataFrame()

    axis_spec = shared_axis_spec if shared_axis_spec is not None else score_axis_spec(all_values)
    categories = [f"{vt}\n{metric_label(metric, language)}" for vt in VARIANT_TYPES for metric in metrics]
    x = np.arange(len(categories))
    width = min(0.82 / max(len(order), 1), 0.18)

    baseline_by_category: dict[tuple[str, str], float] = {}
    if baseline_name:
        for vt in VARIANT_TYPES:
            rows_base = df[(df["scope"] == f"{scope}_{vt}") & (df["name"] == baseline_name)]
            if "variant_type" in rows_base.columns:
                rows_base = rows_base[rows_base["variant_type"] == vt]
            if not rows_base.empty:
                for metric in metrics:
                    if metric in rows_base.columns:
                        baseline_by_category[(vt, metric)] = float(rows_base.iloc[0][metric])

    # Value labels are used in Figure 6.3. Make these bars a little wider and
    # space the pair slightly more so horizontal labels remain legible.
    value_label_mode = labels == "value"
    if value_label_mode:
        width = min(0.28, 0.72 / max(len(order), 1))
        offset_scale = 1.18
    else:
        offset_scale = 1.0

    fig, ax = plt.subplots(figsize=(11.4, 6.1 if value_label_mode else 5.8))
    containers = []
    labels_by_container: list[list[str]] = []
    manifest: list[dict] = []

    for i, name in enumerate(order):
        values = []
        current_labels = []
        for vt in VARIANT_TYPES:
            rows = df[(df["scope"] == f"{scope}_{vt}") & (df["name"] == name)]
            if "variant_type" in rows.columns:
                rows = rows[rows["variant_type"] == vt]
            for metric in metrics:
                value = float(rows.iloc[0][metric]) if not rows.empty and metric in rows.columns else np.nan
                values.append(value)
                baseline_value = baseline_by_category.get((vt, metric), np.nan)
                relative_change = np.nan
                if labels == "pct_vs_baseline" and baseline_name and (vt, metric) in baseline_by_category:
                    relative_change = 0.0 if name == baseline_name else rel_pct_change(value, baseline_value)
                    current_labels.append(format_pct(relative_change, pct_decimals))
                elif labels == "value":
                    current_labels.append(f"{value:.4f}" if np.isfinite(value) else "")
                else:
                    current_labels.append("")
                manifest.append({
                    "figure": out_path.stem,
                    "scope": scope,
                    "variant_type": vt,
                    "model": name,
                    "metric": metric,
                    "value": value,
                    "baseline_model": baseline_name or "",
                    "baseline_value": baseline_value,
                    "relative_change_pct": relative_change,
                })

        offsets = x + (i - (len(order) - 1) / 2) * width * offset_scale
        container = ax.bar(
            offsets,
            values,
            width=width,
            label=friendly_model_name(name, language),
            color=color_for_model(name),
            edgecolor=EDGE_COLOR,
            linewidth=0.55,
        )
        containers.append(container)
        labels_by_container.append(current_labels)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel(t(language, "snp_indel"))
    ax.set_ylabel(t(language, "score"))
    apply_score_axis(ax, axis_spec)
    external_legend(ax, title=t(language, "model"), language=language)
    if labels == "value":
        add_bar_labels_manual(ax, containers, labels_by_container, fontsize=7, rotation=0, y_offset_points=8)
    else:
        add_bar_labels(ax, containers, labels_by_container, fontsize=7, rotation=90)
    fig.tight_layout()
    savefig(fig, out_path, save_svg=save_svg)
    return pd.DataFrame(manifest)


# -----------------------------------------------------------------------------
# Specific figures
# -----------------------------------------------------------------------------


def make_standard_view_figures(
    df: pd.DataFrame,
    *,
    dataset_title: str,
    names_by_view: Mapping[str, Sequence[str]],
    title_es: str,
    title_en: str,
    out_stem: str,
    out_dir: Path,
    language: str,
    pct_decimals: int,
    save_svg: bool,
    sort_by_f1: bool = True,
    shared_y_axis: bool = False,
) -> list[pd.DataFrame]:
    outputs = []
    shared_axis_spec = None
    if shared_y_axis:
        shared_values: list[float] = []
        for shared_scope in ["illumina_view", "ont_view"]:
            shared_values.extend(collect_metric_values(df, shared_scope, list(names_by_view[shared_scope]), metrics=METRICS))
        if shared_values:
            shared_axis_spec = score_axis_spec(shared_values)
    for scope in ["illumina_view", "ont_view"]:
        baseline = VIEW_INFO[scope]["dv"] if VIEW_INFO[scope]["dv"] in names_by_view[scope] else None
        title_base = title_es if language == "es" else title_en
        title = f"{dataset_title} - {view_label(scope, language)} - {title_base}"
        outputs.append(plot_grouped_metric_bars_single_view(
            df,
            scope=scope,
            model_order=names_by_view[scope],
            metrics=METRICS,
            title=title,
            out_path=out_dir / f"{out_stem}_{scope}.png",
            language=language,
            baseline_name=baseline,
            pct_decimals=pct_decimals,
            save_svg=save_svg,
            label_mode="pct_vs_baseline" if baseline else "none",
            sort_by_f1=sort_by_f1,
            shared_axis_spec=shared_axis_spec,
        ))
    return outputs


def make_variant_view_figures(
    df: pd.DataFrame,
    *,
    dataset_title: str,
    names_by_view: Mapping[str, Sequence[str]],
    title_es: str,
    title_en: str,
    out_stem: str,
    out_dir: Path,
    language: str,
    pct_decimals: int,
    save_svg: bool,
    labels: str = "pct_vs_baseline",
    shared_y_axis: bool = False,
) -> list[pd.DataFrame]:
    outputs = []
    shared_axis_spec = None
    if shared_y_axis:
        shared_values: list[float] = []
        for shared_scope in ["illumina_view", "ont_view"]:
            for vt in VARIANT_TYPES:
                rows_vt = df[(df.get("scope") == f"{shared_scope}_{vt}") & (df.get("name").isin(list(names_by_view[shared_scope])))]
                if "variant_type" in rows_vt.columns:
                    rows_vt = rows_vt[rows_vt["variant_type"] == vt]
                for metric in ("macro_f1", "macro_recall"):
                    if metric in rows_vt.columns:
                        shared_values.extend(pd.to_numeric(rows_vt[metric], errors="coerce").dropna().tolist())
        if shared_values:
            shared_axis_spec = score_axis_spec(shared_values)
    for scope in ["illumina_view", "ont_view"]:
        baseline = VIEW_INFO[scope]["dv"] if VIEW_INFO[scope]["dv"] in names_by_view[scope] else None
        title_base = title_es if language == "es" else title_en
        title = f"{dataset_title} - {view_label(scope, language)} - {title_base}"
        outputs.append(plot_variant_breakdown_single_view(
            df,
            scope=scope,
            model_order=names_by_view[scope],
            title=title,
            out_path=out_dir / f"{out_stem}_{scope}.png",
            language=language,
            baseline_name=baseline,
            labels=(labels if (baseline or labels == "value") else "none"),
            pct_decimals=pct_decimals,
            save_svg=save_svg,
            shared_axis_spec=shared_axis_spec,
        ))
    return outputs


def make_two_stage_figure(
    data_root: Path,
    out_dir: Path,
    *,
    language: str,
    save_svg: bool,
) -> pd.DataFrame:
    csv_path = data_root / "HG003_ch20_initial_test" / "hg003_two_stage_hybrid_ablation.csv"
    if not csv_path.exists():
        print(f"[AVISO] No existe el CSV de ablación two-stage: {csv_path}")
        return pd.DataFrame()

    df = clean_numeric(pd.read_csv(csv_path))
    model_col = "model_label" if "model_label" in df.columns else "architecture"
    if model_col not in df.columns:
        print(f"[AVISO] El CSV de ablación no tiene 'model_label' ni 'architecture': {csv_path}")
        return pd.DataFrame()
    if "variant_type" in df.columns:
        df = df[df["variant_type"] == "ALL"].copy()

    family_specs = {
        "simple": {
            "title": "Hybrid simple",
            "models": {
                "1-stage": "Hybrid simple direct 3c",
                "2-stage raw": "Hybrid simple 2-stage raw",
                "2-stage calib.": "Hybrid simple 2-stage 3c calib",
            },
        },
        "groupwise": {
            "title": "Hybrid groupwise",
            "models": {
                "1-stage": "Hybrid groupwise direct 3c",
                "2-stage raw": "Hybrid groupwise 2-stage raw",
                "2-stage calib.": "Hybrid groupwise 2-stage 3c calib",
            },
        },
    }
    stage_colors = {"1-stage": "#5F7FA3", "2-stage raw": "#D7A461", "2-stage calib.": "#7EAE91"}

    keep = [m for spec in family_specs.values() for m in spec["models"].values()]
    sub = df[df[model_col].isin(keep)].copy()
    if sub.empty:
        print("[AVISO] No se encontraron modelos esperados en la ablación two-stage.")
        return pd.DataFrame()

    rows = sub.drop_duplicates(model_col).set_index(model_col)
    score_metrics = ["macro_f1", "variant_recall"]
    count_metrics = ["variant_fp", "variant_fn"]
    score_values = []
    count_values = []
    for model_name in keep:
        if model_name in rows.index:
            score_values.extend([float(rows.loc[model_name, m]) for m in score_metrics if m in rows.columns])
            count_values.extend([float(rows.loc[model_name, m]) for m in count_metrics if m in rows.columns])

    score_axis = score_axis_spec(score_values)
    max_count = max(finite_values(count_values), default=1.0)

    out_path = out_dir / "fig_6_4_hg003_two_stage_ablation.png"
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2), sharey="row")
    fig.suptitle(t(language, "two_stage"), fontsize=14, y=1.03)
    manifest = []

    for col, family_key in enumerate(["simple", "groupwise"]):
        spec = family_specs[family_key]
        stage_to_model = spec["models"]
        stage_order = [s for s in ["1-stage", "2-stage raw", "2-stage calib."] if stage_to_model[s] in rows.index]

        ax_score = axes[0, col]
        x_score = np.arange(len(score_metrics))
        width = 0.23
        containers = []
        labels_by_container = []
        for i, stage in enumerate(stage_order):
            model_name = stage_to_model[stage]
            vals = [float(rows.loc[model_name, metric]) if metric in rows.columns else np.nan for metric in score_metrics]
            offsets = x_score + (i - (len(stage_order) - 1) / 2) * width
            container = ax_score.bar(
                offsets,
                vals,
                width=width,
                label=stage,
                color=stage_colors[stage],
                edgecolor=EDGE_COLOR,
                linewidth=0.55,
            )
            containers.append(container)
            labels_by_container.append([f"{v:.4f}" if np.isfinite(v) else "" for v in vals])
            for metric, value in zip(score_metrics, vals):
                manifest.append({"figure": out_path.stem, "family": family_key, "stage": stage, "model": model_name, "metric": metric, "value": value})
        ax_score.set_title(spec["title"])
        ax_score.set_xticks(x_score)
        ax_score.set_xticklabels([metric_label(m, language) for m in score_metrics])
        ax_score.set_ylabel(t(language, "score") if col == 0 else "")
        apply_score_axis(ax_score, score_axis)
        add_bar_labels(ax_score, containers, labels_by_container, fontsize=7, rotation=0)

        ax_count = axes[1, col]
        x_count = np.arange(len(count_metrics))
        count_containers = []
        count_labels = []
        for i, stage in enumerate(stage_order):
            model_name = stage_to_model[stage]
            vals = [float(rows.loc[model_name, metric]) if metric in rows.columns else np.nan for metric in count_metrics]
            offsets = x_count + (i - (len(stage_order) - 1) / 2) * width
            container = ax_count.bar(
                offsets,
                vals,
                width=width,
                label=stage,
                color=stage_colors[stage],
                edgecolor=EDGE_COLOR,
                linewidth=0.55,
            )
            count_containers.append(container)
            count_labels.append([f"{int(round(v))}" if np.isfinite(v) else "" for v in vals])
            for metric, value in zip(count_metrics, vals):
                manifest.append({"figure": out_path.stem, "family": family_key, "stage": stage, "model": model_name, "metric": metric, "value": value})
        ax_count.set_title(t(language, "variant_fp_fn"))
        ax_count.set_xticks(x_count)
        ax_count.set_xticklabels([metric_label(m, language) for m in count_metrics])
        ax_count.set_ylabel(t(language, "count") if col == 0 else "")
        ax_count.set_ylim(0, max_count * 1.18 if max_count > 0 else 1)
        ax_count.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
        add_bar_labels(ax_count, count_containers, count_labels, fontsize=8, rotation=0)

    legend_order = ["1-stage", "2-stage raw", "2-stage calib."]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=stage_colors[s], edgecolor=EDGE_COLOR, linewidth=0.55) for s in legend_order]
    fig.legend(handles, legend_order, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, out_path, save_svg=save_svg)
    return pd.DataFrame(manifest)


def make_coverage_overview_figures(
    coverage_dfs: Mapping[str, pd.DataFrame],
    out_dir: Path,
    *,
    language: str,
    save_svg: bool,
) -> list[pd.DataFrame]:
    outputs = []
    x = np.arange(len(COVERAGES))

    for scope in ["illumina_view", "ont_view"]:
        view = VIEW_INFO[scope]
        names = [view["dv"], view["unimodal"], "hybrid_simple", "hybrid_groupwise"]
        series = {}
        all_values = []
        manifest = []
        for name in names:
            y = []
            for cov in COVERAGES:
                df = coverage_dfs.get(cov)
                if df is None:
                    value = np.nan
                else:
                    rows = filter_rows(df, scope, [name], "ALL")
                    value = float(rows.iloc[0]["macro_f1"]) if not rows.empty and "macro_f1" in rows.columns else np.nan
                y.append(value)
                all_values.append(value)
                manifest.append({"figure": f"fig_6_9_hg005_coverage_overview_macro_f1_{scope}", "scope": scope, "coverage": cov, "model": name, "metric": "macro_f1", "value": value})
            series[name] = y

        if not finite_values(all_values):
            print(f"[AVISO] Sin datos para cobertura {scope}")
            continue

        axis_spec = score_axis_spec(all_values)
        out_path = out_dir / f"fig_6_9_hg005_coverage_overview_macro_f1_{scope}.png"
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for name, y in series.items():
            ax.plot(x, y, marker="o", linewidth=2.0, markersize=5, label=friendly_model_name(name, language), color=color_for_model(name))
        ax.set_title(f"HG005 chr20+chr21 - {view_label(scope, language)} - {t(language, 'coverage_overview')}")
        ax.set_xticks(x)
        ax.set_xticklabels(COVERAGES)
        ax.set_xlabel(t(language, "coverage"))
        ax.set_ylabel(t(language, "macro_f1"))
        apply_score_axis(ax, axis_spec)
        external_legend(ax, title=t(language, "model"), language=language)
        fig.tight_layout()
        savefig(fig, out_path, save_svg=save_svg)
        outputs.append(pd.DataFrame(manifest))
    return outputs


def make_coverage_snp_indel_figures(
    coverage_dfs: Mapping[str, pd.DataFrame],
    out_dir: Path,
    *,
    language: str,
    pct_decimals: int,
    save_svg: bool,
) -> list[pd.DataFrame]:
    outputs = []
    hybrid_name = "hybrid_groupwise"
    coverage_gap = 0.68
    bar_width = 0.34
    coverage_centers = np.arange(len(COVERAGES)) * (2.0 + coverage_gap)
    block_offsets = {"SNP": -0.42, "INDEL": 0.42}
    # Keep DV on the right, consistent with the original Chapter 6 visual convention.
    model_offsets = {"hybrid": -bar_width * 0.58, "dv": bar_width * 0.58}

    for scope in ["illumina_view", "ont_view"]:
        dv_name = VIEW_INFO[scope]["dv"]
        values = {}
        all_values = []
        for cov in COVERAGES:
            df = coverage_dfs.get(cov)
            for vt in VARIANT_TYPES:
                for role, model_name in [("dv", dv_name), ("hybrid", hybrid_name)]:
                    if df is None:
                        value = np.nan
                    else:
                        rows = df[(df["scope"] == f"{scope}_{vt}") & (df["name"] == model_name)]
                        if "variant_type" in rows.columns:
                            rows = rows[rows["variant_type"] == vt]
                        value = float(rows.iloc[0]["macro_f1"]) if not rows.empty and "macro_f1" in rows.columns else np.nan
                    values[(cov, vt, role)] = value
                    all_values.append(value)

        if not finite_values(all_values):
            print(f"[AVISO] Sin datos para cobertura SNP/INDEL {scope}")
            continue

        axis_spec = score_axis_spec(all_values, n_ticks_below_one=3)
        out_path = out_dir / f"fig_6_10_hg005_coverage_snp_indel_groupwise_vs_dv_{scope}.png"
        fig, ax = plt.subplots(figsize=(11.8, 5.9))
        dv_positions, dv_values = [], []
        hyb_positions, hyb_values, hyb_labels = [], [], []
        dv_labels = []
        manifest = []

        for cov_idx, cov in enumerate(COVERAGES):
            cov_center = coverage_centers[cov_idx]
            for vt in VARIANT_TYPES:
                block_center = cov_center + block_offsets[vt]
                dv_value = values[(cov, vt, "dv")]
                hyb_value = values[(cov, vt, "hybrid")]
                dv_positions.append(block_center + model_offsets["dv"])
                dv_values.append(dv_value)
                dv_labels.append("0%")
                hyb_positions.append(block_center + model_offsets["hybrid"])
                hyb_values.append(hyb_value)
                hyb_labels.append(format_pct(rel_pct_change(hyb_value, dv_value), pct_decimals))
                manifest.append({"figure": out_path.stem, "scope": scope, "coverage": cov, "variant_type": vt, "model": dv_name, "metric": "macro_f1", "value": dv_value, "baseline_model": dv_name, "baseline_value": dv_value, "relative_change_pct": 0.0})
                manifest.append({"figure": out_path.stem, "scope": scope, "coverage": cov, "variant_type": vt, "model": hybrid_name, "metric": "macro_f1", "value": hyb_value, "baseline_model": dv_name, "baseline_value": dv_value, "relative_change_pct": rel_pct_change(hyb_value, dv_value)})

        dv_container = ax.bar(dv_positions, dv_values, width=bar_width, label=friendly_model_name(dv_name, language), color=color_for_model(dv_name), edgecolor=EDGE_COLOR, linewidth=0.55)
        hyb_container = ax.bar(hyb_positions, hyb_values, width=bar_width, label=friendly_model_name(hybrid_name, language), color=color_for_model(hybrid_name), edgecolor=EDGE_COLOR, linewidth=0.55)

        for cov_idx in range(len(COVERAGES) - 1):
            boundary = (coverage_centers[cov_idx] + coverage_centers[cov_idx + 1]) / 2
            ax.axvline(boundary, color="0.86", linewidth=0.8, zorder=0)

        bottom, top, _, _ = axis_spec
        variant_label_y = bottom - (top - bottom) * 0.055
        for cov_center in coverage_centers:
            for vt in VARIANT_TYPES:
                ax.text(cov_center + block_offsets[vt], variant_label_y, vt, ha="center", va="top", fontsize=8, clip_on=False)

        group_half_width = max(abs(block_offsets[vt]) + abs(model_offsets[m]) + bar_width / 2 for vt in VARIANT_TYPES for m in ["hybrid", "dv"])
        coverage_step = coverage_centers[1] - coverage_centers[0]
        internal_gap = coverage_step - 2 * group_half_width
        edge_gap = internal_gap * 0.5
        ax.set_xlim(coverage_centers[0] - group_half_width - edge_gap, coverage_centers[-1] + group_half_width + edge_gap)

        ax.set_title(f"HG005 chr20+chr21 - {view_label(scope, language)} - {t(language, 'coverage_variant')}")
        ax.set_xticks(coverage_centers)
        ax.set_xticklabels(COVERAGES)
        ax.set_xlabel(t(language, "coverage"), labelpad=20)
        ax.set_ylabel(t(language, "f1"))
        apply_score_axis(ax, axis_spec)
        external_legend(ax, title=t(language, "model"), language=language)
        add_bar_labels(ax, [dv_container, hyb_container], [dv_labels, hyb_labels], fontsize=7, rotation=90)
        fig.tight_layout()
        savefig(fig, out_path, save_svg=save_svg)
        outputs.append(pd.DataFrame(manifest))
    return outputs


def make_precision_recall_scatter_figures(
    df: pd.DataFrame,
    *,
    dataset_title: str,
    names_by_view: Mapping[str, Sequence[str]],
    out_stem: str,
    out_dir: Path,
    language: str,
    save_svg: bool,
) -> list[pd.DataFrame]:
    outputs = []
    for scope in ["illumina_view", "ont_view"]:
        names = list(names_by_view[scope])
        sub = filter_rows(df, scope, names, "ALL")
        if sub.empty or "macro_precision" not in sub.columns or "macro_recall" not in sub.columns:
            print(f"[AVISO] Sin datos para scatter {out_stem}_{scope}")
            continue
        sub = sub.drop_duplicates("name")
        out_path = out_dir / f"{out_stem}_{scope}.png"
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        xs, ys = [], []
        manifest = []
        for _, row in sub.iterrows():
            name = row["name"]
            x = float(row["macro_recall"])
            y = float(row["macro_precision"])
            xs.append(x)
            ys.append(y)
            ax.scatter(
                x,
                y,
                s=92,
                color=color_for_model(name),
                edgecolor=EDGE_COLOR,
                linewidth=0.65,
                label=friendly_model_name(name, language),
                zorder=3,
            )
            manifest.append({"figure": out_path.stem, "scope": scope, "model": name, "metric": "macro_precision", "value": y})
            manifest.append({"figure": out_path.stem, "scope": scope, "model": name, "metric": "macro_recall", "value": x})
        set_zoom_axis_from_xy(ax, xs, ys)
        ax.set_title(f"{dataset_title} - {view_label(scope, language)} - {t(language, 'scatter_title')}")
        ax.set_xlabel(t(language, "scatter_xlabel"))
        ax.set_ylabel(t(language, "scatter_ylabel"))
        external_legend(ax, title=t(language, "model"), language=language)
        fig.tight_layout()
        savefig(fig, out_path, save_svg=save_svg)
        outputs.append(pd.DataFrame(manifest))
    return outputs


# -----------------------------------------------------------------------------
# Top-level generation
# -----------------------------------------------------------------------------


def make_all_figures(
    data_root: Path,
    out_dir: Path,
    *,
    language: str,
    save_svg: bool = False,
    pct_decimals: int = 2,
) -> pd.DataFrame:
    ensure_outdir(out_dir)
    configure_matplotlib()
    manifest_parts: list[pd.DataFrame] = []

    hg003 = read_summary(data_root, DATASETS["HG003"])
    hg004 = read_summary(data_root, DATASETS["HG004"])
    hg005 = read_summary(data_root, DATASETS["HG005"])
    hg005_coverage = {cov: read_summary(data_root, spec) for cov, spec in HG005_COVERAGE_DATASETS.items()}

    # HG003: baseline vs unimodal heads.
    if hg003 is not None:
        manifest_parts.extend(make_standard_view_figures(
            hg003,
            dataset_title=DATASETS["HG003"].title(language),
            names_by_view={
                "illumina_view": ["dv_illumina", "unimodal_illumina_linear", "unimodal_illumina_mlp"],
                "ont_view": ["dv_ont", "unimodal_ont_linear", "unimodal_ont_mlp"],
            },
            title_es="DV baseline vs cabezas unimodales",
            title_en="DV baseline vs unimodal heads",
            out_stem="fig_6_1_hg003_unimodal_vs_dv",
            out_dir=out_dir,
            language=language,
            pct_decimals=pct_decimals,
            save_svg=save_svg,
        ))

        # HG003: MLP-only hybrid comparison.
        manifest_parts.extend(make_standard_view_figures(
            hg003,
            dataset_title=DATASETS["HG003"].title(language),
            names_by_view={
                "illumina_view": ["dv_illumina", "unimodal_illumina_mlp", "hybrid_simple_mlp", "hybrid_groupwise_mlp"],
                "ont_view": ["dv_ont", "unimodal_ont_mlp", "hybrid_simple_mlp", "hybrid_groupwise_mlp"],
            },
            title_es="modelos MLP unimodales e híbridos",
            title_en="MLP unimodal and hybrid models",
            out_stem="fig_6_2_hg003_hybrid_mlp",
            out_dir=out_dir,
            language=language,
            pct_decimals=pct_decimals,
            save_svg=save_svg,
        ))

        # HG003: variant type breakdown for hybrid MLPs. No percentage labels by default
        # because there is no DV baseline in this specific comparison and labels crowd ONT/INDEL.
        manifest_parts.extend(make_variant_view_figures(
            hg003,
            dataset_title=DATASETS["HG003"].title(language),
            names_by_view={
                "illumina_view": ["hybrid_simple_mlp", "hybrid_groupwise_mlp"],
                "ont_view": ["hybrid_simple_mlp", "hybrid_groupwise_mlp"],
            },
            title_es="modelos híbridos por tipo de variante",
            title_en="hybrid models by variant type",
            out_stem="fig_6_3_hg003_hybrid_snp_indel",
            out_dir=out_dir,
            language=language,
            pct_decimals=pct_decimals,
            save_svg=save_svg,
            labels="value",
            shared_y_axis=True,
        ))

    manifest_parts.append(make_two_stage_figure(data_root, out_dir, language=language, save_svg=save_svg))

    for dataset_key, df, prefix, overall_num, variant_num in [
        ("HG004", hg004, "hg004", "5", "6"),
        ("HG005", hg005, "hg005", "7", "8"),
    ]:
        if df is None:
            continue
        spec = DATASETS[dataset_key]
        names_by_view = {
            "illumina_view": ["dv_illumina", "unimodal_illumina", "hybrid_simple", "hybrid_groupwise"],
            "ont_view": ["dv_ont", "unimodal_ont", "hybrid_simple", "hybrid_groupwise"],
        }
        manifest_parts.extend(make_standard_view_figures(
            df,
            dataset_title=spec.title(language),
            names_by_view=names_by_view,
            title_es="DV, unimodal e híbridos",
            title_en="DV, unimodal and hybrid models",
            out_stem=f"fig_6_{overall_num}_{prefix}_overall",
            out_dir=out_dir,
            language=language,
            pct_decimals=pct_decimals,
            save_svg=save_svg,
            shared_y_axis=True,
        ))
        manifest_parts.extend(make_variant_view_figures(
            df,
            dataset_title=spec.title(language),
            names_by_view={
                "illumina_view": ["dv_illumina", "unimodal_illumina", "hybrid_simple"],
                "ont_view": ["dv_ont", "unimodal_ont", "hybrid_simple"],
            },
            title_es="desglose SNP/INDEL para DV, unimodal e híbrido simple",
            title_en="SNP/INDEL breakdown for DV, unimodal and hybrid simple",
            out_stem=f"fig_6_{variant_num}_{prefix}_snp_indel",
            out_dir=out_dir,
            language=language,
            pct_decimals=pct_decimals,
            save_svg=save_svg,
            labels="pct_vs_baseline",
        ))
    if any(df is not None for df in hg005_coverage.values()):
        manifest_parts.extend(make_coverage_overview_figures(hg005_coverage, out_dir, language=language, save_svg=save_svg))
        manifest_parts.extend(make_coverage_snp_indel_figures(hg005_coverage, out_dir, language=language, pct_decimals=pct_decimals, save_svg=save_svg))

    manifest_parts = [part for part in manifest_parts if part is not None and not part.empty]
    if manifest_parts:
        manifest = pd.concat(manifest_parts, ignore_index=True)
    else:
        manifest = pd.DataFrame()
    manifest.to_csv(out_dir / "figure_values_manifest.csv", index=False)
    print(f"[OK] Manifest guardado en: {out_dir / 'figure_values_manifest.csv'}")
    print(f"[OK] Figuras PNG generadas: {len(list(out_dir.glob('*.png')))}")
    if save_svg:
        print(f"[OK] Figuras SVG generadas: {len(list(out_dir.glob('*.svg')))}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate refined Variant Calling figures.")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Root folder containing the CSV report subfolders. Default: current folder.")
    parser.add_argument("--out-dir", type=Path, default=Path("plots_refined"), help="Output folder for generated figures.")
    parser.add_argument("--language", "--lang", choices=["es", "en"], default="es", help="Figure language: 'es' or 'en'. Default: es.")
    parser.add_argument("--save-svg", action="store_true", help="Also save editable SVG files. PNG files are always generated.")
    parser.add_argument("--pct-decimals", type=int, default=2, choices=[0, 1, 2, 3], help="Decimals for percentage labels vs DV baseline. Default: 2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    make_all_figures(
        data_root,
        out_dir,
        language=args.language,
        save_svg=args.save_svg,
        pct_decimals=args.pct_decimals,
    )


if __name__ == "__main__":
    main()
