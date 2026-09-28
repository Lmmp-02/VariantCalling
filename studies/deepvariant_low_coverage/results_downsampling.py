#!/usr/bin/env python3
"""
Collects hap.py benchmark results from all (tech, coverage) combinations,
generates comparison plots, and saves a summary table to CSV

Expected input structure:
    data/4_out/happy/HG003/{tech}/{region}/{run_tag}/happy.summary.csv

Where run_tag follows the pattern: cov{N}x_s{seed} or covorig_sorig

Usage (from project root):
    python data/5_scripts/plot_results.py
    python data/5_scripts/plot_results.py --sample HG003 --region chr20
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HAPPY_BASE = Path("data/4_out/happy")
PLOT_DIR   = Path("data/4_out/plots")

TECH_LABELS = {
    "Illumina":              "Illumina",
    "ONT_R104_sup_PAY87794": "ONT R104",
}

# Coverage order for x-axis (orig = full coverage)
COV_ORDER = [5, 10, 15, 30, "orig"]
COV_LABELS = {5: "5x", 10: "10x", 15: "15x", 30: "30x", "orig": "Full"}

METRICS = ["METRIC.Recall", "METRIC.Precision", "METRIC.F1_Score"]
METRIC_LABELS = {
    "METRIC.Recall":    "Recall",
    "METRIC.Precision": "Precision",
    "METRIC.F1_Score":  "F1 Score",
}

TECH_COLORS = {
    "Illumina": "#2563EB",               # blue
    "ONT_R104_sup_PAY87794": "#DC2626",  # red
}
MARKER_STYLE = {"Illumina": "o", "ONT_R104_sup_PAY87794": "s"}

VARIANT_TYPES = ["SNP", "INDEL"]

# ---------------------------------------------------------------------------
# Additional functions
# ---------------------------------------------------------------------------

def parse_run_tag(run_tag: str) -> tuple:
    """Parse run_tag like 'cov30x_s42' or 'covorig_sorig' → (coverage, seed)"""
    m = re.match(r"cov(\d+)x_s(\d+|orig)", run_tag)
    if m:
        return int(m.group(1)), m.group(2)
    m = re.match(r"covorig_sorig", run_tag)
    if m:
        return "orig", "orig"
    return None, None


def load_results(sample: str, region: str) -> pd.DataFrame:
    """Walk the happy output tree and load all summary CSVs into a DataFrame"""
    base = HAPPY_BASE / sample
    records = []

    for tech_dir in sorted(base.iterdir()):
        if not tech_dir.is_dir():
            continue
        tech = tech_dir.name
        region_dir = tech_dir / region
        if not region_dir.is_dir():
            continue

        for run_dir in sorted(region_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            csv_path = run_dir / "happy.summary.csv"
            if not csv_path.exists():
                continue

            cov, seed = parse_run_tag(run_dir.name)
            if cov is None:
                print(f" [WARN] Could not parse run_tag: {run_dir.name}, skipping")
                continue

            df = pd.read_csv(csv_path)
            # Keep only PASS filter rows
            df = df[df["Filter"] == "PASS"].copy()
            df["tech"]     = tech
            df["coverage"] = cov
            df["seed"]     = seed
            df["run_tag"]  = run_dir.name
            records.append(df)

    if not records:
        print(f"ERROR: No happy.summary.csv files found in {base}")
        sys.exit(1)

    result = pd.concat(records, ignore_index=True)
    print(f"  Loaded {len(records)} result(s) from {len(result['tech'].unique())} tech(s)")
    return result


def cov_sort_key(cov):
    return 9999 if cov == "orig" else int(cov)


# ---------------------------------------------------------------------------
# Plot 1: Line plots — Recall / Precision / F1 vs Coverage
# ---------------------------------------------------------------------------

def plot_metrics_vs_coverage(df: pd.DataFrame, out_dir: Path):
    """One figure per metric, two subplots (SNP / INDEL), one line per tech"""
    techs = sorted(df["tech"].unique())
    covs_present = sorted(df["coverage"].unique(), key=cov_sort_key)

    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        fig.suptitle(METRIC_LABELS[metric], fontsize=15, fontweight="bold", y=1.01)

        for ax, vtype in zip(axes, VARIANT_TYPES):
            sub = df[df["Type"] == vtype]

            for tech in techs:
                t = sub[sub["tech"] == tech].copy()
                t = t.sort_values("coverage", key=lambda s: s.map(cov_sort_key))

                x_labels = [COV_LABELS.get(c, str(c)) for c in t["coverage"]]
                y_vals   = t[metric].values

                ax.plot(
                    x_labels, y_vals,
                    marker=MARKER_STYLE.get(tech, "o"),
                    color=TECH_COLORS.get(tech, "gray"),
                    linewidth=2,
                    markersize=7,
                    label=TECH_LABELS.get(tech, tech),
                )

            ax.set_title(vtype, fontsize=12, fontweight="bold")
            ax.set_xlabel("Coverage", fontsize=10)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)
            ax.tick_params(axis="x", rotation=0)

        plt.tight_layout()
        fname = out_dir / f"metric_{metric.replace('METRIC.', '').lower()}_vs_coverage.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 2: Grouped bar chart — TP / FN / FP per combination
# ---------------------------------------------------------------------------

def plot_tp_fn_fp(df: pd.DataFrame, out_dir: Path):
    """Grouped bar chart of TP, FN, FP for every (tech, coverage) combination"""
    for vtype in VARIANT_TYPES:
        sub = df[df["Type"] == vtype].copy()
        sub = sub.sort_values(
            ["tech", "coverage"],
            key=lambda s: s.map(cov_sort_key) if s.name == "coverage" else s
        )

        labels = [
            f"{TECH_LABELS.get(r.tech, r.tech)}\n{COV_LABELS.get(r.coverage, r.coverage)}"
            for _, r in sub.iterrows()
        ]
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5))

        bars_tp = ax.bar(x - width, sub["TRUTH.TP"],  width, label="TP", color="#16A34A", alpha=0.85)
        bars_fn = ax.bar(x,         sub["TRUTH.FN"],  width, label="FN", color="#F59E0B", alpha=0.85)
        bars_fp = ax.bar(x + width, sub["QUERY.FP"],  width, label="FP", color="#EF4444", alpha=0.85)

        ax.set_title(f"{vtype} — True Positives / False Negatives / False Positives",
                     fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

        plt.tight_layout()
        fname = out_dir / f"tp_fn_fp_{vtype.lower()}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 3: Heatmap — F1 score matrix (tech × coverage)
# ---------------------------------------------------------------------------

def plot_f1_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap of F1 scores with tech on y-axis and coverage on x-axis"""
    techs = sorted(df["tech"].unique())
    covs  = sorted(df["coverage"].unique(), key=cov_sort_key)

    for vtype in VARIANT_TYPES:
        sub = df[df["Type"] == vtype]
        matrix = pd.DataFrame(index=techs, columns=covs, dtype=float)

        for tech in techs:
            for cov in covs:
                val = sub[(sub["tech"] == tech) & (sub["coverage"] == cov)]["METRIC.F1_Score"]
                matrix.loc[tech, cov] = val.values[0] if len(val) else np.nan

        fig, ax = plt.subplots(figsize=(len(covs) * 1.4 + 1, len(techs) * 1.2 + 1))
        im = ax.imshow(matrix.values.astype(float), aspect="auto",
                       cmap="RdYlGn", vmin=0.95, vmax=1.0)

        ax.set_xticks(range(len(covs)))
        ax.set_xticklabels([COV_LABELS.get(c, str(c)) for c in covs], fontsize=11)
        ax.set_yticks(range(len(techs)))
        ax.set_yticklabels([TECH_LABELS.get(t, t) for t in techs], fontsize=11)
        ax.set_title(f"{vtype} — F1 Score Heatmap", fontsize=13, fontweight="bold")

        for i, tech in enumerate(techs):
            for j, cov in enumerate(covs):
                val = matrix.loc[tech, cov]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                            fontsize=9, color="black", fontweight="bold")

        plt.colorbar(im, ax=ax, label="F1 Score")
        plt.tight_layout()
        fname = out_dir / f"f1_heatmap_{vtype.lower()}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 4: Combined figure — Accuracy / Precision / Recall / F1 (2×2 grid)
# ---------------------------------------------------------------------------

def plot_four_metrics(df: pd.DataFrame, out_dir: Path):
    """
    Single figure with a 2×2 grid showing Accuracy, Precision, Recall and F1
    vs coverage for each variant type (SNP / INDEL), one line per tech

    Accuracy is computed as TP / (TP + FN + FP)
    """
    techs = sorted(df["tech"].unique())

    # Calculate accuracy and attach it to the dataframe
    df = df.copy()
    df["METRIC.Accuracy"] = df["TRUTH.TP"] / (
        df["TRUTH.TP"] + df["TRUTH.FN"] + df["QUERY.FP"]
    )

    all_metrics = [
        ("METRIC.Accuracy",   "Accuracy"),
        ("METRIC.Precision",  "Precision"),
        ("METRIC.Recall",     "Recall"),
        ("METRIC.F1_Score",   "F1 Score"),
    ]

    for vtype in VARIANT_TYPES:
        sub = df[df["Type"] == vtype]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Benchmark metrics vs Coverage — {vtype}",
            fontsize=15, fontweight="bold", y=1.01,
        )

        for ax, (metric_col, metric_name) in zip(axes.flat, all_metrics):
            for tech in techs:
                t = sub[sub["tech"] == tech].copy()
                t = t.sort_values("coverage", key=lambda s: s.map(cov_sort_key))

                x_labels = [COV_LABELS.get(c, str(c)) for c in t["coverage"]]
                y_vals   = t[metric_col].values

                ax.plot(
                    x_labels, y_vals,
                    marker=MARKER_STYLE.get(tech, "o"),
                    color=TECH_COLORS.get(tech, "gray"),
                    linewidth=2,
                    markersize=7,
                    label=TECH_LABELS.get(tech, tech),
                )

            ax.set_title(metric_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Coverage", fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)
            ax.tick_params(axis="x", rotation=0)

        plt.tight_layout()
        fname = out_dir / f"four_metrics_{vtype.lower()}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Summary table → CSV
# ---------------------------------------------------------------------------

def save_summary_csv(df: pd.DataFrame, out_dir: Path):
    """Save the summary table to a CSV file"""
    cols = ["tech", "coverage", "Type", "METRIC.Recall", "METRIC.Precision", "METRIC.F1_Score",
            "TRUTH.TP", "TRUTH.FN", "QUERY.FP"]
    sub = df[df["Type"].isin(VARIANT_TYPES)][cols].copy()
    sub = sub.sort_values(
        ["tech", "Type", "coverage"],
        key=lambda s: s.map(cov_sort_key) if s.name == "coverage" else s
    )

    sub["coverage"] = sub["coverage"].map(lambda c: COV_LABELS.get(c, str(c)))
    sub["tech"]     = sub["tech"].map(lambda t: TECH_LABELS.get(t, t))
    sub.rename(columns={
        "tech": "Tech", "coverage": "Coverage", "Type": "Variant",
        "METRIC.Recall": "Recall", "METRIC.Precision": "Precision",
        "METRIC.F1_Score": "F1",
        "TRUTH.TP": "TP", "TRUTH.FN": "FN", "QUERY.FP": "FP",
    }, inplace=True)

    csv_path = out_dir / "benchmark_summary.csv"
    sub.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved summary table: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot DeepVariant + hap.py benchmark results")
    parser.add_argument("--sample", default="HG003")
    parser.add_argument("--region", default="chr20")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results for sample={args.sample}, region={args.region}...")
    df = load_results(args.sample, args.region)

    print("\nGenerating plots...")
    plot_metrics_vs_coverage(df, PLOT_DIR)
    plot_tp_fn_fp(df, PLOT_DIR)
    plot_f1_heatmap(df, PLOT_DIR)
    plot_four_metrics(df, PLOT_DIR)

    print("\nSaving summary CSV...")
    save_summary_csv(df, PLOT_DIR)

    print(f"\nAll outputs saved to: {PLOT_DIR}/\n")


if __name__ == "__main__":
    main()