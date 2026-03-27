#!/usr/bin/env python3
"""
Run mosdepth on the HG003 BAMs (Illumina and ONT), generate the .bed.gz files
and save a coverage plot in 10 kb windows for each one

Usage:
    conda activate hts
    cd ~/VariantCalling
    python .../plot_coverage.py
"""

import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

THREADS = 4
WIN = 10000

DATASETS = {
    "Illumina": {
        "bam":  "data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam",
        "name": "HG003.Illumina.chr20",
    },
    "ONT": {
        "bam":  "data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87794.chr20.bam",
        "name": "HG003.ONT.chr20",
    },
}

OUT_BASE = "data/4_out/mosdepth/HG003/chr20"


# ----------------------------------------------------------------------------
# Mosdepth execution
# ----------------------------------------------------------------------------

def run_mosdepth(label: str, bam: str, prefix: str) -> None:
    """Execute mosdepth with windows of WIN b. Skip if the .bed.gz file already exists"""
    
    bed_gz = f"{prefix}.regions.bed.gz"

    if os.path.isfile(bed_gz):
        print(f"[SKIP] {label}: .bed.gz already exists → {bed_gz}")
        return

    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    cmd = [
        "mosdepth",
        "-t", str(THREADS),
        "-n",            # without output base-level
        "-x",            # quick mode
        "--by", str(WIN),
        prefix,
        bam,
    ]
    print(f"[RUN] mosdepth {label}: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] mosdepth failed for {label} (code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[OK]  mosdepth {label} → {bed_gz}")


# ----------------------------------------------------------------------------
# Plot coverages
# ----------------------------------------------------------------------------

def plot_coverage(label: str, prefix: str) -> None:
    """Reads the .bed.gz file and saves a PNG image with the coverage along chr20."""
    bed_gz = f"{prefix}.regions.bed.gz"

    df = pd.read_csv(bed_gz, sep="\t", header=None, compression="gzip")
    df.columns = ["chrom", "start", "end", "depth"]
    df = df[df["chrom"] == "chr20"].copy()

    df["mid_mb"] = (df["start"] + df["end"]) / 2 / 1e6
    df["depth_smooth"] = df["depth"].rolling(10, center=True, min_periods=1).mean()

    mean_depth = df["depth"].mean()

    plt.figure(figsize=(12, 4))
    plt.plot(df["mid_mb"].to_numpy(), df["depth_smooth"].to_numpy(), linewidth=2, label="Smoothed coverage")
    plt.axhline(mean_depth, linestyle="--", linewidth=1.5, color="black", label=f"Mean depth = {mean_depth:.2f}x")

    plt.xlabel("chr20 position (Mb)")
    plt.ylabel("Mean depth (X)")
    plt.title(f"Coverage along chr20 ({WIN // 1000} kb windows) — {label}")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(prefix), f"coverage_{label}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[OK]  Plot saved: {out_png}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    for label, cfg in DATASETS.items():
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")

        prefix = os.path.join(OUT_BASE, label, cfg["name"] + f".win{WIN}")

        run_mosdepth(label, cfg["bam"], prefix)
        plot_coverage(label, prefix)

    print("\n[DONE] Processing completed for Illumina and ONT")


if __name__ == "__main__":
    main()