#!/usr/bin/env python3
"""
Este script permite generar un plot que muestra la cobertura (depth) disponible a lo largo de un BAM. Sirve para dar información más detallada que el summary que saca mosdepth habitualmente. 
Se lanza como: 
    python  python3 data/4_out/mosdepth/plot_coverage.py --folder "Illumina/ONT_FULL/ONT_PARTIAL"

Previo a lanzar este script, se ha tenido que hacer el análisis de cobertura por ventanas usando mosdepth:

conda activate hts
cd ~/VariantCalling

THREADS=4
WIN=10000
OUT_BASE="data/4_out/mosdepth/HG003/chr20/Illumina"
mkdir -p "$OUT_BASE"

BAM="data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam"

mosdepth -t "$THREADS" -n -x --by "$WIN" \
  "$OUT_BASE/HG003.Illumina.chr20.win${WIN}" \
  "$BAM"

Esto generará un archivo .bed.gz y un archivo .bed.gz.csi, ambos necesarios para estre script
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def solve_name_with_folder(folder: str) -> str:
    if folder == "Illumina":
        return folder
    elif folder == "ONT_FULL":
        return "ONT.full"
    else:
        return "ONT.partial"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="One of: Illumina, ONT_FULL, ONT_PARTIAL (or anything else -> ONT.partial)")
    args = parser.parse_args()

    FOLDER = args.folder
    name = solve_name_with_folder(FOLDER)

    path = f"data/4_out/mosdepth/HG003/chr20/{FOLDER}/HG003.{name}.chr20.win10000.regions.bed.gz"
    df = pd.read_csv(path, sep="\t", header=None, compression="gzip")

    # Formato típico: chrom, start, end, mean_depth
    df.columns = ["chrom", "start", "end", "depth"]
    df = df[df["chrom"] == "chr20"].copy()

    df["mid_mb"] = (df["start"] + df["end"]) / 2 / 1e6  # posición en Mb
    df["depth_smooth"] = df["depth"].rolling(10, center=True, min_periods=1).mean()

    plt.figure(figsize=(12, 4))
    # plt.plot(df["mid_mb"], df["depth"], linewidth=1)
    plt.plot(df["mid_mb"], df["depth_smooth"], linewidth=2)
    plt.xlabel("chr20 position (Mb)")
    plt.ylabel("Mean depth (X)")
    plt.title(f"Coverage along chr20 (10kb windows) — {name}")
    plt.tight_layout()

    out_png = f"data/4_out/mosdepth/HG003/chr20/{FOLDER}/coverage_{name}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[OK] Saved plot: {out_png}")


if __name__ == "__main__":
    main()