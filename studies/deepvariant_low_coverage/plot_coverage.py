#!/usr/bin/env python3
"""
Auto-discover regional BAMs under data/1_input_bams, run mosdepth for each one,
store outputs in data/4_out/mosdepth/{sample}/{region_tag}/{tech}/, generate
summary TSVs, and save one coverage plot per BAM if matplotlib is available.

Examples:
    conda activate hts
    cd ~/VariantCalling
    python data/5_scripts/plot_coverage_v2.py

Optional environment variables:
    THREADS=8 WIN=10000 SMOOTH=10 FORCE=0 python data/5_scripts/plot_coverage_v2.py
"""

from __future__ import annotations

import csv
import gzip
import os
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------

HAS_MATPLOTLIB = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path("data/1_input_bams")
OUT_BASE = Path("data/4_out/mosdepth")

THREADS = int(os.environ.get("THREADS", "4"))
WIN = int(os.environ.get("WIN", "10000"))
SMOOTH = int(os.environ.get("SMOOTH", "10"))
FORCE = os.environ.get("FORCE", "0") == "1"

REPORT = OUT_BASE / "coverage_report.tsv"
COMPACT = OUT_BASE / "coverage_report_compact.tsv"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

def sanitize_colname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)

def quickcheck_bam(bam: Path) -> bool:
    result = subprocess.run(
        ["samtools", "quickcheck", str(bam)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0

def discover_bams(root: Path) -> List[Dict[str, str]]:
    bams: List[Dict[str, str]] = []
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root}")

    for bam in sorted(root.rglob("*.bam")):
        rel = bam.relative_to(root)
        parts = rel.parts
        if len(parts) < 3:
            log(f"[WARN] Skipping BAM with unexpected path layout: {bam}")
            continue

        sample = parts[0]
        tech = parts[1]
        stem = bam.stem
        region_tag = stem.rsplit(".", 1)[-1] if "." in stem else "unknown_region"

        bams.append({
            "sample": sample,
            "tech": tech,
            "bam": str(bam),
            "stem": stem,
            "region_tag": region_tag,
        })

    return bams

def mosdepth_prefix(sample: str, tech: str, region_tag: str, stem: str) -> Path:
    out_dir = OUT_BASE / sample / region_tag / tech
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.win{WIN}"

def run_mosdepth(label: str, bam: Path, prefix: Path) -> None:
    regions_bed_gz = Path(f"{prefix}.regions.bed.gz")
    summary_txt = Path(f"{prefix}.mosdepth.summary.txt")

    if not FORCE and regions_bed_gz.exists() and summary_txt.exists():
        log(f"[SKIP] mosdepth already exists for {label} -> {regions_bed_gz}")
        return

    cmd = [
        "mosdepth",
        "-t", str(THREADS),
        "-n",
        "--by", str(WIN),
        str(prefix),
        str(bam),
    ]
    log(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"mosdepth failed for {bam} (code {result.returncode})")

def read_summary(summary_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(summary_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            chrom = row["chrom"]
            if chrom == "*":
                continue
            if chrom != "total":
                try:
                    bases = float(row["bases"])
                except Exception:
                    bases = 0.0
                if bases <= 0:
                    continue
            rows.append(row)
    return rows

def rolling_mean(values: List[float], k: int) -> List[float]:
    if not values:
        return []
    if k <= 1:
        return list(values)

    out: List[float] = []
    n = len(values)
    half = k // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = values[lo:hi]
        out.append(sum(window) / len(window))
    return out

def read_regions_bed_gz(regions_bed_gz: Path) -> Dict[str, List[Tuple[int, int, float]]]:
    by_chrom: Dict[str, List[Tuple[int, int, float]]] = OrderedDict()
    with gzip.open(regions_bed_gz, "rt", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            chrom, start, end, depth = row[0], int(row[1]), int(row[2]), float(row[3])
            by_chrom.setdefault(chrom, []).append((start, end, depth))
    return by_chrom

def plot_coverage(
    sample: str,
    tech: str,
    region_tag: str,
    stem: str,
    prefix: Path,
    chroms_with_coverage: List[str],
) -> None:
    if not HAS_MATPLOTLIB:
        log("[WARN] matplotlib not available, skipping plots")
        return

    regions_bed_gz = Path(f"{prefix}.regions.bed.gz")
    if not regions_bed_gz.exists():
        log(f"[WARN] Missing {regions_bed_gz}, cannot plot")
        return

    by_chrom = read_regions_bed_gz(regions_bed_gz)
    if not by_chrom:
        log(f"[WARN] No coverage windows found in {regions_bed_gz}")
        return

    # Plot only chromosomes with actual coverage according to mosdepth summary
    chroms = [c for c in chroms_with_coverage if c in by_chrom]
    if not chroms:
        log(f"[WARN] No chromosomes with coverage found for plotting in {regions_bed_gz}")
        return

    chroms = sorted(chroms, key=natural_key)

    fig, axes = plt.subplots(
        nrows=len(chroms),
        ncols=1,
        figsize=(12, max(3.5, 3.2 * len(chroms))),
        squeeze=False,
    )

    for ax, chrom in zip(axes.flatten(), chroms):
        rows = by_chrom[chrom]
        mids_mb = [((start + end) / 2) / 1e6 for start, end, _ in rows]
        depth = [d for _, _, d in rows]
        depth_smooth = rolling_mean(depth, SMOOTH)
        mean_depth = sum(depth) / len(depth) if depth else 0.0

        ax.plot(mids_mb, depth_smooth, linewidth=1.5, label="Smoothed coverage")
        ax.axhline(
            mean_depth,
            linestyle="--",
            linewidth=1.2,
            color="black",
            label=f"Mean depth = {mean_depth:.2f}x",
        )
        ax.set_title(chrom)
        ax.set_xlabel(f"{chrom} position (Mb)")
        ax.set_ylabel("Depth (X)")
        ax.legend(loc="best")

    plt.suptitle(f"Coverage ({WIN // 1000} kb windows) — {sample} {tech} {region_tag}", y=0.995)
    plt.tight_layout()

    out_png = prefix.parent / f"coverage_{stem}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log(f"[OK] Plot saved: {out_png}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    bams = discover_bams(ROOT)
    if not bams:
        log(f"[WARN] No BAMs found under {ROOT}")
        return

    log(f"[INFO] Discovered {len(bams)} BAM(s) under {ROOT}")

    report_rows: List[Dict[str, str]] = []

    for meta in bams:
        sample = meta["sample"]
        tech = meta["tech"]
        bam = Path(meta["bam"])
        stem = meta["stem"]
        region_tag = meta["region_tag"]

        label = f"{sample} | {tech} | {stem}"
        log("\n" + "=" * 80)
        log(f"[INFO] {label}")
        log("=" * 80)

        if not quickcheck_bam(bam):
            log(f"[WARN] BAM invalid or incomplete, skipping: {bam}")
            continue

        prefix = mosdepth_prefix(sample, tech, region_tag, stem)
        run_mosdepth(label, bam, prefix)

        summary_path = Path(f"{prefix}.mosdepth.summary.txt")
        if not summary_path.exists():
            raise FileNotFoundError(f"Expected mosdepth summary not found: {summary_path}")

        rows = read_summary(summary_path)

        chroms_with_coverage: List[str] = []
        for row in rows:
            if row["chrom"] != "total":
                chroms_with_coverage.append(row["chrom"])
            report_rows.append({
                "sample": sample,
                "tech": tech,
                "region_tag": region_tag,
                "bam": str(bam),
                "chrom": row["chrom"],
                "length": row["length"],
                "bases": row["bases"],
                "mean": row["mean"],
            })

        plot_coverage(sample, tech, region_tag, stem, prefix, chroms_with_coverage)

    # -----------------------------------------------------------------------
    # Write detailed report
    # -----------------------------------------------------------------------

    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["sample", "tech", "region_tag", "bam", "chrom", "length", "bases", "mean"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(report_rows)

    # -----------------------------------------------------------------------
    # Write compact report
    # -----------------------------------------------------------------------

    grouped: Dict[Tuple[str, str, str, str], Dict[str, str]] = OrderedDict()
    chroms_seen = []

    for r in report_rows:
        key = (r["sample"], r["tech"], r["region_tag"], r["bam"])
        grouped.setdefault(key, {})
        grouped[key][r["chrom"]] = r["mean"]
        if r["chrom"] != "total" and r["chrom"] not in chroms_seen:
            chroms_seen.append(r["chrom"])

    chroms_seen = sorted(chroms_seen, key=natural_key)

    compact_fields = ["sample", "tech", "region_tag", "bam"] + \
                     [f"mean_{sanitize_colname(c)}" for c in chroms_seen] + \
                     ["mean_total"]

    with open(COMPACT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(compact_fields)

        for (sample, tech, region_tag, bam), d in grouped.items():
            row = [sample, tech, region_tag, bam]
            for chrom in chroms_seen:
                row.append(d.get(chrom, ""))
            row.append(d.get("total", ""))
            writer.writerow(row)

    log(f"\n[OK] Report:  {REPORT}")
    log(f"[OK] Compact: {COMPACT}")
    log(f"[DONE] Processed BAMs from {ROOT}")

if __name__ == "__main__":
    main()