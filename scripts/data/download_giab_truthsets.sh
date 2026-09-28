#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Download full GIAB/NIST v4.2.1 small-variant truth sets (GRCh38)
# for selected GIAB samples into data/3_truth/<SAMPLE>/
#
# Run from repo root.
#
# Examples:
#   bash data/5_scripts/download_giab_truthsets.sh
#   bash data/5_scripts/download_giab_truthsets.sh --samples HG002,HG005
#   bash data/5_scripts/download_giab_truthsets.sh --samples HG005,HG006,HG007
#   bash data/5_scripts/download_giab_truthsets.sh --force 1
#
# Outputs per sample:
#   data/3_truth/<SAMPLE>/
#     - <SAMPLE>_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
#     - <SAMPLE>_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi
#     - sample-specific BED name (see manifest)
#     - source_manifest.json
#
# Notes:
# - This script downloads the full truth sets.
# - Region subsetting (e.g. chr20_chr21) is intentionally left as a separate step.
# - BED filenames are sample-specific: Ashkenazim Trio uses *_benchmark_noinconsistent.bed,
#   while the Chinese Trio uses *_benchmark.bed.

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat >&2 <<'USAGE'
Download full GIAB/NIST v4.2.1 GRCh38 truth sets for selected GIAB samples.

Optional:
  --samples LIST   Comma-separated subset
                   (default: HG002,HG003,HG004,HG005)
                   (also supports HG006,HG007)
  --out_root DIR   Output root (default: data/3_truth)
  --force 0|1      Re-download even if files exist (default: 0)
  -h, --help       Show this help
USAGE
}

SAMPLES="HG002,HG003,HG004,HG005"
OUT_ROOT="data/3_truth"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --samples) SAMPLES="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

command -v curl >/dev/null 2>&1 || die "curl not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

mkdir -p "$OUT_ROOT"

BASE_FTP="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release"

sample_base_url() {
  local s="$1"
  case "$s" in
    HG002) echo "$BASE_FTP/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38" ;;
    HG003) echo "$BASE_FTP/AshkenazimTrio/HG003_NA24149_father/NISTv4.2.1/GRCh38" ;;
    HG004) echo "$BASE_FTP/AshkenazimTrio/HG004_NA24143_mother/NISTv4.2.1/GRCh38" ;;
    HG005) echo "$BASE_FTP/ChineseTrio/HG005_NA24631_son/NISTv4.2.1/GRCh38" ;;
    HG006) echo "$BASE_FTP/ChineseTrio/HG006_NA24694_father/NISTv4.2.1/GRCh38" ;;
    HG007) echo "$BASE_FTP/ChineseTrio/HG007_NA24695_mother/NISTv4.2.1/GRCh38" ;;
    *) die "Unsupported sample: $s" ;;
  esac
}

sample_bed_name() {
  local s="$1"
  case "$s" in
    HG002|HG003|HG004)
      echo "${s}_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"
      ;;
    HG005|HG006|HG007)
      echo "${s}_GRCh38_1_22_v4.2.1_benchmark.bed"
      ;;
    *) die "Unsupported sample for BED naming: $s" ;;
  esac
}

download_if_needed() {
  local url="$1"
  local out="$2"

  if [[ -s "$out" && "$FORCE" != "1" ]]; then
    echo "[skip] $out"
    return 0
  fi

  echo "[download] $url"
  curl -fL --retry 5 --retry-delay 3 -o "$out" "$url"
}

write_manifest() {
  local sample="$1"
  local base_url="$2"
  local sample_dir="$3"
  local vcf="$4"
  local tbi="$5"
  local bed="$6"

  SAMPLE="$sample" BASE_URL="$base_url" SAMPLE_DIR="$sample_dir" VCF="$vcf" TBI="$tbi" BED="$bed" python3 - <<'PY'
import json
import os
from pathlib import Path

sample = os.environ['SAMPLE']
base_url = os.environ['BASE_URL']
sample_dir = Path(os.environ['SAMPLE_DIR'])
manifest = {
    "sample": sample,
    "release": "NISTv4.2.1",
    "build": "GRCh38",
    "variant_type": "small_variants",
    "base_url": base_url,
    "files": {
        "truth_vcf": os.environ['VCF'],
        "truth_vcf_tbi": os.environ['TBI'],
        "confident_bed": os.environ['BED'],
    },
}
with open(sample_dir / "source_manifest.json", "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
PY
}

IFS=',' read -r -a SAMPLE_ARR <<< "$SAMPLES"

for SAMPLE in "${SAMPLE_ARR[@]}"; do
  SAMPLE="$(echo "$SAMPLE" | xargs)"
  [[ -n "$SAMPLE" ]] || continue

  BASE_URL="$(sample_base_url "$SAMPLE")"
  SAMPLE_DIR="$OUT_ROOT/$SAMPLE"
  mkdir -p "$SAMPLE_DIR"

  VCF="${SAMPLE}_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
  TBI="${VCF}.tbi"
  BED="$(sample_bed_name "$SAMPLE")"

  download_if_needed "$BASE_URL/$VCF" "$SAMPLE_DIR/$VCF"
  download_if_needed "$BASE_URL/$TBI" "$SAMPLE_DIR/$TBI"
  download_if_needed "$BASE_URL/$BED" "$SAMPLE_DIR/$BED"

  write_manifest "$SAMPLE" "$BASE_URL" "$SAMPLE_DIR" "$VCF" "$TBI" "$BED"
  echo "[ok] $SAMPLE"
done

echo "Done."
