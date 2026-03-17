#!/usr/bin/env bash
set -euo pipefail

#cd ~/VariantCalling

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hts

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

THREADS=8
SEED=42
TARGETS=(30 15 10 5)

# Inputs
ILLUMINA_IN="data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam"
ONT_IN="data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87794.chr20.bam"

# Outputs
OUT_BASE="data/1_input_bams/HG003/downsampled"
MOS_BASE="data/4_out/mosdepth/HG003/chr20/downsampled"

ILL_OUTDIR="$OUT_BASE/Illumina"
ONT_OUTDIR="$OUT_BASE/ONT"
ILL_MOSDIR="$MOS_BASE/Illumina"
ONT_MOSDIR="$MOS_BASE/ONT"

mkdir -p "$ILL_OUTDIR" "$ONT_OUTDIR" "$ILL_MOSDIR" "$ONT_MOSDIR"

# ----------------------------------------------------------------------------
# Get mean depth from mosdepth summary
# ----------------------------------------------------------------------------

get_mean_depth() {
  local summary="$1"
  awk '$1=="total"{print $4}' "$summary"
}

# ----------------------------------------------------------------------------
# Downsample + index + mosdepth for one target depth
# ----------------------------------------------------------------------------

downsample_one() {
  local bam="$1"
  local mean="$2"
  local target="$3"
  local outbam="$4"
  local mos_prefix="$5"

  local frac
  frac=$(awk -v t="$target" -v m="$mean" 'BEGIN { printf "%.6f", t/m }')
  local seed_frac="${SEED}.$(echo "$frac" | cut -c3-)"   # e.g. 42.333333

  echo "  target=${target}x  frac=${frac}  out=$(basename "$outbam")"

  # 1. Downsample and sort
  samtools view  -@ "$THREADS" -s "$seed_frac" -b "$bam" \
    | samtools sort -@ "$THREADS" -o "$outbam" -

  # 2. Index
  samtools index -@ "$THREADS" "$outbam"

  # 3. Verify integrity
  samtools quickcheck -v "$outbam"

  # 4. Calculate stats with mosdepth
  mosdepth -t "$THREADS" -n "$mos_prefix" "$outbam"
  awk '$1 == "total" { print "  mosdepth mean =", $4 "x" }' \
    "${mos_prefix}.mosdepth.summary.txt"
}

# ----------------------------------------------------------------------------
# Process one technology (Illumina or ONT)
# ----------------------------------------------------------------------------

process_tech() {
  local tech="$1"
  local bam="$2"
  local outdir="$3"
  local mosdir="$4"

  # Run mosdepth on the original BAM to get mean depth
  local orig_summary="$mosdir/${tech}_orig.mosdepth.summary.txt"
  # Verify that summary file does not exist
  if [[ ! -f "$orig_summary" ]]; then
    echo "Running mosdepth on original $tech BAM..."
    mosdepth -t "$THREADS" -n "$mosdir/${tech}_orig" "$bam"
  fi
  local mean
  mean=$(get_mean_depth "$orig_summary")

  echo ""
  echo "=== $tech | original mean: ${mean}x ==="

  for target in "${TARGETS[@]}"; do
    local outbam="$outdir/HG003.${tech}.chr20.${target}x.s${SEED}.bam"
    local mos_prefix="$mosdir/${tech}_${target}x_s${SEED}"

    # Skip if BAM, index and mosdepth summary already exist
    if [[ -f "$outbam" && -f "${outbam}.bai" && -f "${mos_prefix}.mosdepth.summary.txt" ]]; then
      echo "  (skip) already done: $(basename "$outbam")"
      continue
    fi

    downsample_one "$bam" "$mean" "$target" "$outbam" "$mos_prefix"
  done
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

process_tech "Illumina" "$ILLUMINA_IN" "$ILL_OUTDIR" "$ILL_MOSDIR"
process_tech "ONT"      "$ONT_IN"      "$ONT_OUTDIR" "$ONT_MOSDIR"

echo ""
echo "All done."
echo ""
echo "Outputs:"
echo "  $ILL_OUTDIR"
echo "  $ONT_OUTDIR"
echo "Mosdepth summaries:"
echo "  $ILL_MOSDIR"
echo "  $ONT_MOSDIR"
