#!/usr/bin/env bash
set -euo pipefail

cd ~/VariantCalling

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hts

THREADS=8
SEED=42
TARGETS=(30 15 10 5)

# Inputs
ILLUMINA_IN="data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam"

# ONT partial: try common paths (adjust if needed)
ONT_CANDIDATES=(
  "data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87794.chr20.bam"
  "data/1_input_bams/HG003/HG003.GRCh38.ONT_R104_sup_PAY87794.chr20.bam"
)
ONT_IN=""
for c in "${ONT_CANDIDATES[@]}"; do
  if [[ -f "$c" ]]; then
    ONT_IN="$c"
    break
  fi
done
: "${ONT_IN:?No encuentro el BAM ONT partial. Ajusta ONT_CANDIDATES.}"

# Outputs
OUT_BASE="data/1_input_bams/HG003/downsampled"
MOS_BASE="data/4_out/mosdepth/HG003/chr20/downsampled"

ILL_OUTDIR="$OUT_BASE/ILLUMINA_chr20"
ONT_OUTDIR="$OUT_BASE/ONT_partial_chr20"
ILL_MOSDIR="$MOS_BASE/ILLUMINA"
ONT_MOSDIR="$MOS_BASE/ONT_partial"

mkdir -p "$ILL_OUTDIR" "$ONT_OUTDIR" "$ILL_MOSDIR" "$ONT_MOSDIR"

mean_from_summary () {
  local summary="$1"
  awk '$1=="total"{print $4}' "$summary"
}

ensure_orig_mosdepth_and_mean () {
  local bam="$1"
  local prefix="$2"
  local outdir="$3"
  local summary="$outdir/${prefix}.mosdepth.summary.txt"

  if [[ ! -f "$summary" ]]; then
    mosdepth -t "$THREADS" -n "$outdir/$prefix" "$bam"
  fi

  local mean
  mean=$(mean_from_summary "$summary")
  echo "$mean"
}

should_skip_run () {
  local outbam="$1"
  local mos_summary="$2"

  if [[ -f "$outbam" && -f "${outbam}.bai" && -f "$mos_summary" ]]; then
    return 0
  fi
  return 1
}

downsample_one () {
  local bam="$1"
  local mean="$2"
  local tech="$3"     # "ILLUMINA" / "ONT_partial"
  local outdir="$4"
  local mosdir="$5"

  echo ""
  echo "=== $tech ==="
  echo "Input:  $bam"
  echo "Mean (orig, from summary): $mean x"

  for target in "${TARGETS[@]}"; do
    local frac outbam mos_prefix mos_summary

    frac=$(awk -v t="$target" -v m="$mean" 'BEGIN{printf "%.6f", t/m}')

    outbam="$outdir/HG003.${tech}.chr20.${target}x.s${SEED}.bam"
    mos_prefix="$mosdir/${tech}_${target}x_s${SEED}"
    mos_summary="${mos_prefix}.mosdepth.summary.txt"

    if should_skip_run "$outbam" "$mos_summary"; then
      echo ">> (skip) ya existe completo: $outbam"
      awk '$1=="total"{print "  mosdepth mean=", $4 "x"}' "$mos_summary" || true
      continue
    fi

    echo ">> TARGET=${target}x  FRAC=${frac}  OUT=$outbam"

    samtools view -@ "$THREADS" -s "${SEED}.${frac#0.}" -b "$bam" \
      | samtools sort -@ "$THREADS" -o "$outbam" -

    samtools index -@ "$THREADS" "$outbam"
    samtools quickcheck -v "$outbam"

    mosdepth -t "$THREADS" -n "$mos_prefix" "$outbam"
    awk '$1=="total"{print "  mosdepth mean=", $4 "x"}' "$mos_summary"
  done
}

# ==========================
# Extra seeds (optional)
# ==========================
EXTRA_SEEDS=(123 2026)
EXTRA_TARGETS=(10 5)

downsample_extra_seeds () {
  local bam="$1"
  local mean="$2"
  local tech="$3"
  local outdir="$4"
  local mosdir="$5"

  echo ""
  echo "=== EXTRA SEEDS: $tech ==="
  echo "Input: $bam"
  echo "Mean (orig): $mean x"

  for seed in "${EXTRA_SEEDS[@]}"; do
    for target in "${EXTRA_TARGETS[@]}"; do
      local frac outbam mos_prefix mos_summary

      frac=$(awk -v t="$target" -v m="$mean" 'BEGIN{printf "%.6f", t/m}')

      outbam="$outdir/HG003.${tech}.chr20.${target}x.s${seed}.bam"
      mos_prefix="$mosdir/${tech}_${target}x_s${seed}"
      mos_summary="${mos_prefix}.mosdepth.summary.txt"

      if should_skip_run "$outbam" "$mos_summary"; then
        echo ">> (skip) ya existe completo: $outbam"
        awk '$1=="total"{print "  mosdepth mean=", $4 "x"}' "$mos_summary" || true
        continue
      fi

      echo ">> SEED=${seed} TARGET=${target}x FRAC=${frac} OUT=$outbam"

      samtools view -@ "$THREADS" -s "${seed}.${frac#0.}" -b "$bam" \
        | samtools sort -@ "$THREADS" -o "$outbam" -

      samtools index -@ "$THREADS" "$outbam"
      samtools quickcheck -v "$outbam"

      mosdepth -t "$THREADS" -n "$mos_prefix" "$outbam"
      awk '$1=="total"{print "  mosdepth mean=", $4 "x"}' "$mos_summary"
    done
  done
}

# 1) Get means from mosdepth summaries (generate if missing)
ILL_MEAN=$(ensure_orig_mosdepth_and_mean "$ILLUMINA_IN" "ILLUMINA_orig" "$ILL_MOSDIR")
ONT_MEAN=$(ensure_orig_mosdepth_and_mean "$ONT_IN"      "ONT_partial_orig" "$ONT_MOSDIR")

# 2) Base downsample set (seed 42) + checks
downsample_one "$ILLUMINA_IN" "$ILL_MEAN" "ILLUMINA"    "$ILL_OUTDIR" "$ILL_MOSDIR"
downsample_one "$ONT_IN"      "$ONT_MEAN" "ONT_partial" "$ONT_OUTDIR" "$ONT_MOSDIR"

echo ""
echo "Done. Outputs:"
echo "  $ILL_OUTDIR"
echo "  $ONT_OUTDIR"
echo "Mosdepth summaries:"
echo "  $ILL_MOSDIR"
echo "  $ONT_MOSDIR"

# 3) Optional extra seeds (10x, 5x) if RUN_EXTRA=1
if [[ "${RUN_EXTRA:-0}" == "1" ]]; then
  downsample_extra_seeds "$ILLUMINA_IN" "$ILL_MEAN" "ILLUMINA"    "$ILL_OUTDIR" "$ILL_MOSDIR"
  downsample_extra_seeds "$ONT_IN"      "$ONT_MEAN" "ONT_partial" "$ONT_OUTDIR" "$ONT_MOSDIR"
fi