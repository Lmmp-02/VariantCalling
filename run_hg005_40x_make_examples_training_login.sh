#!/usr/bin/env bash
set -euo pipefail

cd /home/lantik-deploy/jlazaro/projects/variantcalling

# ---------------------------------------------------------------------
# Dirty login-node runner for HG005 chr20+chr21 make_examples training
# 40x harmonized Illumina + ONT
# ---------------------------------------------------------------------

export PATH="$PWD/.udocker:$PATH"
export PYTHONUNBUFFERED=1

RUNTIME="udocker"
IMAGE="dv19"

SAMPLE="HG005"
MODE="training"
REGIONS_ID="chr20_chr21"
COVERAGE_TAG="harmonized_40x"
DATASET_ID="hg005_chr20_chr21_harmonized_40x"

# Conservative values for login node
SHARDS=1
JOBS=1

FORCE=0
RESUME=1

# ---------------------------------------------------------------------
# BAMs
# Adjust these two paths if your 40x BAM names differ.
# ---------------------------------------------------------------------

BAM_ILL="data/1_input_bams/HG005/derived/chr20_chr21/harmonized_40x/Illumina/HG005.GRCh38.300x.chr20_chr21.cov40x.s42.bam"
BAM_ONT="data/1_input_bams/HG005/derived/chr20_chr21/harmonized_40x/ONT/HG005.GRCh38.ONT_R104_sup_PAW87816_PAW88001.chr20_chr21.cov40x.s42.bam"

LOG_DIR="out/login_make_examples_hg005_40x_training"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "HG005 chr20+chr21 make_examples TRAINING mode"
echo "Coverage:   ${COVERAGE_TAG}"
echo "Dataset ID: ${DATASET_ID}"
echo "Runtime:    ${RUNTIME}"
echo "Image:      ${IMAGE}"
echo "Shards:     ${SHARDS}"
echo "Jobs:       ${JOBS}"
echo "Started:    $(date)"
echo "Host:       $(hostname)"
echo "============================================================"

echo
echo "Checking udocker..."
command -v udocker

echo
echo "Checking input BAMs..."
if [[ ! -f "$BAM_ILL" ]]; then
  echo "ERROR: Illumina BAM not found:"
  echo "  $BAM_ILL"
  echo
  echo "Candidate 40x Illumina BAMs:"
  find data/1_input_bams/HG005 -type f -iname "*40x*.bam" | grep -i illumina || true
  exit 1
fi

if [[ ! -f "$BAM_ONT" ]]; then
  echo "ERROR: ONT BAM not found:"
  echo "  $BAM_ONT"
  echo
  echo "Candidate 40x ONT BAMs:"
  find data/1_input_bams/HG005 -type f -iname "*40x*.bam" | grep -Ei "ont|nanopore|r104" || true
  exit 1
fi

ls -lh "$BAM_ILL"
ls -lh "$BAM_ONT"

echo
echo "============================================================"
echo "1/2 Illumina WGS make_examples"
echo "============================================================"

bash data/5_scripts/dv_make_examples.sh \
  --runtime "$RUNTIME" \
  --image "$IMAGE" \
  --tech illumina \
  --bam "$BAM_ILL" \
  --sample "$SAMPLE" \
  --mode "$MODE" \
  --regions_id "$REGIONS_ID" \
  --coverage_tag "$COVERAGE_TAG" \
  --dataset_id "$DATASET_ID" \
  --shards "$SHARDS" \
  --jobs "$JOBS" \
  --force "$FORCE" \
  --resume "$RESUME" \
  2>&1 | tee "${LOG_DIR}/illumina_wgs_training.log"

echo
echo "============================================================"
echo "2/2 ONT R104 make_examples"
echo "============================================================"

bash data/5_scripts/dv_make_examples.sh \
  --runtime "$RUNTIME" \
  --image "$IMAGE" \
  --tech ont \
  --bam "$BAM_ONT" \
  --sample "$SAMPLE" \
  --mode "$MODE" \
  --regions_id "$REGIONS_ID" \
  --coverage_tag "$COVERAGE_TAG" \
  --dataset_id "$DATASET_ID" \
  --shards "$SHARDS" \
  --jobs "$JOBS" \
  --force "$FORCE" \
  --resume "$RESUME" \
  2>&1 | tee "${LOG_DIR}/ont_r104_training.log"

echo
echo "============================================================"
echo "DONE: $(date)"
echo "Expected outputs:"
echo "  data/4_out/deepvariant/examples/${DATASET_ID}/illumina_wgs/training"
echo "  data/4_out/deepvariant/examples/${DATASET_ID}/ont_r104/training"
echo "============================================================"