#!/usr/bin/env bash
set -euo pipefail

cd /home/lantik-deploy/jlazaro/projects/variantcalling

export PATH="$PWD/.udocker:$PATH"
export PYTHONUNBUFFERED=1

RUNTIME="udocker"
IMAGE="dv19"

SAMPLE="HG005"
MODE="training"
REGIONS_ID="chr20_chr21"

SHARDS=1
JOBS=1

FORCE="${FORCE:-1}"
RESUME="${RESUME:-0}"

LOG_DIR="out/login_make_examples_hg005_lowcov_training"
mkdir -p "$LOG_DIR"

for COV in 20 10 5; do
  DATASET_ID="hg005_chr20_chr21_coverage_study_${COV}x"
  COVERAGE_TAG="coverage_study_${COV}x"

  BAM_ILL="data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/Illumina/HG005.GRCh38.300x.chr20_chr21.cov${COV}x.s42.bam"
  BAM_ONT="data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/ONT/HG005.GRCh38.ONT_R104_sup_PAW87816_PAW88001.chr20_chr21.cov${COV}x.s42.bam"

  echo
  echo "============================================================"
  echo "HG005 chr20+chr21 make_examples TRAINING mode"
  echo "Coverage:   ${COV}x"
  echo "Dataset ID: ${DATASET_ID}"
  echo "Started:    $(date)"
  echo "Host:       $(hostname)"
  echo "============================================================"

  test -f "$BAM_ILL" || { echo "ERROR: missing $BAM_ILL"; exit 1; }
  test -f "$BAM_ONT" || { echo "ERROR: missing $BAM_ONT"; exit 1; }

  echo
  echo "1/2 Illumina WGS make_examples — ${COV}x"
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
    2>&1 | tee "${LOG_DIR}/illumina_${COV}x_training.log"

  echo
  echo "2/2 ONT R104 make_examples — ${COV}x"
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
    2>&1 | tee "${LOG_DIR}/ont_${COV}x_training.log"

  echo "DONE ${DATASET_ID}: $(date)"
done
