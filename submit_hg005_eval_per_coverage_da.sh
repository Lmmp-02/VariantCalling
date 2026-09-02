#!/usr/bin/env bash
set -euo pipefail

REPO="/home/lantik-deploy/jlazaro/projects/variantcalling"
cd "$REPO"

# Relanzamos todos los niveles para tener artefactos homogéneos.
COVS=(40 20 10 5)

BASE_OUT="training/out/reports/hg005_coverage_sweep_internal/per_coverage_rerun"
mkdir -p "$BASE_OUT"
mkdir -p out/submit_logs

SBATCH_DA_COMMON=(
  --partition=da
  --account=fsas
  --nodes=1
  --ntasks=1
  --cpus-per-task=1
  --gres=gpu:1
  --time=09:50:00
)

echo "============================================================"
echo "Submitting HG005 per-coverage evaluation rerun"
echo "Repo:     $REPO"
echo "Output:   $BASE_OUT"
echo "Coverage: ${COVS[*]}"
echo "Resources: da, 1 task, 1 CPU, 1 GPU"
echo "============================================================"

echo
echo "Checking resolved splits..."
for COV in "${COVS[@]}"; do
  RES="training/configs/splits/resolved__hg005_trainmode_${COV}x_final_eval.json"
  [[ -f "$RES" ]] || { echo "ERROR: missing resolved split: $RES"; exit 1; }
  echo "OK $RES"
done

echo
echo "Checking experiment dirs..."
for EXP in \
  training/out/experiments/unimodal_illumina_hg002_hg003_vs_hg004 \
  training/out/experiments/unimodal_ont_hg002_hg003_vs_hg004 \
  training/out/experiments/hybrid_simple_hg002_hg003_vs_hg004 \
  training/out/experiments/hybrid_groupwise_hg002_hg003_vs_hg004
do
  [[ -d "$EXP" ]] || { echo "ERROR: missing experiment dir: $EXP"; exit 1; }
  echo "OK $EXP"
done

JOB_PREV=""

submit_eval_teacher() {
  local COV="$1"
  local TEACHER="$2"
  local NAME="$3"

  local RES="training/configs/splits/resolved__hg005_trainmode_${COV}x_final_eval.json"
  local OUT_DIR="${BASE_OUT}/hg005_trainmode_${COV}x"
  local OUT_CSV="${BASE_OUT}/summary_hg005_trainmode_${COV}x_long.csv"

  mkdir -p "$OUT_DIR"

  local DEP_ARGS=()
  if [[ -n "$JOB_PREV" ]]; then
    DEP_ARGS=(--dependency=afterok:${JOB_PREV})
  fi

  echo
  echo "Submitting ${COV}x teacher eval: ${NAME}"
  [[ -n "$JOB_PREV" ]] && echo "Depends on: $JOB_PREV"

  local JID
  JID=$(sbatch --parsable \
    "${SBATCH_DA_COMMON[@]}" \
    "${DEP_ARGS[@]}" \
    --job-name="rerun_${NAME}_${COV}x" \
    --export=ALL,SOURCE=teacher,TEACHER="${TEACHER}",RESOLVED_SPLIT="${RES}",OUT_DIR="${OUT_DIR}",OUT_CSV="${OUT_CSV}",NAME="${NAME}" \
    training/runners/run_eval_checkpoint.sh)

  echo "Job ID: $JID"
  JOB_PREV="$JID"
}

submit_eval_checkpoint() {
  local COV="$1"
  local EXPERIMENT_DIR="$2"
  local NAME="$3"

  local RES="training/configs/splits/resolved__hg005_trainmode_${COV}x_final_eval.json"
  local OUT_DIR="${BASE_OUT}/hg005_trainmode_${COV}x"
  local OUT_CSV="${BASE_OUT}/summary_hg005_trainmode_${COV}x_long.csv"

  mkdir -p "$OUT_DIR"

  local DEP_ARGS=()
  if [[ -n "$JOB_PREV" ]]; then
    DEP_ARGS=(--dependency=afterok:${JOB_PREV})
  fi

  echo
  echo "Submitting ${COV}x checkpoint eval: ${NAME}"
  [[ -n "$JOB_PREV" ]] && echo "Depends on: $JOB_PREV"

  local JID
  JID=$(sbatch --parsable \
    "${SBATCH_DA_COMMON[@]}" \
    "${DEP_ARGS[@]}" \
    --job-name="rerun_${NAME}_${COV}x" \
    --export=ALL,SOURCE=checkpoint,EXPERIMENT_DIR="${EXPERIMENT_DIR}",CHECKPOINT=best.pt,RESOLVED_SPLIT="${RES}",OUT_DIR="${OUT_DIR}",OUT_CSV="${OUT_CSV}",NAME="${NAME}" \
    training/runners/run_eval_checkpoint.sh)

  echo "Job ID: $JID"
  JOB_PREV="$JID"
}

# Limpiar solo los CSVs nuevos de este rerun, no los resultados antiguos.
for COV in "${COVS[@]}"; do
  rm -f "${BASE_OUT}/summary_hg005_trainmode_${COV}x_long.csv"
done

# Evaluaciones secuenciales para evitar escrituras concurrentes al CSV.
for COV in "${COVS[@]}"; do
  submit_eval_teacher "$COV" illumina dv_illumina
  submit_eval_teacher "$COV" ont      dv_ont

  submit_eval_checkpoint "$COV" "training/out/experiments/unimodal_illumina_hg002_hg003_vs_hg004" unimodal_illumina
  submit_eval_checkpoint "$COV" "training/out/experiments/unimodal_ont_hg002_hg003_vs_hg004"      unimodal_ont
  submit_eval_checkpoint "$COV" "training/out/experiments/hybrid_simple_hg002_hg003_vs_hg004"      hybrid_simple
  submit_eval_checkpoint "$COV" "training/out/experiments/hybrid_groupwise_hg002_hg003_vs_hg004"   hybrid_groupwise
done

echo
echo "============================================================"
echo "Submission completed"
echo "Final chained job: $JOB_PREV"
echo "$JOB_PREV" > out/hg005_eval_per_coverage_rerun_final_jobid.txt
echo
echo "Per-coverage CSVs will be written to:"
for COV in "${COVS[@]}"; do
  echo "  ${BASE_OUT}/summary_hg005_trainmode_${COV}x_long.csv"
done
echo "============================================================"
