#!/bin/bash
#SBATCH --job-name=dv_build_join
#SBATCH --partition=da
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=fsas
#SBATCH --output=/home/lantik-deploy/jlazaro/projects/variantcalling/out/%x_%j.log
#SBATCH --error=/home/lantik-deploy/jlazaro/projects/variantcalling/out/%x_%j.err

set -euo pipefail

REPO=/home/lantik-deploy/jlazaro/projects/variantcalling
cd "$REPO"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p out

# Configurables from sbatch --export=ALL,...
DATASET_ID="${DATASET_ID:-}"
MODE="${MODE:-training}"
UNIMODAL_ROOT="${UNIMODAL_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-}"
BUILD_INNER="${BUILD_INNER:-0}"
BUILD_OUTER="${BUILD_OUTER:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-40000}"
DROP_AMBIGUOUS_BIMODAL="${DROP_AMBIGUOUS_BIMODAL:-1}"
INCLUDE_TEACHER="${INCLUDE_TEACHER:-1}"
WRITE_META_CSV="${WRITE_META_CSV:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-2000}"
DEBUG_SHARDS="${DEBUG_SHARDS:-0}"
FORCE="${FORCE:-0}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

[[ -n "$DATASET_ID" ]] || { echo "ERROR: DATASET_ID not set"; exit 1; }
[[ "$MODE" == "training" || "$MODE" == "calling" ]] || { echo "ERROR: MODE must be training|calling"; exit 1; }

echo "=== ENV CHECK ==="
hostname
uname -m
echo "PWD=$(pwd)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "================="

echo "=== EFFECTIVE CONFIG ==="
echo "DATASET_ID=${DATASET_ID}"
echo "MODE=${MODE}"
echo "UNIMODAL_ROOT=${UNIMODAL_ROOT:-<canonical>}"
echo "OUT_ROOT=${OUT_ROOT:-<canonical>}"
echo "BUILD_INNER=${BUILD_INNER}"
echo "BUILD_OUTER=${BUILD_OUTER}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "DROP_AMBIGUOUS_BIMODAL=${DROP_AMBIGUOUS_BIMODAL}"
echo "INCLUDE_TEACHER=${INCLUDE_TEACHER}"
echo "WRITE_META_CSV=${WRITE_META_CSV}"
echo "PROGRESS_EVERY=${PROGRESS_EVERY}"
echo "DEBUG_SHARDS=${DEBUG_SHARDS}"
echo "FORCE=${FORCE}"
echo "========================"

RUN_ROOT="${OUT_ROOT:-data/4_out/datasets/multimodal/by_subject}/${DATASET_ID}"
mkdir -p "${RUN_ROOT}/logs"
SLURM_LOG="${RUN_ROOT}/logs/slurm_job_${SLURM_JOB_ID:-nojob}.log"
exec > >(tee -a "$SLURM_LOG") 2>&1

cmd=(
  bash data/5_scripts/hybrid_dv/dv_build_join_datasets.sh
  --dataset_id "$DATASET_ID"
  --mode "$MODE"
  --build_inner "$BUILD_INNER"
  --build_outer "$BUILD_OUTER"
  --chunk_size "$CHUNK_SIZE"
  --drop_ambiguous_bimodal "$DROP_AMBIGUOUS_BIMODAL"
  --include_teacher "$INCLUDE_TEACHER"
  --write_meta_csv "$WRITE_META_CSV"
  --progress_every "$PROGRESS_EVERY"
  --debug_shards "$DEBUG_SHARDS"
  --force "$FORCE"
)

[[ -n "$UNIMODAL_ROOT" ]] && cmd+=( --unimodal_root "$UNIMODAL_ROOT" )
[[ -n "$OUT_ROOT" ]] && cmd+=( --out_root "$OUT_ROOT" )

printf 'RUN CMD: '
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}"

echo "Finished at: $(date)"