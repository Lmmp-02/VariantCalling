#!/bin/bash
#SBATCH --job-name=dv_build_join
#SBATCH --partition=a64
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
source .venv/bin/activate
mkdir -p out


# Configurables from sbatch --export=ALL,...
DATASET_ID="${DATASET_ID:-}"
MODE="${MODE:-training}"
UNIMODAL_ROOT="${UNIMODAL_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-}"
BUILD_INNER="${BUILD_INNER:-0}"
BUILD_OUTER="${BUILD_OUTER:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-40000}"
DROP_DISAGREE="${DROP_DISAGREE:-1}"
INCLUDE_TEACHER="${INCLUDE_TEACHER:-1}"
WRITE_META_CSV="${WRITE_META_CSV:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-2000}"
DEBUG_SHARDS="${DEBUG_SHARDS:-0}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

[[ -n "$DATASET_ID" ]] || { echo "ERROR: DATASET_ID not set"; exit 1; }

echo "=== ENV CHECK ==="
hostname
uname -m
echo "PWD=$(pwd)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "================="

echo "=== EFFECTIVE CONFIG ==="
echo "DATASET_ID=${DATASET_ID}"
echo "MODE=${MODE}"
echo "BUILD_INNER=${BUILD_INNER}"
echo "BUILD_OUTER=${BUILD_OUTER}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "INCLUDE_TEACHER=${INCLUDE_TEACHER}"
echo "WRITE_META_CSV=${WRITE_META_CSV}"
echo "========================"

RUN_ROOT="${OUT_ROOT:-data/4_out/datasets/multimodal/by_subject}/${DATASET_ID}"
mkdir -p "${RUN_ROOT}/logs"
SLURM_LOG="${RUN_ROOT}/logs/slurm_job_${SLURM_JOB_ID}.log"
exec > >(tee -a "$SLURM_LOG") 2>&1

cmd=(
  bash data/5_scripts/hybrid_dv/dv_build_join_datasets.sh
  --dataset_id "$DATASET_ID"
  --mode "$MODE"
  --build_inner "$BUILD_INNER"
  --build_outer "$BUILD_OUTER"
  --chunk_size "$CHUNK_SIZE"
  --drop_disagree "$DROP_DISAGREE"
  --include_teacher "$INCLUDE_TEACHER"
  --write_meta_csv "$WRITE_META_CSV"
  --progress_every "$PROGRESS_EVERY"
  --debug_shards "$DEBUG_SHARDS"
)
[[ -n "$UNIMODAL_ROOT" ]] && cmd+=( --unimodal_root "$UNIMODAL_ROOT" )
[[ -n "$OUT_ROOT" ]] && cmd+=( --out_root "$OUT_ROOT" )

printf 'RUN CMD: '
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}"

echo "Finished at: $(date)"
