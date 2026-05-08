#!/bin/bash
#SBATCH --job-name=dv_full_call
#SBATCH --partition=da
#SBATCH --time=10:00:00
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

mkdir -p out

RUNTIME="${RUNTIME:-udocker}"
IMAGE="${IMAGE:-dv19gpu}"
OUT_ROOT="${OUT_ROOT:-data/4_out/deepvariant/full_calls}"

DATASET_ID="${DATASET_ID:-}"
TECH="${TECH:-}"
BAM="${BAM:-}"
SAMPLE="${SAMPLE:-HG005}"
SCOPE_ID="${SCOPE_ID:-chr20_chr21}"
REF="${REF:-}"
THREADS="${THREADS:-1}"
FORCE="${FORCE:-0}"
CLEAN_INTERMEDIATE="${CLEAN_INTERMEDIATE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PATH="$REPO/.udocker:$PATH"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

[[ -n "$DATASET_ID" ]] || { echo "ERROR: DATASET_ID not set"; exit 1; }
[[ "$TECH" == "illumina" || "$TECH" == "ont" ]] || { echo "ERROR: TECH must be illumina|ont"; exit 1; }
[[ -n "$BAM" ]] || { echo "ERROR: BAM not set"; exit 1; }

if [[ -z "$REF" ]]; then
  REF="data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.${SCOPE_ID}.fa"
fi

if [[ "$TECH" == "illumina" ]]; then
  DV_PRESET="illumina_wgs"
else
  DV_PRESET="ont_r104"
fi

RUN_DIR="${OUT_ROOT}/${DATASET_ID}/${DV_PRESET}"
JOB_LOG="${RUN_DIR}/logs/slurm_job_${SLURM_JOB_ID:-nojob}.log"
mkdir -p "${RUN_DIR}/logs"

exec > >(tee -a "$JOB_LOG") 2>&1

echo "=== ENV CHECK ==="
hostname
uname -m
echo "PWD=$(pwd)"
echo "udocker=$(command -v udocker || echo '<not found>')"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "================="

echo "=== EFFECTIVE CONFIG ==="
echo "DATASET_ID=${DATASET_ID}"
echo "TECH=${TECH}"
echo "DV_PRESET=${DV_PRESET}"
echo "BAM=${BAM}"
echo "SAMPLE=${SAMPLE}"
echo "SCOPE_ID=${SCOPE_ID}"
echo "REF=${REF}"
echo "THREADS=${THREADS}"
echo "RUNTIME=${RUNTIME}"
echo "IMAGE=${IMAGE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "FORCE=${FORCE}"
echo "CLEAN_INTERMEDIATE=${CLEAN_INTERMEDIATE}"
echo "EXTRA_ARGS=${EXTRA_ARGS:-<none>}"
echo "RUN_DIR=${RUN_DIR}"
echo "========================"

cmd=(
  bash data/5_scripts/dv_run_deepvariant_full.sh
  --runtime "$RUNTIME"
  --image "$IMAGE"
  --dataset_id "$DATASET_ID"
  --tech "$TECH"
  --bam "$BAM"
  --sample "$SAMPLE"
  --scope_id "$SCOPE_ID"
  --ref "$REF"
  --threads "$THREADS"
  --out_root "$OUT_ROOT"
  --force "$FORCE"
  --clean_intermediate "$CLEAN_INTERMEDIATE"
)

[[ -n "$EXTRA_ARGS" ]] && cmd+=( --extra_args "$EXTRA_ARGS" )

printf 'RUN CMD: '
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}"

echo "Finished at: $(date)"
