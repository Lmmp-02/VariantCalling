#!/bin/bash
#SBATCH --job-name=dv_export_features
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

mkdir -p out

# Fixed defaults for this runner
RUNTIME="udocker"
IMAGE="dv19gpu"
OUT_ROOT="data/4_out/datasets/unimodal"
MODE="${MODE:-training}"

# Configurable from sbatch --export=ALL,...
DATASET_ID="${DATASET_ID:-}"
TECH="${TECH:-}"
EXAMPLES_DIR="${EXAMPLES_DIR:-}"
MODEL_DIR="${MODEL_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CHUNK_SIZE="${CHUNK_SIZE:-40000}"
MAX_RECORDS="${MAX_RECORDS:-}"
EMB_TENSOR="${EMB_TENSOR:-}"
LOGITS_TENSOR="${LOGITS_TENSOR:-}"

# Metadata export controls
EMIT_IDENTITY_META="${EMIT_IDENTITY_META:-0}"
EMIT_VCF_META="${EMIT_VCF_META:-1}"
STRICT_VCF_META="${STRICT_VCF_META:-1}"

# Derived from Slurm / environment
export PATH="$REPO/.udocker:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# Minimal checks
[[ -n "$DATASET_ID" ]] || { echo "ERROR: DATASET_ID not set"; exit 1; }
[[ "$TECH" == "illumina" || "$TECH" == "ont" ]] || { echo "ERROR: TECH must be illumina|ont"; exit 1; }
[[ "$MODE" == "training" || "$MODE" == "calling" ]] || { echo "ERROR: MODE must be training|calling"; exit 1; }

[[ "$EMIT_IDENTITY_META" == "0" || "$EMIT_IDENTITY_META" == "1" ]] || {
  echo "ERROR: EMIT_IDENTITY_META must be 0|1"; exit 1;
}
[[ "$EMIT_VCF_META" == "0" || "$EMIT_VCF_META" == "1" ]] || {
  echo "ERROR: EMIT_VCF_META must be 0|1"; exit 1;
}
[[ "$STRICT_VCF_META" == "0" || "$STRICT_VCF_META" == "1" ]] || {
  echo "ERROR: STRICT_VCF_META must be 0|1"; exit 1;
}

if [[ "$TECH" == "illumina" ]]; then
  DV_PRESET="illumina_wgs"
else
  DV_PRESET="ont_r104"
fi

RUN_DIR="${OUT_ROOT}/${DATASET_ID}/${DV_PRESET}/${MODE}"
JOB_LOG="${RUN_DIR}/logs/slurm_job_${SLURM_JOB_ID:-nojob}.log"
mkdir -p "${RUN_DIR}/logs"

exec > >(tee -a "$JOB_LOG") 2>&1

echo "=== ENV CHECK ==="
hostname
uname -m
echo "PWD=$(pwd)"
echo "udocker=$(command -v udocker || echo '<not found>')"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "================="

echo "=== EFFECTIVE CONFIG ==="
echo "DATASET_ID=${DATASET_ID}"
echo "TECH=${TECH}"
echo "DV_PRESET=${DV_PRESET}"
echo "MODE=${MODE}"
echo "EXAMPLES_DIR=${EXAMPLES_DIR:-<canonical>}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "RUN_DIR=${RUN_DIR}"
echo "MODEL_DIR=${MODEL_DIR:-<auto>}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "MAX_RECORDS=${MAX_RECORDS:-<none>}"
echo "EMB_TENSOR=${EMB_TENSOR:-<auto>}"
echo "LOGITS_TENSOR=${LOGITS_TENSOR:-<auto>}"
echo "EMIT_IDENTITY_META=${EMIT_IDENTITY_META}"
echo "EMIT_VCF_META=${EMIT_VCF_META}"
echo "STRICT_VCF_META=${STRICT_VCF_META}"
echo "IMAGE=${IMAGE}"
echo "========================"

cmd=(
  bash data/5_scripts/dv_export_features.sh
  --runtime "$RUNTIME"
  --image "$IMAGE"
  --dataset_id "$DATASET_ID"
  --tech "$TECH"
  --mode "$MODE"
  --out_root "$OUT_ROOT"
  --batch_size "$BATCH_SIZE"
  --chunk_size "$CHUNK_SIZE"
  --emit_identity_meta "$EMIT_IDENTITY_META"
  --emit_vcf_meta "$EMIT_VCF_META"
  --strict_vcf_meta "$STRICT_VCF_META"
)

[[ -n "$EXAMPLES_DIR" ]] && cmd+=( --examples_dir "$EXAMPLES_DIR" )
[[ -n "$MODEL_DIR" ]] && cmd+=( --model_dir "$MODEL_DIR" )
[[ -n "$MAX_RECORDS" ]] && cmd+=( --max_records "$MAX_RECORDS" )
[[ -n "$EMB_TENSOR" ]] && cmd+=( --emb_tensor "$EMB_TENSOR" )
[[ -n "$LOGITS_TENSOR" ]] && cmd+=( --logits_tensor "$LOGITS_TENSOR" )

printf 'RUN CMD: '
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}"

echo "Finished at: $(date)"