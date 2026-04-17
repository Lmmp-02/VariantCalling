#!/bin/bash
#SBATCH --job-name=dv_make_examples
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

# Fijos para este runner
RUNTIME="udocker"
IMAGE="dv19"

# Configurables desde fuera con sbatch --export=ALL,...
TECH="${TECH:-ont}"
BAM="${BAM:-}"
SAMPLE="${SAMPLE:-}"
MODE="${MODE:-training}"
REGIONS_ID="${REGIONS_ID:-}"
COVERAGE_TAG="${COVERAGE_TAG:-}"
DATASET_ID="${DATASET_ID:-}"
REGIONS="${REGIONS:-}"

# Opcionales avanzados, normalmente vacíos
REF="${REF:-}"
TRUTH_VCF="${TRUTH_VCF:-}"
CONFIDENT_BED="${CONFIDENT_BED:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VALIDATE_LABELS="${VALIDATE_LABELS:-}"
VALIDATOR_PY="${VALIDATOR_PY:-}"
VALIDATOR_MAX_RECORDS="${VALIDATOR_MAX_RECORDS:-}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-1}"

# Derivados de Slurm
export PATH="$REPO/.udocker:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

SHARDS="${SLURM_CPUS_PER_TASK:-1}"
JOBS="${SLURM_CPUS_PER_TASK:-1}"

# Checks mínimos
[[ -n "$TECH" ]] || { echo "ERROR: TECH not set"; exit 1; }
[[ -n "$BAM" ]] || { echo "ERROR: BAM not set"; exit 1; }
[[ -n "$SAMPLE" ]] || { echo "ERROR: SAMPLE not set"; exit 1; }
[[ -n "$REGIONS_ID" ]] || { echo "ERROR: REGIONS_ID not set"; exit 1; }

echo "=== ENV CHECK ==="
hostname
uname -m
echo "PWD=$(pwd)"
echo "udocker=$(command -v udocker || echo '<not found>')"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "================="

echo "=== EFFECTIVE CONFIG ==="
echo "TECH=${TECH}"
echo "BAM=${BAM}"
echo "SAMPLE=${SAMPLE}"
echo "MODE=${MODE}"
echo "REGIONS_ID=${REGIONS_ID}"
echo "COVERAGE_TAG=${COVERAGE_TAG}"
echo "DATASET_ID=${DATASET_ID:-<auto>}"
echo "REGIONS=${REGIONS:-<none>}"
echo "SHARDS=${SHARDS}"
echo "JOBS=${JOBS}"
echo "========================"

cmd=(
  bash data/5_scripts/dv_make_examples.sh
  --runtime "$RUNTIME"
  --image "$IMAGE"
  --tech "$TECH"
  --bam "$BAM"
  --sample "$SAMPLE"
  --mode "$MODE"
  --regions_id "$REGIONS_ID"
  --shards "$SHARDS"
  --jobs "$JOBS"
  --force "$FORCE"
  --resume "$RESUME"
)

[[ -n "$COVERAGE_TAG" ]] && cmd+=( --coverage_tag "$COVERAGE_TAG" )
[[ -n "$DATASET_ID" ]] && cmd+=( --dataset_id "$DATASET_ID" )
[[ -n "$REGIONS" ]] && cmd+=( --regions "$REGIONS" )
[[ -n "$REF" ]] && cmd+=( --ref "$REF" )
[[ -n "$TRUTH_VCF" ]] && cmd+=( --truth_vcf "$TRUTH_VCF" )
[[ -n "$CONFIDENT_BED" ]] && cmd+=( --confident_bed "$CONFIDENT_BED" )
[[ -n "$EXTRA_ARGS" ]] && cmd+=( --extra_args "$EXTRA_ARGS" )
[[ -n "$VALIDATE_LABELS" ]] && cmd+=( --validate_labels "$VALIDATE_LABELS" )
[[ -n "$VALIDATOR_PY" ]] && cmd+=( --validator_py "$VALIDATOR_PY" )
[[ -n "$VALIDATOR_MAX_RECORDS" ]] && cmd+=( --validator_max_records "$VALIDATOR_MAX_RECORDS" )

printf 'RUN CMD: '
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}"

echo "Finished at: $(date)"