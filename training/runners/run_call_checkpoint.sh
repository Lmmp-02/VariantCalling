#!/bin/bash
#SBATCH --job-name=vc_call
#SBATCH --partition=da
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --account=fsas
#SBATCH --output=/home/lantik-deploy/jlazaro/projects/variantcalling/training/out/slurm/%x_%j.log

set -euo pipefail

REPO=/home/lantik-deploy/jlazaro/projects/variantcalling
cd "$REPO"

source .venv/bin/activate

mkdir -p training/out/slurm
mkdir -p training/out/calls

# ---------------------------------------------------------------------
# Required config from sbatch --export=ALL,...
# ---------------------------------------------------------------------
: "${EXPERIMENT_DIR:?ERROR: set EXPERIMENT_DIR}"
: "${DATASET_ID:?ERROR: set DATASET_ID}"

# ---------------------------------------------------------------------
# Optional config
# ---------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-best.pt}"
MODE="${MODE:-calling}"
INPUT_KIND="${INPUT_KIND:-auto}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"

UNIMODAL_ROOT="${UNIMODAL_ROOT:-data/4_out/datasets/unimodal}"
MULTIMODAL_ROOT="${MULTIMODAL_ROOT:-data/4_out/datasets/multimodal/by_subject}"

UNIMODAL_DIR="${UNIMODAL_DIR:-}"
MULTIMODAL_OUTER_DIR="${MULTIMODAL_OUTER_DIR:-}"

OUT_DIR="${OUT_DIR:-}"
OUT_CSV="${OUT_CSV:-}"
OUT_REPORT="${OUT_REPORT:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "[Job] Variant Calling checkpoint calling"
echo "============================================================"
echo "HOSTNAME             : $(hostname)"
echo "SLURM_JOB_ID         : ${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NAME       : ${SLURM_JOB_NAME:-NA}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-NA}"
echo "REPO                 : $REPO"
echo "EXPERIMENT_DIR       : $EXPERIMENT_DIR"
echo "CHECKPOINT           : $CHECKPOINT"
echo "DATASET_ID           : $DATASET_ID"
echo "MODE                 : $MODE"
echo "INPUT_KIND           : $INPUT_KIND"
echo "DEVICE               : $DEVICE"
echo "BATCH_SIZE           : $BATCH_SIZE"
echo "LIMIT_ROWS           : $LIMIT_ROWS"
echo "UNIMODAL_ROOT        : $UNIMODAL_ROOT"
echo "MULTIMODAL_ROOT      : $MULTIMODAL_ROOT"
echo "UNIMODAL_DIR         : ${UNIMODAL_DIR:-<auto>}"
echo "MULTIMODAL_OUTER_DIR : ${MULTIMODAL_OUTER_DIR:-<auto>}"
echo "OUT_DIR              : ${OUT_DIR:-<script default>}"
echo "OUT_CSV              : ${OUT_CSV:-<script default>}"
echo "OUT_REPORT           : ${OUT_REPORT:-<script default>}"
echo "============================================================"

echo "[GPU] nvidia-smi"
nvidia-smi || true

echo "[Python] CUDA availability"
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device name:", torch.cuda.get_device_name(0))
PY

if [ "$DEVICE" = "cuda" ]; then
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: DEVICE=cuda but torch.cuda.is_available() is False")
PY
fi

cmd=(
  python -u training/scripts/call/call_checkpoint.py
  --experiment_dir "$EXPERIMENT_DIR"
  --checkpoint "$CHECKPOINT"
  --dataset_id "$DATASET_ID"
  --mode "$MODE"
  --input_kind "$INPUT_KIND"
  --unimodal_root "$UNIMODAL_ROOT"
  --multimodal_root "$MULTIMODAL_ROOT"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --limit_rows "$LIMIT_ROWS"
)

[[ -n "$UNIMODAL_DIR" ]] && cmd+=( --unimodal_dir "$UNIMODAL_DIR" )
[[ -n "$MULTIMODAL_OUTER_DIR" ]] && cmd+=( --multimodal_outer_dir "$MULTIMODAL_OUTER_DIR" )
[[ -n "$OUT_DIR" ]] && cmd+=( --out_dir "$OUT_DIR" )
[[ -n "$OUT_CSV" ]] && cmd+=( --out_csv "$OUT_CSV" )
[[ -n "$OUT_REPORT" ]] && cmd+=( --out_report "$OUT_REPORT" )

echo "[Run] Starting checkpoint calling..."
printf 'RUN CMD: '
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}"

echo "[Done] Calling completed."