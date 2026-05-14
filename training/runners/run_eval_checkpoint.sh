#!/bin/bash
#SBATCH --job-name=vc_eval
#SBATCH --partition=da
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=fsas
#SBATCH --output=/home/lantik-deploy/jlazaro/projects/variantcalling/training/out/slurm/%x_%j.log

set -euo pipefail

REPO=/home/lantik-deploy/jlazaro/projects/variantcalling
cd "$REPO"

source .venv/bin/activate

mkdir -p training/out/slurm
mkdir -p training/out/reports

# ---------------------------------------------------------------------
# Required config from sbatch --export=ALL,...
# ---------------------------------------------------------------------
: "${SOURCE:?ERROR: set SOURCE=teacher or SOURCE=checkpoint}"
: "${RESOLVED_SPLIT:?ERROR: set RESOLVED_SPLIT}"

# ---------------------------------------------------------------------
# Optional/shared config
# ---------------------------------------------------------------------
EVAL_PARTITION="${EVAL_PARTITION:-test}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
OUT_DIR="${OUT_DIR:-training/out/reports/hg005_trainmode_40x_final_eval}"
NAME="${NAME:-}"

# ---------------------------------------------------------------------
# Teacher/checkpoint specific config
# ---------------------------------------------------------------------
TEACHER="${TEACHER:-}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
CHECKPOINT="${CHECKPOINT:-best.pt}"

OUT_JSON="${OUT_JSON:-}"
OUT_CSV="${OUT_CSV:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "[Job] Variant Calling internal eval"
echo "============================================================"
echo "HOSTNAME             : $(hostname)"
echo "SLURM_JOB_ID         : ${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NAME       : ${SLURM_JOB_NAME:-NA}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-NA}"
echo "REPO                 : $REPO"
echo "SOURCE               : $SOURCE"
echo "TEACHER              : ${TEACHER:-<none>}"
echo "EXPERIMENT_DIR       : ${EXPERIMENT_DIR:-<none>}"
echo "CHECKPOINT           : $CHECKPOINT"
echo "RESOLVED_SPLIT       : $RESOLVED_SPLIT"
echo "EVAL_PARTITION       : $EVAL_PARTITION"
echo "DEVICE               : $DEVICE"
echo "BATCH_SIZE           : $BATCH_SIZE"
echo "OUT_DIR              : $OUT_DIR"
echo "NAME                 : ${NAME:-<auto>}"
echo "OUT_JSON             : ${OUT_JSON:-<auto>}"
echo "OUT_CSV              : ${OUT_CSV:-<auto>}"
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
  python -u training/scripts/eval/eval_checkpoint.py
  --source "$SOURCE"
  --resolved_split "$RESOLVED_SPLIT"
  --partition "$EVAL_PARTITION"
  --batch_size "$BATCH_SIZE"
  --device "$DEVICE"
  --out_dir "$OUT_DIR"
)

if [ -n "$NAME" ]; then
  cmd+=( --name "$NAME" )
fi

if [ -n "$OUT_JSON" ]; then
  cmd+=( --out_json "$OUT_JSON" )
fi

if [ -n "$OUT_CSV" ]; then
  cmd+=( --out_csv "$OUT_CSV" )
fi

if [ "$SOURCE" = "teacher" ]; then
  : "${TEACHER:?ERROR: SOURCE=teacher requires TEACHER=illumina or TEACHER=ont}"
  cmd+=( --teacher "$TEACHER" )
elif [ "$SOURCE" = "checkpoint" ]; then
  : "${EXPERIMENT_DIR:?ERROR: SOURCE=checkpoint requires EXPERIMENT_DIR}"
  cmd+=( --experiment_dir "$EXPERIMENT_DIR" )
  cmd+=( --checkpoint "$CHECKPOINT" )
else
  echo "ERROR: unsupported SOURCE=$SOURCE"
  exit 1
fi

echo "[Run] Starting internal evaluation..."
printf 'RUN CMD: '
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}"

echo "[Done] Evaluation completed."