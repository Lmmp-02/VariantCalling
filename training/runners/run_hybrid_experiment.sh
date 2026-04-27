#!/bin/bash
#SBATCH --job-name=vc_hybrid
#SBATCH --partition=da
#SBATCH --time=10:00:00
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
mkdir -p training/out/experiments

: "${EXPERIMENT_CONFIG:?ERROR: set EXPERIMENT_CONFIG}"
: "${RESOLVED_SPLIT:?ERROR: set RESOLVED_SPLIT}"

DEVICE="${DEVICE:-cuda}"

echo "============================================================"
echo "[Job] Variant Calling hybrid training"
echo "============================================================"
echo "HOSTNAME          : $(hostname)"
echo "SLURM_JOB_ID      : ${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NAME    : ${SLURM_JOB_NAME:-NA}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NA}"
echo "REPO              : $REPO"
echo "EXPERIMENT_CONFIG : $EXPERIMENT_CONFIG"
echo "RESOLVED_SPLIT    : $RESOLVED_SPLIT"
echo "DEVICE            : $DEVICE"
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

echo "[Run] Starting training..."

python -u training/scripts/train/hybrid/train_hybrid.py \
  --experiment_config "$EXPERIMENT_CONFIG" \
  --resolved_split "$RESOLVED_SPLIT" \
  --device "$DEVICE"

echo "[Done] Training completed."