#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT="training/scripts/hybrid_dv/hybrid/two_stage/calibrate_stage1_thresholds.py"

SHARED_CKPT="training/out/models/stage1/hybrid_stage1_shared_mlp/best.pt"
GROUPWISE_CKPT="training/out/models/stage1/hybrid_stage1_groupwise_mlp/best.pt"

# 1) shared + global
$PY "$SCRIPT" \
  --ckpt "$SHARED_CKPT" \
  --threshold_mode global \
  --global_rule recall_at_min_precision \
  --global_min_precision 0.60 \
  --out_json training/out/models/stage1/hybrid_stage1_shared_mlp/calib_global.json

# 2) shared + per-group
$PY "$SCRIPT" \
  --ckpt "$SHARED_CKPT" \
  --threshold_mode per_group \
  --g1_rule recall_at_max_fpr --g1_max_fpr 0.01 \
  --g10_rule best_f2 \
  --g11_rule precision_at_min_recall --g11_min_recall 0.999 \
  --out_json training/out/models/stage1/hybrid_stage1_shared_mlp/calib_per_group.json

# 3) groupwise + global
$PY "$SCRIPT" \
  --ckpt "$GROUPWISE_CKPT" \
  --threshold_mode global \
  --global_rule recall_at_min_precision \
  --global_min_precision 0.60 \
  --out_json training/out/models/stage1/hybrid_stage1_groupwise_mlp/calib_global.json

# 4) groupwise + per-group
$PY "$SCRIPT" \
  --ckpt "$GROUPWISE_CKPT" \
  --threshold_mode per_group \
  --g1_rule recall_at_max_fpr --g1_max_fpr 0.01 \
  --g10_rule best_f2 \
  --g11_rule precision_at_min_recall --g11_min_recall 0.999 \
  --out_json training/out/models/stage1/hybrid_stage1_groupwise_mlp/calib_per_group.json