#!/usr/bin/env bash
# Additional experiments of the classical paper, run sequentially on one machine.
#
# Stages (run in this order; each one only needs the outputs of the previous ones):
#
#   resolve     resolve the training split and the four HG005 evaluation splits
#   configs     write the 36 seed / dv_probs experiment configs (paper_seeds/)
#   canonical   train the 4 canonical representation heads (seed 0)
#   train       train the 36 seed / dv_probs heads; verifies the canonical checkpoints
#   dump        write one prediction NPZ per evaluation set with all 40 checkpoints
#   analyse     compute the CSV files in training/out/paper/analysis/ (CPU only)
#
#   bash training/scripts/paper/run_paper_experiments.sh <stage> [<stage> ...]
#   bash training/scripts/paper/run_paper_experiments.sh all
#
# Environment: DEVICE (auto | cpu | cuda, default auto), BATCH_SIZE (dump, default 8192),
# DRY_RUN (0 | 1, default 0). DRY_RUN=1 is a pure planner: it prints commands without
# creating directories, generating configs, training models or writing outputs.
# Prerequisite: the singleton-locus offline datasets (scripts/multimodal/).
# A job scheduler is not required. On a cluster, wrap each command in your own job script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
DRY_RUN="${DRY_RUN:-0}"
case "$DRY_RUN" in
  0|1) ;;
  *) echo "[ERROR] DRY_RUN must be 0 or 1 (got: $DRY_RUN)"; exit 1 ;;
esac

RESOLVED="training/out/resolved"
EXP="training/out/experiments"
PRED="training/out/paper/predictions"
ANALYSIS="training/out/paper/analysis"
SEED_CONFIGS="training/configs/experiments/paper_seeds"
CANONICAL_CONFIGS="training/configs/experiments"

TRAIN_SPLIT_ID="hg002_hg003_train__hg004_valtest__singleton_locus"
TRAIN_SPLIT="$RESOLVED/$TRAIN_SPLIT_ID.json"
FAMILIES=(unimodal_illumina unimodal_ont hybrid_simple hybrid_groupwise)
EVAL_KEYS=(hg004_chr20_40x hg005_40x hg005_20x hg005_10x hg005_5x)

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

eval_split() {
  case "$1" in
    hg004_chr20_40x) echo "$TRAIN_SPLIT" ;;                 # test partition = HG004 chr20
    hg005_*)         echo "$RESOLVED/hg005_trainmode_${1#hg005_}_final_eval.json" ;;
  esac
}

trainer_for() {
  case "$1" in
    unimodal_*) echo "training/scripts/train/unimodal/train_unimodal.py" ;;
    hybrid_*)   echo "training/scripts/train/hybrid/train_hybrid.py" ;;
  esac
}

train_config() {
  local name="$1" cfg="$2"
  echo "[Train] $name"
  run_cmd python "$(trainer_for "$name")" \
    --experiment_config "$cfg" \
    --resolved_split "$TRAIN_SPLIT" \
    --device "$DEVICE"
}

stage_resolve() {
  if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p "$RESOLVED"
  fi
  for id in "$TRAIN_SPLIT_ID" hg005_trainmode_{40x,20x,10x,5x}_final_eval; do
    run_cmd python training/scripts/preprocess/resolve_split.py \
      --split_config "training/configs/splits/$id.json" \
      --output_json "$RESOLVED/$id.json"
  done
}

stage_configs() {
  run_cmd python training/scripts/paper/make_seed_configs.py --seeds 1 2 3 4
  run_cmd python training/scripts/paper/make_seed_configs.py --input_features dv_probs --seeds 0 1 2 3 4
}

stage_canonical() {
  local fam cfg
  for fam in "${FAMILIES[@]}"; do
    cfg="$CANONICAL_CONFIGS/${fam}_hg002_hg003_vs_hg004.json"
    train_config "$fam" "$cfg"
  done
}

stage_train() {
  local fam cfg name
  if [[ "$DRY_RUN" == "0" ]]; then
    for fam in "${FAMILIES[@]}"; do
      [[ -f "$EXP/${fam}_hg002_hg003_vs_hg004/best.pt" ]] || {
        echo "[ERROR] missing canonical checkpoint $EXP/${fam}_hg002_hg003_vs_hg004/best.pt"; exit 1; }
    done
  fi
  for cfg in "$SEED_CONFIGS"/*.json; do
    name="$(basename "$cfg" .json)"
    train_config "$name" "$cfg"
  done
}

all_models() {
  local out=() fam s
  for fam in "${FAMILIES[@]}"; do
    out+=("${fam}=$EXP/${fam}_hg002_hg003_vs_hg004")
    for s in 1 2 3 4; do
      out+=("${fam}__seed${s}=$EXP/paper_seeds/${fam}_hg002_hg003_vs_hg004__seed${s}")
    done
    for s in 0 1 2 3 4; do
      out+=("${fam}__dvprobs__seed${s}=$EXP/paper_seeds/${fam}_hg002_hg003_vs_hg004__dvprobs__seed${s}")
    done
  done
  printf '%s\n' "${out[@]}"
}

stage_dump() {
  if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p "$PRED"
  fi
  local args=() spec k
  while read -r spec; do args+=(--model "$spec"); done < <(all_models)
  for k in "${EVAL_KEYS[@]}"; do
    echo "[Dump] $k"
    run_cmd python training/scripts/paper/dump_predictions.py \
      --resolved_split "$(eval_split "$k")" \
      --partition test \
      --device "$DEVICE" --batch_size "$BATCH_SIZE" \
      "${args[@]}" \
      --out_npz "$PRED/${k}__all.npz"
  done
}

stage_analyse() {
  run_cmd python training/scripts/paper/analyse_predictions.py \
    --pred_dir "$PRED" --out_dir "$ANALYSIS"
}

[[ $# -ge 1 ]] || { sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 1; }
[[ "$1" == "all" ]] && set -- resolve configs canonical train dump analyse
for stage in "$@"; do
  case "$stage" in
    resolve|configs|canonical|train|dump|analyse) "stage_$stage" ;;
    *) echo "[ERROR] unknown stage: $stage (resolve | configs | canonical | train | dump | analyse | all)"; exit 1 ;;
  esac
done
