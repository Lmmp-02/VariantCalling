#!/usr/bin/env bash
# Additional experiments of the classical paper, run sequentially on one machine.
#
# Stages (run in this order; each one only needs the outputs of the previous ones):
#
#   resolve   resolve the training split and the four HG005 evaluation splits
#   configs   write the 36 seed / dv_probs experiment configs (paper_seeds/)
#   train     train the 36 heads (the 4 canonical seed-0 heads are trained as usual)
#   dump      write one prediction NPZ per evaluation set with all 40 checkpoints
#   analyse   compute the CSV files in training/out/paper/analysis/ (CPU only)
#
#   bash training/scripts/paper/run_paper_experiments.sh <stage> [<stage> ...]
#   bash training/scripts/paper/run_paper_experiments.sh all
#
# Environment: DEVICE (auto | cpu | cuda, default auto), BATCH_SIZE (dump, default 8192).
# Prerequisite: the singleton-locus offline datasets (scripts/multimodal/) and the four
# canonical seed-0 checkpoints in training/out/experiments/<experiment_id>/.
# A job scheduler is not required. On a cluster, wrap each command in your own job script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8192}"

RESOLVED="training/out/resolved"
EXP="training/out/experiments"
PRED="training/out/paper/predictions"
ANALYSIS="training/out/paper/analysis"
SEED_CONFIGS="training/configs/experiments/paper_seeds"

TRAIN_SPLIT_ID="hg002_hg003_train__hg004_valtest__singleton_locus"
TRAIN_SPLIT="$RESOLVED/$TRAIN_SPLIT_ID.json"
FAMILIES=(unimodal_illumina unimodal_ont hybrid_simple hybrid_groupwise)
EVAL_KEYS=(hg004_chr20_40x hg005_40x hg005_20x hg005_10x hg005_5x)

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

stage_resolve() {
  mkdir -p "$RESOLVED"
  for id in "$TRAIN_SPLIT_ID" hg005_trainmode_{40x,20x,10x,5x}_final_eval; do
    python training/scripts/preprocess/resolve_split.py \
      --split_config "training/configs/splits/$id.json" \
      --output_json "$RESOLVED/$id.json"
  done
}

stage_configs() {
  python training/scripts/paper/make_seed_configs.py --seeds 1 2 3 4
  python training/scripts/paper/make_seed_configs.py --input_features dv_probs --seeds 0 1 2 3 4
}

stage_train() {
  for fam in "${FAMILIES[@]}"; do
    [[ -f "$EXP/${fam}_hg002_hg003_vs_hg004/best.pt" ]] || {
      echo "[ERROR] missing canonical checkpoint $EXP/${fam}_hg002_hg003_vs_hg004/best.pt"; exit 1; }
  done
  for cfg in "$SEED_CONFIGS"/*.json; do
    name="$(basename "$cfg" .json)"
    echo "[Train] $name"
    python "$(trainer_for "$name")" \
      --experiment_config "$cfg" \
      --resolved_split "$TRAIN_SPLIT" \
      --device "$DEVICE"
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
  mkdir -p "$PRED"
  local args=() spec k
  while read -r spec; do args+=(--model "$spec"); done < <(all_models)
  for k in "${EVAL_KEYS[@]}"; do
    echo "[Dump] $k"
    python training/scripts/paper/dump_predictions.py \
      --resolved_split "$(eval_split "$k")" \
      --partition test \
      --device "$DEVICE" --batch_size "$BATCH_SIZE" \
      "${args[@]}" \
      --out_npz "$PRED/${k}__all.npz"
  done
}

stage_analyse() {
  python training/scripts/paper/analyse_predictions.py \
    --pred_dir "$PRED" --out_dir "$ANALYSIS"
}

[[ $# -ge 1 ]] || { sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 1; }
[[ "$1" == "all" ]] && set -- resolve configs train dump analyse
for stage in "$@"; do
  case "$stage" in
    resolve|configs|train|dump|analyse) "stage_$stage" ;;
    *) echo "[ERROR] unknown stage: $stage (resolve | configs | train | dump | analyse | all)"; exit 1 ;;
  esac
done
