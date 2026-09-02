#!/usr/bin/env bash
set -euo pipefail

REPO="/home/lantik-deploy/jlazaro/projects/variantcalling"
cd "$REPO"

SAMPLE="HG005"
MODE="training"
REGIONS_ID="chr20_chr21"

BUNDLE_DIR="training/out/reports/hg005_coverage_sweep_internal"
BUNDLE_CSV="${BUNDLE_DIR}/bundle_hg005_coverage_sweep_internal_long.csv"

mkdir -p out
mkdir -p "$BUNDLE_DIR"
mkdir -p training/out/reports
mkdir -p training/out/slurm

# Start clean for this final bundle. The individual JSON/CSVs per eval still go
# into training/out/reports/hg005_trainmode_<COV>x/
rm -f "$BUNDLE_CSV"

SBATCH_DA_COMMON=(
  --partition=da
  --account=fsas
  --nodes=1
  --ntasks=1
  --cpus-per-task=1
  --gres=gpu:1
)

TIME_HEAVY="09:50:00"
TIME_EVAL="09:50:00"
TIME_LIGHT="00:30:00"

echo "============================================================"
echo "Submitting HG005 low-coverage final sequential campaign"
echo "Repo:       $REPO"
echo "Bundle CSV: $BUNDLE_CSV"
echo "Resources: da, 1 task, 1 CPU, 1 GPU"
echo "============================================================"

echo
echo "Checking required runners..."
for f in \
  cluster/run_dv_make_examples.sh \
  cluster/run_dv_export_features.sh \
  cluster/run_dv_build_join_datasets_singleton_locus.sh \
  training/runners/run_eval_checkpoint.sh \
  training/scripts/preprocess/resolve_split.py
do
  [[ -f "$f" ]] || { echo "ERROR: missing required file: $f"; exit 1; }
  echo "OK $f"
done

echo
echo "Checking input BAMs..."
for COV in 20 10 5; do
  BAM_ILL="data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/Illumina/HG005.GRCh38.300x.chr20_chr21.cov${COV}x.s42.bam"
  BAM_ONT="data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/ONT/HG005.GRCh38.ONT_R104_sup_PAW87816_PAW88001.chr20_chr21.cov${COV}x.s42.bam"

  [[ -f "$BAM_ILL" ]] || { echo "ERROR: missing $BAM_ILL"; exit 1; }
  [[ -f "$BAM_ONT" ]] || { echo "ERROR: missing $BAM_ONT"; exit 1; }

  echo "OK ${COV}x Illumina: $BAM_ILL"
  echo "OK ${COV}x ONT:      $BAM_ONT"
done

echo
echo "Creating split configs..."
for COV in 20 10 5; do
  DS="hg005_chr20_chr21_coverage_study_${COV}x"
  SPLIT="hg005_trainmode_${COV}x_final_eval"

  cat > "training/configs/splits/${SPLIT}.json" <<JSON
{
  "split_id": "${SPLIT}",
  "split_strategy": "scope_partition",
  "source_type": "multimodal",
  "description": "Diagnostic supervised evaluation on HG005 chr20+chr21 ${COV}x generated with make_examples in training mode.",
  "partitions": {
    "train": [],
    "val": [],
    "test": [
      {
        "dataset_id": "${DS}",
        "subject": "HG005",
        "chroms": ["chr20", "chr21"],
        "coverage": "${COV}x"
      }
    ]
  }
}
JSON

  echo "Wrote training/configs/splits/${SPLIT}.json"
done

JOB_PREV=""

submit_mex() {
  local COV="$1"
  local TECH="$2"

  local DS="hg005_chr20_chr21_coverage_study_${COV}x"
  local COVERAGE_TAG="coverage_study_${COV}x"
  local BAM=""

  if [[ "$TECH" == "illumina" ]]; then
    BAM="data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/Illumina/HG005.GRCh38.300x.chr20_chr21.cov${COV}x.s42.bam"
  elif [[ "$TECH" == "ont" ]]; then
    BAM="data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/ONT/HG005.GRCh38.ONT_R104_sup_PAW87816_PAW88001.chr20_chr21.cov${COV}x.s42.bam"
  else
    echo "ERROR: unknown TECH=$TECH"
    exit 1
  fi

  local PRESET=""
  if [[ "$TECH" == "illumina" ]]; then
    PRESET="illumina_wgs"
  elif [[ "$TECH" == "ont" ]]; then
    PRESET="ont_r104"
  fi

  local SUCCESS_PATH="data/4_out/deepvariant/examples/${DS}/${PRESET}/training/SUCCESS"

  if [[ -f "$SUCCESS_PATH" ]]; then
    echo
    echo "Skipping make_examples ${TECH} ${COV}x because SUCCESS exists:"
    echo "  ${SUCCESS_PATH}"
    return 0
  fi

  local DEP_ARGS=()
  if [[ -n "$JOB_PREV" ]]; then
    DEP_ARGS=(--dependency=afterok:${JOB_PREV})
  fi

  echo
  echo "Submitting make_examples ${TECH} ${COV}x"
  [[ -n "$JOB_PREV" ]] && echo "Depends on: $JOB_PREV"

  local JID
  JID=$(sbatch --parsable \
    "${SBATCH_DA_COMMON[@]}" \
    --time="$TIME_HEAVY" \
    "${DEP_ARGS[@]}" \
    --job-name="mex_${TECH}_${COV}x" \
    --export=ALL,TECH="${TECH}",BAM="${BAM}",SAMPLE="${SAMPLE}",MODE="${MODE}",REGIONS_ID="${REGIONS_ID}",COVERAGE_TAG="${COVERAGE_TAG}",DATASET_ID="${DS}",FORCE=0,RESUME=1 \
    cluster/run_dv_make_examples.sh)

  echo "Job ID: $JID"
  JOB_PREV="$JID"
}

submit_export() {
  local COV="$1"
  local TECH="$2"
  local DS="hg005_chr20_chr21_coverage_study_${COV}x"

  echo
  echo "Submitting export ${TECH} ${COV}x"
  echo "Depends on: $JOB_PREV"

  local JID
  JID=$(sbatch --parsable \
    "${SBATCH_DA_COMMON[@]}" \
    --time="$TIME_HEAVY" \
    --dependency=afterok:${JOB_PREV} \
    --job-name="exp_${TECH}_${COV}x" \
    --export=ALL,DATASET_ID="${DS}",TECH="${TECH}",MODE=training \
    cluster/run_dv_export_features.sh)

  echo "Job ID: $JID"
  JOB_PREV="$JID"
}

submit_join() {
  local COV="$1"
  local DS="hg005_chr20_chr21_coverage_study_${COV}x"

  echo
  echo "Submitting join ${COV}x"
  echo "Depends on: $JOB_PREV"

  local JID
  JID=$(sbatch --parsable \
    "${SBATCH_DA_COMMON[@]}" \
    --time="$TIME_HEAVY" \
    --dependency=afterok:${JOB_PREV} \
    --job-name="join_${COV}x" \
    --export=ALL,DATASET_ID="${DS}",MODE=training,BUILD_INNER=0,BUILD_OUTER=1,FORCE=1,WRITE_META_CSV=1 \
    cluster/run_dv_build_join_datasets_singleton_locus.sh)

  echo "Job ID: $JID"
  JOB_PREV="$JID"
}

submit_split() {
  local COV="$1"
  local SPLIT="hg005_trainmode_${COV}x_final_eval"

  echo
  echo "Submitting resolve_split ${COV}x"
  echo "Depends on: $JOB_PREV"

  local JID
  JID=$(sbatch --parsable \
    "${SBATCH_DA_COMMON[@]}" \
    --time="$TIME_LIGHT" \
    --dependency=afterok:${JOB_PREV} \
    --job-name="split_${COV}x" \
    --output="${REPO}/out/split_${COV}x_%j.log" \
    --wrap="bash -lc 'cd ${REPO}; if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi; python training/scripts/preprocess/resolve_split.py --split_config training/configs/splits/${SPLIT}.json'")

  echo "Job ID: $JID"
  JOB_PREV="$JID"
}

submit_eval_teacher() {
  local COV="$1"
  local TEACHER="$2"
  local NAME="$3"

  local RES="training/configs/splits/resolved__hg005_trainmode_${COV}x_final_eval.json"
  local OUT="training/out/reports/hg005_trainmode_${COV}x"

  mkdir -p "$OUT"

  echo
  echo "Submitting eval teacher ${NAME} ${COV}x"
  echo "Depends on: $JOB_PREV"

  local JID
  JID=$(sbatch --parsable \
    "${SBATCH_DA_COMMON[@]}" \
    --time="$TIME_EVAL" \
    --dependency=afterok:${JOB_PREV} \
    --job-name="eval_${NAME}_${COV}x" \
    --export=ALL,SOURCE=teacher,TEACHER="${TEACHER}",RESOLVED_SPLIT="${RES}",OUT_DIR="${OUT}",OUT_CSV="${BUNDLE_CSV}",NAME="${NAME}" \
    training/runners/run_eval_checkpoint.sh)

  echo "Job ID: $JID"
  JOB_PREV="$JID"
}

submit_eval_checkpoint() {
  local COV="$1"
  local EXPERIMENT_DIR="$2"
  local NAME="$3"

  local RES="training/configs/splits/resolved__hg005_trainmode_${COV}x_final_eval.json"
  local OUT="training/out/reports/hg005_trainmode_${COV}x"

  mkdir -p "$OUT"

  echo
  echo "Submitting eval checkpoint ${NAME} ${COV}x"
  echo "Depends on: $JOB_PREV"

  local JID
  JID=$(sbatch --parsable \
    "${SBATCH_DA_COMMON[@]}" \
    --time="$TIME_EVAL" \
    --dependency=afterok:${JOB_PREV} \
    --job-name="eval_${NAME}_${COV}x" \
    --export=ALL,SOURCE=checkpoint,EXPERIMENT_DIR="${EXPERIMENT_DIR}",CHECKPOINT=best.pt,RESOLVED_SPLIT="${RES}",OUT_DIR="${OUT}",OUT_CSV="${BUNDLE_CSV}",NAME="${NAME}" \
    training/runners/run_eval_checkpoint.sh)

  echo "Job ID: $JID"
  JOB_PREV="$JID"
}

echo
echo "============================================================"
echo "Submitting sequential chain"
echo "============================================================"

# 1) make_examples
submit_mex 20 illumina
submit_mex 20 ont
submit_mex 10 illumina
submit_mex 10 ont
submit_mex 5 illumina
submit_mex 5 ont

# 2) unimodal exports
submit_export 20 illumina
submit_export 20 ont
submit_export 10 illumina
submit_export 10 ont
submit_export 5 illumina
submit_export 5 ont

# 3) multimodal joins
submit_join 20
submit_join 10
submit_join 5

# 4) split resolution
submit_split 20
submit_split 10
submit_split 5

# 5) internal evaluation. Sequential on purpose so OUT_CSV append is safe.
for COV in 20 10 5; do
  submit_eval_teacher "$COV" illumina dv_illumina
  submit_eval_teacher "$COV" ont      dv_ont

  submit_eval_checkpoint "$COV" "training/out/experiments/unimodal_illumina_hg002_hg003_vs_hg004" unimodal_illumina
  submit_eval_checkpoint "$COV" "training/out/experiments/unimodal_ont_hg002_hg003_vs_hg004"      unimodal_ont
  submit_eval_checkpoint "$COV" "training/out/experiments/hybrid_simple_hg002_hg003_vs_hg004"      hybrid_simple
  submit_eval_checkpoint "$COV" "training/out/experiments/hybrid_groupwise_hg002_hg003_vs_hg004"   hybrid_groupwise
done

echo
echo "============================================================"
echo "Submission completed"
echo "Final chained job: $JOB_PREV"
echo "$JOB_PREV" > out/hg005_lowcov_final_chained_jobid.txt
echo "Bundle CSV:"
echo "$BUNDLE_CSV"
echo "============================================================"
