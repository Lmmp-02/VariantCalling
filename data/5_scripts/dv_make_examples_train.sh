#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ------------------------------------------------------------------------------
# Examples (run from repo root):
#
# Illumina (chr20):
#   bash data/5_scripts/dv_make_examples_train.sh \
#     --tech illumina \
#     --bam data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam \
#     --region chr20
#
# ONT (chr20): (adds channel 7 via --channel_list haplotype)
#   bash data/5_scripts/dv_make_examples_train.sh \
#     --tech ont \
#     --bam data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87794.chr20.bam \
#     --region chr20
#
# Notes:
# - Defaults assume HG003 + chr20 reference/truth/confident bed from your tree.
# - Output: data/4_out/deepvariant/<SAMPLE>/make_examples_train/<run_id>/
# - Symlinks updated on success:
#     latest            -> last successful run (any tech)
#     latest_illumina   -> last successful Illumina run
#     latest_ont        -> last successful ONT run
# ------------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat >&2 <<'EOF'
DeepVariant make_examples (TRAINING) runner, adapted to VariantCalling repo layout.

Required:
  --tech illumina|ont
  --bam PATH
  --region REGION         (e.g. chr20 or chr20:1-1000000)

Optional (repo defaults are correct for your tree):
  --sample NAME           (default: HG003)
  --ref PATH              (default: data/2_reference/.../GRCh38_no_alt_plus_hs38d1.chr20.fa)
  --truth_vcf PATH        (default: data/3_truth/HG003/HG003_chr20.truth.vcf.gz)
  --confident_bed PATH    (default: data/3_truth/HG003/HG003_chr20.confident.bed)
  --docker_image IMAGE    (default: google/deepvariant:1.9.0)
  --shards N              (default: nproc)
  --jobs N                (default: min(shards, nproc))
  --resume 0|1            (default: 1)
  --out_root DIR          (default: data/4_out/deepvariant/<sample>/make_examples_train)
  --extra_args "..."      (default: "")

Outputs:
  <out_root>/<run_id>/
    - *.tfrecord-00000-of-000NN.gz (+ example_info.json)
    - logs/
    - manifest.txt
    - label_check.txt
    - SUCCESS
    - latest -> <run_id>   (symlink updated only if success)

EOF
}

to_abs() {
  local p="$1"
  [[ "$p" == /* ]] && { echo "$p"; return; }
  echo "$(pwd)/$p"
}

# ---- defaults (adapted to your repo) ----
TECH=""
BAM=""
REGION=""
SAMPLE="HG003"

REF_DEFAULT="data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa"
TRUTH_DEFAULT="data/3_truth/HG003/HG003_chr20.truth.vcf.gz"
CONF_DEFAULT="data/3_truth/HG003/HG003_chr20.confident.bed"

REF="$REF_DEFAULT"
TRUTH_VCF="$TRUTH_DEFAULT"
CONF_BED="$CONF_DEFAULT"

DOCKER_IMAGE="google/deepvariant:1.9.0"
SHARDS="$(nproc)"
JOBS=""
RESUME=1
OUT_ROOT=""   # computed later
EXTRA_ARGS=""

# ---- args ----
[[ $# -eq 0 ]] && { usage; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tech) TECH="$2"; shift 2;;
    --bam) BAM="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;

    --sample) SAMPLE="$2"; shift 2;;
    --ref) REF="$2"; shift 2;;
    --truth_vcf) TRUTH_VCF="$2"; shift 2;;
    --confident_bed) CONF_BED="$2"; shift 2;;

    --docker_image) DOCKER_IMAGE="$2"; shift 2;;
    --shards) SHARDS="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --resume) RESUME="$2"; shift 2;;
    --out_root) OUT_ROOT="$2"; shift 2;;
    --extra_args) EXTRA_ARGS="$2"; shift 2;;

    -h|--help) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

# ---- validate required ----
[[ -n "$TECH" && -n "$BAM" && -n "$REGION" ]] || { usage; die "Missing required args."; }
[[ "$TECH" == "illumina" || "$TECH" == "ont" ]] || die "--tech must be illumina|ont"

command -v docker >/dev/null 2>&1 || die "docker not found in PATH"

# ---- resolve paths (from repo root) ----
REPO_ROOT="$(pwd)"
BAM="$(to_abs "$BAM")"
REF="$(to_abs "$REF")"
TRUTH_VCF="$(to_abs "$TRUTH_VCF")"
CONF_BED="$(to_abs "$CONF_BED")"

[[ -f "$BAM" ]] || die "BAM not found: $BAM"
[[ -f "$REF" ]] || die "REF not found: $REF"
[[ -f "$TRUTH_VCF" ]] || die "TRUTH_VCF not found: $TRUTH_VCF"
[[ -f "$CONF_BED" ]] || die "CONFIDENT_BED not found: $CONF_BED"

# indexes: fail fast
[[ -f "${REF}.fai" ]] || die "Missing FASTA index: ${REF}.fai"
[[ -f "${BAM}.bai" || -f "${BAM%.bam}.bai" ]] || die "Missing BAM index next to BAM (.bai): $BAM"
[[ -f "${TRUTH_VCF}.tbi" ]] || die "Missing TRUTH_VCF index (.tbi): ${TRUTH_VCF}.tbi"

# ---- output root default (your repo) ----
if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="data/4_out/deepvariant/${SAMPLE}/make_examples_train"
fi
OUT_ROOT="$(to_abs "$OUT_ROOT")"
mkdir -p "$OUT_ROOT"

if [[ -z "$JOBS" ]]; then
  NP="$(nproc)"
  if (( SHARDS < NP )); then JOBS="$SHARDS"; else JOBS="$NP"; fi
fi

# ---- tech presets ----
CHANNEL_LIST=""
PILEUP_WIDTH=""
ALT_ALIGNED_ARGS=""

if [[ "$TECH" == "illumina" ]]; then
  CHANNEL_LIST="BASE_CHANNELS,insert_size"
  PILEUP_WIDTH="221"
else
  # ONT R10.4 models expect 9 channels: [1..6,7,9,10]
  # - channel 7 comes from adding 'haplotype' to channel_list (add_hp_channel is deprecated)
  # - channels 9,10 come from diff_channels
  CHANNEL_LIST="BASE_CHANNELS,haplotype"
  PILEUP_WIDTH="99"
  ALT_ALIGNED_ARGS="--alt_aligned_pileup=diff_channels"
fi

# ---- rollback-friendly run dir ----
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SAFE_REGION="${REGION//[:\/]/_}"
RUN_ID="${TS}_${TECH}_${SAMPLE}_${SAFE_REGION}"
RUN_DIR="${OUT_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

MANIFEST="${RUN_DIR}/manifest.txt"
SUCCESS_MARK="${RUN_DIR}/SUCCESS"
LATEST_LINK="${OUT_ROOT}/latest"
LATEST_TECH_LINK="${OUT_ROOT}/latest_${TECH}"

{
  echo "run_id: ${RUN_ID}"
  echo "timestamp_utc: ${TS}"
  echo "repo_root: ${REPO_ROOT}"
  echo "tech: ${TECH}"
  echo "sample: ${SAMPLE}"
  echo "region: ${REGION}"
  echo "docker_image: ${DOCKER_IMAGE}"
  echo "shards: ${SHARDS}"
  echo "jobs: ${JOBS}"
  echo "bam: ${BAM}"
  echo "ref: ${REF}"
  echo "truth_vcf: ${TRUTH_VCF}"
  echo "confident_bed: ${CONF_BED}"
  echo "channel_list: ${CHANNEL_LIST}"
  echo "pileup_image_width: ${PILEUP_WIDTH}"
  echo "preset_extra: ${ALT_ALIGNED_ARGS}"
  echo "user_extra_args: ${EXTRA_ARGS}"
  echo "git_head: $(git rev-parse HEAD 2>/dev/null || echo NA)"
} > "$MANIFEST"

EX_BASE="${RUN_DIR}/${TECH}.${SAMPLE}.${SAFE_REGION}.training.with_label.tfrecord"
EXAMPLES="${EX_BASE}@${SHARDS}.gz"

shard_path() {
  local task="$1"
  printf "%s-%05d-of-%05d.gz" "$EX_BASE" "$task" "$SHARDS"
}

run_task() {
  local task="$1"
  local shard log
  shard="$(shard_path "$task")"
  log="${EX_BASE%/*}/logs/make_examples.task_${task}.log"

  if [[ "$RESUME" == "1" && -s "$shard" && -s "${shard}.example_info.json" ]]; then
    echo "[task ${task}] skip (resume)"
    return 0
  fi

  docker run --rm \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${REPO_ROOT}" \
    "${DOCKER_IMAGE}" \
    /opt/deepvariant/bin/make_examples \
      --mode training \
      --ref "${REF}" \
      --reads "${BAM}" \
      --truth_variants "${TRUTH_VCF}" \
      --confident_regions "${CONF_BED}" \
      --regions "${REGION}" \
      --examples "${EXAMPLES}" \
      --task "${task}" \
      --channel_list "${CHANNEL_LIST}" \
      --pileup_image_width "${PILEUP_WIDTH}" \
      ${ALT_ALIGNED_ARGS} \
      ${EXTRA_ARGS} \
      --logtostderr \
    >"${log}" 2>&1
}

export -f run_task shard_path
export REPO_ROOT DOCKER_IMAGE REF BAM TRUTH_VCF CONF_BED REGION EXAMPLES EX_BASE SHARDS RESUME ALT_ALIGNED_ARGS EXTRA_ARGS LOG_DIR CHANNEL_LIST PILEUP_WIDTH

echo "==> RUN_DIR: ${RUN_DIR}"
echo "==> EXAMPLES: ${EXAMPLES}"
echo "==> Launching ${SHARDS} tasks with concurrency=${JOBS}"

seq 0 $((SHARDS - 1)) | xargs -I{} -P "${JOBS}" bash -lc 'run_task "$@"' _ {}

echo "==> Verifying outputs..."
missing=0
for t in $(seq 0 $((SHARDS - 1))); do
  s="$(shard_path "$t")"
  [[ -s "$s" ]] || { echo "Missing shard: $s"; missing=1; }
  [[ -s "${s}.example_info.json" ]] || { echo "Missing example_info: ${s}.example_info.json"; missing=1; }
done
[[ "$missing" == "0" ]] || die "Some shards are missing. Check logs in ${LOG_DIR}"

# ---- label presence check (uses a real .py, no heredoc) ----
LABEL_GLOB="${EX_BASE}-*-of-*.gz"

docker run --rm \
  -v "${REPO_ROOT}:${REPO_ROOT}" \
  -w "${REPO_ROOT}" \
  "${DOCKER_IMAGE}" \
  python3 -u data/5_scripts/check_tfrecord_has_label.py \
    --glob "${LABEL_GLOB}" \
    --max_records 3 \
  > "${RUN_DIR}/label_check.txt" 2>&1

grep -q "label_found=True" "${RUN_DIR}/label_check.txt" || die "Label not found (or check failed)! See ${RUN_DIR}/label_check.txt"

echo "ok" > "$SUCCESS_MARK"
ln -sfn "$RUN_DIR" "$LATEST_LINK"
ln -sfn "$RUN_DIR" "$LATEST_TECH_LINK"

echo "==> SUCCESS"
echo "Latest: ${LATEST_LINK}"
echo "Latest (tech): ${LATEST_TECH_LINK}"