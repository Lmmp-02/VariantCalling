#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# DeepVariant make_examples runner for the VariantCalling repo.
#
# Canonical intent:
# - consume a BAM that already represents the scope to process
# - generate DeepVariant examples for that BAM under a given technology/mode
#
# Canonical output contract:
#   data/4_out/deepvariant/examples/<dataset_id>/<dv_preset>/<mode>/
#
# Notes:
# - Intended as a stable, one-time artifact generator.
# - No latest symlinks, no timestamped run folders.
# - Re-running the same command is idempotent:
#     * if SUCCESS exists, exits cleanly
#     * if partial shards exist, can resume
#     * use --force 1 to rebuild from scratch
# - Post-generation validation is OPTIONAL and NON-BLOCKING.
# -----------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

usage() {
  cat >&2 <<'EOF_USAGE'
DeepVariant make_examples runner.

Required:
  --tech illumina|ont
  --bam PATH

Optional but strongly recommended:
  --sample NAME              (if omitted, inferred from BAM basename when possible)
  --scope_id ID              naming scope for dataset/default paths, e.g. chr20_chr21
  --regions_id ID            alias of --scope_id for backward compatibility
  --coverage_tag TAG         e.g. harmonized_40x
  --dataset_id ID            override computed dataset_id
  --mode training|calling    default: training

Inputs:
  --ref PATH                 default: data/2_reference/.../GRCh38_no_alt_plus_hs38d1.<scope_id>.fa
  --truth_vcf PATH           required in training mode; default: data/3_truth/<sample>/<sample>_<scope_id>.truth.vcf.gz
  --confident_bed PATH       required in training mode; default: data/3_truth/<sample>/<sample>_<scope_id>.confident.bed

Container runtime:
  --runtime docker|udocker   default: docker
  --image IMAGE              default: google/deepvariant:1.9.0

Execution:
  --shards N                 default: nproc
  --jobs N                   default: min(shards, nproc)
  --resume 0|1               default: 1
  --force 0|1                default: 0
  --out_root DIR             default: data/4_out/deepvariant/examples
  --extra_args "..."         passed to make_examples

Optional validation:
  --validate_labels 0|1      default: 1 in training, 0 in calling
  --validator_py PATH        default: data/5_scripts/analysis/check_tfrecord_has_label.py
  --validator_max_records N  default: 3

Output:
  data/4_out/deepvariant/examples/<dataset_id>/<dv_preset>/<mode>/
EOF_USAGE
}

to_abs() {
  local p="$1"
  [[ "$p" == /* ]] && { echo "$p"; return; }
  echo "$(pwd)/$p"
}

infer_sample_from_bam() {
  local bam_path="$1"
  local base
  base="$(basename "$bam_path")"
  if [[ "$base" =~ ^(HG[0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

container_exists() {
  case "$RUNTIME" in
    docker) command -v docker >/dev/null 2>&1 ;;
    udocker) command -v udocker >/dev/null 2>&1 ;;
    *) return 1 ;;
  esac
}

container_exec() {
  case "$RUNTIME" in
    docker)
      docker run --rm \
        -v "${REPO_ROOT}:${REPO_ROOT}" \
        -w "${REPO_ROOT}" \
        "$IMAGE" "$@"
      ;;
    udocker)
      udocker run \
        --volume="${REPO_ROOT}:${REPO_ROOT}" \
        --workdir="${REPO_ROOT}" \
        "$IMAGE" "$@"
      ;;
    *)
      die "Unsupported runtime: ${RUNTIME}"
      ;;
  esac
}

write_manifest_json() {
  python3 - "$MANIFEST" <<'PY'
import json
import os
import sys

out = sys.argv[1]
obj = {
    "dataset_id": os.environ["DATASET_ID"],
    "sample": os.environ["SAMPLE"],
    "scope_id": os.environ.get("SCOPE_ID") or None,
    "coverage_tag": os.environ.get("COVERAGE_TAG", ""),
    "tech": os.environ["TECH"],
    "dv_preset": os.environ["DV_PRESET"],
    "mode": os.environ["MODE"],
    "bam": os.environ["BAM"],
    "ref": os.environ["REF"],
    "truth_vcf": os.environ.get("TRUTH_VCF") or None,
    "confident_bed": os.environ.get("CONF_BED") or None,
    "runtime": os.environ["RUNTIME"],
    "image": os.environ["IMAGE"],
    "shards": int(os.environ["SHARDS"]),
    "jobs": int(os.environ["JOBS"]),
    "resume": os.environ["RESUME"] == "1",
    "force": os.environ["FORCE"] == "1",
    "channel_list": os.environ["CHANNEL_LIST"],
    "pileup_image_width": int(os.environ["PILEUP_WIDTH"]),
    "preset_extra": os.environ.get("ALT_ALIGNED_ARGS", ""),
    "user_extra_args": os.environ.get("EXTRA_ARGS", ""),
    "validate_labels": os.environ.get("VALIDATE_LABELS", "0") == "1",
    "validator_py": os.environ.get("VALIDATOR_PY") or None,
    "validator_max_records": int(os.environ.get("VALIDATOR_MAX_RECORDS", "0") or 0),
    "git_head": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or None,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
PY
}

check_label_presence() {
  local glob_path="$1"
  local out_txt="$2"
  local status_txt="$3"

  if [[ ! -f "$VALIDATOR_PY" ]]; then
    {
      echo "validator_found=False"
      echo "validation_ok=False"
      echo "reason=validator_script_missing"
      echo "validator_py=$VALIDATOR_PY"
    } > "$out_txt"
    echo "warning" > "$status_txt"
    return 0
  fi

  local require_label=0
  if [[ "$MODE" == "training" ]]; then
    require_label=1
  fi

  if container_exec python3 "$VALIDATOR_PY" \
      --glob "$glob_path" \
      --max_records "$VALIDATOR_MAX_RECORDS" \
      --require_label "$require_label" \
      --expected_width "$PILEUP_WIDTH" \
      > "$out_txt" 2>&1; then
    echo "ok" > "$status_txt"
  else
    {
      echo
      echo "validation_ok=False"
      echo "note=Validation failed but make_examples generation completed successfully. Review this file."
    } >> "$out_txt"
    echo "warning" > "$status_txt"
  fi
  return 0
}

launch_tasks() {
  local running=0
  local fail=0
  local task

  for task in $(seq 0 $((SHARDS - 1))); do
    run_task "$task" &
    running=$((running + 1))

    if (( running >= JOBS )); then
      if ! wait -n; then
        fail=1
      fi
      running=$((running - 1))
    fi
  done

  while (( running > 0 )); do
    if ! wait -n; then
      fail=1
    fi
    running=$((running - 1))
  done

  [[ "$fail" == "0" ]] || die "At least one shard task failed. Check logs in ${LOG_DIR}"
}

run_task() {
  local task="$1"
  local shard log
  shard="$(printf '%s-%05d-of-%05d.gz' "$EX_PREFIX" "$task" "$SHARDS")"
  log="${LOG_DIR}/make_examples.task_${task}.log"

  if [[ "$RESUME" == "1" && -s "$shard" && -s "${shard}.example_info.json" ]]; then
    echo "[task ${task}] skip (resume)"
    return 0
  fi

  local -a cmd
  cmd=(
    /opt/deepvariant/bin/make_examples
    --mode "$MODE"
    --ref "$REF"
    --reads "$BAM"
    --examples "$EXAMPLES_SPEC"
    --task "$task"
    --sample_name "$SAMPLE"
    --channel_list "$CHANNEL_LIST"
    --pileup_image_width "$PILEUP_WIDTH"
    --logtostderr
  )

  if [[ "$MODE" == "training" ]]; then
    cmd+=(
      --truth_variants "$TRUTH_VCF"
      --confident_regions "$CONF_BED"
    )
  fi

  if [[ -n "$ALT_ALIGNED_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_preset=( $ALT_ALIGNED_ARGS )
    cmd+=("${extra_preset[@]}")
  fi

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local user_extra=( $EXTRA_ARGS )
    cmd+=("${user_extra[@]}")
  fi

  container_exec "${cmd[@]}" > "$log" 2>&1
}

# ---- defaults ----
TECH=""
BAM=""
SAMPLE=""
SCOPE_ID=""
COVERAGE_TAG=""
DATASET_ID=""
MODE="training"

RUNTIME="docker"
IMAGE="google/deepvariant:1.9.0"

SHARDS="$(nproc)"
JOBS=""
RESUME=1
FORCE=0
OUT_ROOT="data/4_out/deepvariant/examples"
EXTRA_ARGS=""

REF=""
TRUTH_VCF=""
CONF_BED=""

VALIDATE_LABELS=""
VALIDATOR_PY="data/5_scripts/analysis/check_tfrecord_has_label.py"
VALIDATOR_MAX_RECORDS=3

# ---- args ----
[[ $# -eq 0 ]] && { usage; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tech) TECH="$2"; shift 2 ;;
    --bam) BAM="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --scope_id) SCOPE_ID="$2"; shift 2 ;;
    --regions_id) SCOPE_ID="$2"; shift 2 ;;  # backward compatibility
    --coverage_tag) COVERAGE_TAG="$2"; shift 2 ;;
    --dataset_id) DATASET_ID="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;

    --ref) REF="$2"; shift 2 ;;
    --truth_vcf) TRUTH_VCF="$2"; shift 2 ;;
    --confident_bed) CONF_BED="$2"; shift 2 ;;

    --runtime) RUNTIME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;

    --shards) SHARDS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --extra_args) EXTRA_ARGS="$2"; shift 2 ;;

    --validate_labels) VALIDATE_LABELS="$2"; shift 2 ;;
    --validator_py) VALIDATOR_PY="$2"; shift 2 ;;
    --validator_max_records) VALIDATOR_MAX_RECORDS="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

# ---- validate ----
[[ -n "$TECH" ]] || die "Missing --tech"
[[ "$TECH" == "illumina" || "$TECH" == "ont" ]] || die "--tech must be illumina|ont"
[[ -n "$BAM" ]] || die "Missing --bam"
[[ "$MODE" == "training" || "$MODE" == "calling" ]] || die "--mode must be training|calling"
[[ "$RUNTIME" == "docker" || "$RUNTIME" == "udocker" ]] || die "--runtime must be docker|udocker"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || die "--resume must be 0|1"
[[ "$FORCE" == "0" || "$FORCE" == "1" ]] || die "--force must be 0|1"
container_exists || die "${RUNTIME} not found in PATH"

REPO_ROOT="$(pwd)"
BAM="$(to_abs "$BAM")"
OUT_ROOT="$(to_abs "$OUT_ROOT")"
VALIDATOR_PY="$(to_abs "$VALIDATOR_PY")"
[[ -f "$BAM" ]] || die "BAM not found: $BAM"

if [[ -z "$SAMPLE" ]]; then
  SAMPLE="$(infer_sample_from_bam "$BAM" || true)"
fi
[[ -n "$SAMPLE" ]] || die "Could not infer sample from BAM. Please pass --sample."

if [[ -z "$DATASET_ID" ]]; then
  [[ -n "$SCOPE_ID" ]] || die "Missing --scope_id (or --regions_id) when --dataset_id is not provided."
  DATASET_ID="$(echo "$SAMPLE" | tr '[:upper:]' '[:lower:]')_${SCOPE_ID}"
  if [[ -n "$COVERAGE_TAG" ]]; then
    DATASET_ID+="_${COVERAGE_TAG}"
  fi
fi

if [[ -z "$REF" ]]; then
  [[ -n "$SCOPE_ID" ]] || die "Missing --scope_id (or --regions_id): needed to resolve default --ref."
  REF="data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.${SCOPE_ID}.fa"
fi
REF="$(to_abs "$REF")"
[[ -f "$REF" ]] || die "REF not found: $REF"
[[ -f "${REF}.fai" ]] || die "Missing FASTA index: ${REF}.fai"
[[ -f "${BAM}.bai" || -f "${BAM%.bam}.bai" ]] || die "Missing BAM index next to BAM (.bai): $BAM"

if [[ "$MODE" == "training" ]]; then
  if [[ -z "$TRUTH_VCF" ]]; then
    [[ -n "$SCOPE_ID" ]] || die "Missing --scope_id (or --regions_id): needed to resolve default --truth_vcf."
    TRUTH_VCF="data/3_truth/${SAMPLE}/${SAMPLE}_${SCOPE_ID}.truth.vcf.gz"
  fi
  if [[ -z "$CONF_BED" ]]; then
    [[ -n "$SCOPE_ID" ]] || die "Missing --scope_id (or --regions_id): needed to resolve default --confident_bed."
    CONF_BED="data/3_truth/${SAMPLE}/${SAMPLE}_${SCOPE_ID}.confident.bed"
  fi
  TRUTH_VCF="$(to_abs "$TRUTH_VCF")"
  CONF_BED="$(to_abs "$CONF_BED")"
  [[ -f "$TRUTH_VCF" ]] || die "TRUTH_VCF not found: $TRUTH_VCF"
  [[ -f "$CONF_BED" ]] || die "CONFIDENT_BED not found: $CONF_BED"
  [[ -f "${TRUTH_VCF}.tbi" ]] || die "Missing TRUTH_VCF index (.tbi): ${TRUTH_VCF}.tbi"
else
  TRUTH_VCF=""
  CONF_BED=""
fi

if [[ -z "$JOBS" ]]; then
  NP="$(nproc)"
  if (( SHARDS < NP )); then JOBS="$SHARDS"; else JOBS="$NP"; fi
fi

if [[ -z "$VALIDATE_LABELS" ]]; then
  if [[ "$MODE" == "training" ]]; then
    VALIDATE_LABELS=1
  else
    VALIDATE_LABELS=0
  fi
fi
[[ "$VALIDATE_LABELS" == "0" || "$VALIDATE_LABELS" == "1" ]] || die "--validate_labels must be 0|1"

# ---- preset-specific config ----
CHANNEL_LIST=""
PILEUP_WIDTH=""
ALT_ALIGNED_ARGS=""
DV_PRESET=""

if [[ "$TECH" == "illumina" ]]; then
  DV_PRESET="illumina_wgs"
  CHANNEL_LIST="BASE_CHANNELS,insert_size"
  PILEUP_WIDTH="221"
else
  DV_PRESET="ont_r104"
  CHANNEL_LIST="BASE_CHANNELS,haplotype"
  PILEUP_WIDTH="99"
  ALT_ALIGNED_ARGS="--alt_aligned_pileup=diff_channels"
fi

RUN_DIR="${OUT_ROOT}/${DATASET_ID}/${DV_PRESET}/${MODE}"
LOG_DIR="${RUN_DIR}/logs"
MANIFEST="${RUN_DIR}/manifest.json"
SUCCESS_MARK="${RUN_DIR}/SUCCESS"
LABEL_CHECK_TXT="${RUN_DIR}/label_check.txt"
LABEL_CHECK_STATUS="${RUN_DIR}/label_check.status"
EX_PREFIX="${RUN_DIR}/make_examples.tfrecord"
EXAMPLES_SPEC="${EX_PREFIX}@${SHARDS}.gz"

if [[ -f "$SUCCESS_MARK" && "$FORCE" == "0" ]]; then
  echo "==> SUCCESS already present: ${RUN_DIR}"
  echo "Nothing to do. Use --force 1 to rebuild."
  exit 0
fi

if [[ "$FORCE" == "1" && -d "$RUN_DIR" ]]; then
  echo "==> Removing existing run dir due to --force 1: ${RUN_DIR}"
  rm -rf "$RUN_DIR"
fi

mkdir -p "$LOG_DIR"
rm -f "$SUCCESS_MARK"

export DATASET_ID SAMPLE SCOPE_ID COVERAGE_TAG TECH DV_PRESET MODE
export BAM REF TRUTH_VCF CONF_BED RUNTIME IMAGE SHARDS JOBS RESUME FORCE
export CHANNEL_LIST PILEUP_WIDTH ALT_ALIGNED_ARGS EXTRA_ARGS
export VALIDATE_LABELS VALIDATOR_PY VALIDATOR_MAX_RECORDS

write_manifest_json

echo "==> RUN_DIR: ${RUN_DIR}"
echo "==> DATASET_ID: ${DATASET_ID}"
echo "==> SCOPE_ID: ${SCOPE_ID:-<none>}"
echo "==> DV_PRESET: ${DV_PRESET}"
echo "==> MODE: ${MODE}"
echo "==> EXAMPLES: ${EXAMPLES_SPEC}"
echo "==> Launching ${SHARDS} tasks with concurrency=${JOBS}"

launch_tasks

echo "==> Verifying outputs..."
missing=0
for t in $(seq 0 $((SHARDS - 1))); do
  s="$(printf '%s-%05d-of-%05d.gz' "$EX_PREFIX" "$t" "$SHARDS")"
  [[ -s "$s" ]] || { echo "Missing shard: $s"; missing=1; }
  [[ -s "${s}.example_info.json" ]] || { echo "Missing example_info: ${s}.example_info.json"; missing=1; }
done
[[ "$missing" == "0" ]] || die "Some shards are missing. Check logs in ${LOG_DIR}"

if [[ "$VALIDATE_LABELS" == "1" ]]; then
  echo "==> Running optional TFRecord sanity check (non-blocking)..."
  check_label_presence "${EX_PREFIX}-*-of-*.gz" "$LABEL_CHECK_TXT" "$LABEL_CHECK_STATUS"
  if [[ -f "$LABEL_CHECK_STATUS" && "$(cat "$LABEL_CHECK_STATUS")" != "ok" ]]; then
    warn "TFRecord sanity check did not complete successfully. Review ${LABEL_CHECK_TXT}."
  fi
fi

echo "ok" > "$SUCCESS_MARK"

echo "==> SUCCESS"
echo "Artifacts frozen at: ${RUN_DIR}"