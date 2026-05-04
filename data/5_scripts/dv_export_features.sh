#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# DeepVariant offline feature export runner for the VariantCalling repo.
#
# Canonical input contract:
#   data/4_out/deepvariant/examples/<dataset_id>/<dv_preset>/<mode>/
#
# Canonical output contract:
#   data/4_out/datasets/unimodal/<dataset_id>/<dv_preset>/<mode>/
#
# Notes:
# - Intended as the canonical unimodal export step after make_examples.
# - Works with both training and calling TFRecords.
# - In training mode, labels are exported if present in TFRecords.
# - In calling mode, export_dv_features.py will emit label=-1.
# -----------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

usage() {
  cat >&2 <<'EOF_USAGE'
DeepVariant offline feature export runner.

Required:
  --dataset_id ID            e.g. hg002_chr20_chr21_harmonized_40x
  --tech illumina|ont

Optional:
  --mode training|calling    default: training
  --examples_dir DIR         override canonical examples dir
  --out_root DIR             default: data/4_out/datasets/unimodal

Container runtime:
  --runtime docker|udocker   default: docker
  --image IMAGE              default: google/deepvariant:1.9.0-gpu

Model/export settings:
  --model_dir DIR            default: /opt/models/wgs or /opt/models/ont_r104
  --batch_size N             default: 128
  --chunk_size N             default: 40000
  --max_records N            optional cap for smoke tests
  --emb_tensor NAME          optional explicit embedding tensor
  --logits_tensor NAME       optional explicit logits tensor
  --emit_identity_meta 0|1   default: 0
                             if 1, export legacy identity metadata:
                             variant_hash, example_id, variant_len
  --emit_vcf_meta 0|1        default: 1
                             if 1, export VCF metadata decoded from variant/encoded:
                             vcf_chrom, vcf_pos, vcf_ref, vcf_alt, variant_key, ...
  --strict_vcf_meta 0|1      default: 1
                             if 1, fail when VCF metadata cannot be decoded

Output:
  data/4_out/datasets/unimodal/<dataset_id>/<dv_preset>/<mode>/
EOF_USAGE
}

to_abs() {
  local p="$1"
  [[ "$p" == /* ]] && { echo "$p"; return; }
  echo "$(pwd)/$p"
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
    "tech": os.environ["TECH"],
    "dv_preset": os.environ["DV_PRESET"],
    "mode": os.environ["MODE"],
    "examples_dir": os.environ["EXAMPLES_DIR"],
    "tfrecord_glob": os.environ["TF_GLOB"],
    "out_dir": os.environ["RUN_DIR"],
    "out_prefix": os.environ["OUT_PREFIX"],
    "runtime": os.environ["RUNTIME"],
    "image": os.environ["IMAGE"],
    "model_dir": os.environ["MODEL_DIR"],
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "chunk_size": int(os.environ["CHUNK_SIZE"]),
    "max_records": int(os.environ["MAX_RECORDS"]) if os.environ.get("MAX_RECORDS") else None,
    "emb_tensor": os.environ.get("EMB_TENSOR") or None,
    "logits_tensor": os.environ.get("LOGITS_TENSOR") or None,
    "emit_identity_meta": os.environ["EMIT_IDENTITY_META"] == "1",
    "emit_vcf_meta": os.environ["EMIT_VCF_META"] == "1",
    "strict_vcf_meta": os.environ["STRICT_VCF_META"] == "1",
    "git_head": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or None,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
PY
}

DATASET_ID=""
TECH=""
MODE="training"
EXAMPLES_DIR=""
OUT_ROOT="data/4_out/datasets/unimodal"

RUNTIME="docker"
IMAGE="google/deepvariant:1.9.0-gpu"
MODEL_DIR=""
BATCH_SIZE=128
CHUNK_SIZE=40000
MAX_RECORDS=""
EMB_TENSOR=""
LOGITS_TENSOR=""
EMIT_IDENTITY_META=0
EMIT_VCF_META=1
STRICT_VCF_META=1

[[ $# -eq 0 ]] && { usage; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_id) DATASET_ID="$2"; shift 2 ;;
    --tech) TECH="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --examples_dir) EXAMPLES_DIR="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --chunk_size) CHUNK_SIZE="$2"; shift 2 ;;
    --max_records) MAX_RECORDS="$2"; shift 2 ;;
    --emb_tensor) EMB_TENSOR="$2"; shift 2 ;;
    --logits_tensor) LOGITS_TENSOR="$2"; shift 2 ;;
    --emit_identity_meta) EMIT_IDENTITY_META="$2"; shift 2 ;;
    --emit_vcf_meta) EMIT_VCF_META="$2"; shift 2 ;;
    --strict_vcf_meta) STRICT_VCF_META="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$DATASET_ID" ]] || die "Missing --dataset_id"
[[ "$TECH" == "illumina" || "$TECH" == "ont" ]] || die "--tech must be illumina|ont"
[[ "$MODE" == "training" || "$MODE" == "calling" ]] || die "--mode must be training|calling"
[[ "$RUNTIME" == "docker" || "$RUNTIME" == "udocker" ]] || die "--runtime must be docker|udocker"
[[ "$EMIT_IDENTITY_META" == "0" || "$EMIT_IDENTITY_META" == "1" ]] || die "--emit_identity_meta must be 0|1"
[[ "$EMIT_VCF_META" == "0" || "$EMIT_VCF_META" == "1" ]] || die "--emit_vcf_meta must be 0|1"
[[ "$STRICT_VCF_META" == "0" || "$STRICT_VCF_META" == "1" ]] || die "--strict_vcf_meta must be 0|1"
container_exists || die "${RUNTIME} not found in PATH"

REPO_ROOT="$(pwd)"
OUT_ROOT="$(to_abs "$OUT_ROOT")"

if [[ "$TECH" == "illumina" ]]; then
  DV_PRESET="illumina_wgs"
  DEFAULT_MODEL_DIR="/opt/models/wgs"
else
  DV_PRESET="ont_r104"
  DEFAULT_MODEL_DIR="/opt/models/ont_r104"
fi
MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"

if [[ -z "$EXAMPLES_DIR" ]]; then
  EXAMPLES_DIR="data/4_out/deepvariant/examples/${DATASET_ID}/${DV_PRESET}/${MODE}"
fi
EXAMPLES_DIR="$(to_abs "$EXAMPLES_DIR")"
[[ -d "$EXAMPLES_DIR" ]] || die "Examples dir not found: $EXAMPLES_DIR"
[[ -f "${EXAMPLES_DIR}/SUCCESS" ]] || warn "SUCCESS not found in ${EXAMPLES_DIR}; proceeding anyway"

TF_GLOB="${EXAMPLES_DIR}/make_examples.tfrecord-*-of-*.gz"
compgen -G "$TF_GLOB" > /dev/null || die "No TFRecords matched: $TF_GLOB"

RUN_DIR="${OUT_ROOT}/${DATASET_ID}/${DV_PRESET}/${MODE}"
LOG_DIR="${RUN_DIR}/logs"
MANIFEST="${RUN_DIR}/export_manifest.json"
SUCCESS_MARK="${RUN_DIR}/SUCCESS"
OUT_PREFIX="${RUN_DIR}/${DATASET_ID}.${DV_PRESET}.${MODE}"
EXPORT_LOG="${LOG_DIR}/export_features.log"

mkdir -p "$LOG_DIR"
rm -f "$SUCCESS_MARK"
rm -f "${OUT_PREFIX}"_*.npz

export DATASET_ID TECH DV_PRESET MODE EXAMPLES_DIR TF_GLOB RUN_DIR OUT_PREFIX
export RUNTIME IMAGE MODEL_DIR BATCH_SIZE CHUNK_SIZE MAX_RECORDS EMB_TENSOR LOGITS_TENSOR EMIT_IDENTITY_META EMIT_VCF_META STRICT_VCF_META

write_manifest_json

cmd=(
  python3 -u data/5_scripts/export_dv_features.py
  --model_dir "$MODEL_DIR"
  --tfrecord_glob "$TF_GLOB"
  --out_prefix "$OUT_PREFIX"
  --batch_size "$BATCH_SIZE"
  --chunk_size "$CHUNK_SIZE"
  --emit_identity_meta "$EMIT_IDENTITY_META"
  --emit_vcf_meta "$EMIT_VCF_META"
  --strict_vcf_meta "$STRICT_VCF_META"
)
[[ -n "$MAX_RECORDS" ]] && cmd+=( --max_records "$MAX_RECORDS" )
[[ -n "$EMB_TENSOR" ]] && cmd+=( --emb_tensor "$EMB_TENSOR" )
[[ -n "$LOGITS_TENSOR" ]] && cmd+=( --logits_tensor "$LOGITS_TENSOR" )

{
  echo "==> RUN_DIR: ${RUN_DIR}"
  echo "==> DATASET_ID: ${DATASET_ID}"
  echo "==> DV_PRESET: ${DV_PRESET}"
  echo "==> MODE: ${MODE}"
  echo "==> TF_GLOB: ${TF_GLOB}"
  echo "==> MODEL_DIR: ${MODEL_DIR}"
  echo "==> BATCH_SIZE: ${BATCH_SIZE}"
  echo "==> CHUNK_SIZE: ${CHUNK_SIZE}"
  echo "==> EMIT_IDENTITY_META: ${EMIT_IDENTITY_META}"
  echo "==> EMIT_VCF_META: ${EMIT_VCF_META}"
  echo "==> STRICT_VCF_META: ${STRICT_VCF_META}"
  printf '==> RUN CMD: '
  printf '%q ' "${cmd[@]}"
  echo
} | tee "$EXPORT_LOG"

container_exec "${cmd[@]}" 2>&1 | tee -a "$EXPORT_LOG"

compgen -G "${OUT_PREFIX}_*.npz" > /dev/null || die "No NPZ shards were generated under prefix: ${OUT_PREFIX}"

echo "ok" > "$SUCCESS_MARK"

echo "==> SUCCESS" | tee -a "$EXPORT_LOG"
echo "Artifacts frozen at: ${RUN_DIR}" | tee -a "$EXPORT_LOG"