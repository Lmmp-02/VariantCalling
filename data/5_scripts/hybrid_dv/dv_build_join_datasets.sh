#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Canonical multimodal join-by-locus runner for the VariantCalling repo.
#
# Canonical unimodal inputs:
#   data/4_out/datasets/unimodal/<dataset_id>/illumina_wgs/<mode>/
#   data/4_out/datasets/unimodal/<dataset_id>/ont_r104/<mode>/
#
# Canonical multimodal outputs:
#   data/4_out/datasets/multimodal/by_subject/<dataset_id>/inner/
#   data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/
#
# Default intent for current phase:
# - build OUTER only
# - preserve subject boundaries
# - keep HG002/HG003/HG004 as separate source-of-truth multimodal datasets
# -----------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

usage() {
  cat >&2 <<'EOF_USAGE'
Canonical multimodal join-by-locus runner.

Required:
  --dataset_id ID              e.g. hg002_chr20_chr21_harmonized_40x

Optional:
  --mode training|calling      default: training
  --unimodal_root DIR          default: data/4_out/datasets/unimodal
  --out_root DIR               default: data/4_out/datasets/multimodal/by_subject

Join settings:
  --build_inner 0|1            default: 0
  --build_outer 0|1            default: 1
  --chunk_size N               default: 20000
  --drop_disagree 0|1          default: 1 (INNER only)
  --include_teacher 0|1        default: 1
  --write_meta_csv 0|1         default: 0
  --progress_every N           default: 2000
  --debug_shards 0|1           default: 0

Output:
  data/4_out/datasets/multimodal/by_subject/<dataset_id>/inner/
  data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/
EOF_USAGE
}

to_abs() {
  local p="$1"
  [[ "$p" == /* ]] && { echo "$p"; return; }
  echo "$(pwd)/$p"
}

write_manifest_json() {
  python3 - "$MANIFEST" <<'PY'
import json
import os
import sys

out = sys.argv[1]
obj = {
    "dataset_id": os.environ["DATASET_ID"],
    "mode": os.environ["MODE"],
    "unimodal_root": os.environ["UNIMODAL_ROOT"],
    "out_root": os.environ["OUT_ROOT"],
    "ill_glob": os.environ["ILL_GLOB"],
    "ont_glob": os.environ["ONT_GLOB"],
    "inner_dir": os.environ["INNER_DIR"],
    "outer_dir": os.environ["OUTER_DIR"],
    "prefix": os.environ["PREFIX"],
    "build_inner": os.environ["BUILD_INNER"] == "1",
    "build_outer": os.environ["BUILD_OUTER"] == "1",
    "chunk_size": int(os.environ["CHUNK_SIZE"]),
    "drop_disagree": os.environ["DROP_DISAGREE"] == "1",
    "include_teacher": os.environ["INCLUDE_TEACHER"] == "1",
    "write_meta_csv": os.environ["WRITE_META_CSV"] == "1",
    "progress_every": int(os.environ["PROGRESS_EVERY"]),
    "debug_shards": os.environ["DEBUG_SHARDS"] == "1",
    "git_head": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or None,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
PY
}

DATASET_ID=""
MODE="training"
UNIMODAL_ROOT="data/4_out/datasets/unimodal"
OUT_ROOT="data/4_out/datasets/multimodal/by_subject"
BUILD_INNER=0
BUILD_OUTER=1
CHUNK_SIZE=20000
DROP_DISAGREE=1
INCLUDE_TEACHER=1
WRITE_META_CSV=0
PROGRESS_EVERY=2000
DEBUG_SHARDS=0

[[ $# -eq 0 ]] && { usage; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_id) DATASET_ID="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --unimodal_root) UNIMODAL_ROOT="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --build_inner) BUILD_INNER="$2"; shift 2 ;;
    --build_outer) BUILD_OUTER="$2"; shift 2 ;;
    --chunk_size) CHUNK_SIZE="$2"; shift 2 ;;
    --drop_disagree) DROP_DISAGREE="$2"; shift 2 ;;
    --include_teacher) INCLUDE_TEACHER="$2"; shift 2 ;;
    --write_meta_csv) WRITE_META_CSV="$2"; shift 2 ;;
    --progress_every) PROGRESS_EVERY="$2"; shift 2 ;;
    --debug_shards) DEBUG_SHARDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$DATASET_ID" ]] || die "Missing --dataset_id"
[[ "$MODE" == "training" || "$MODE" == "calling" ]] || die "--mode must be training|calling"
[[ "$BUILD_INNER" == "0" || "$BUILD_INNER" == "1" ]] || die "--build_inner must be 0|1"
[[ "$BUILD_OUTER" == "0" || "$BUILD_OUTER" == "1" ]] || die "--build_outer must be 0|1"
[[ "$DROP_DISAGREE" == "0" || "$DROP_DISAGREE" == "1" ]] || die "--drop_disagree must be 0|1"
[[ "$INCLUDE_TEACHER" == "0" || "$INCLUDE_TEACHER" == "1" ]] || die "--include_teacher must be 0|1"
[[ "$WRITE_META_CSV" == "0" || "$WRITE_META_CSV" == "1" ]] || die "--write_meta_csv must be 0|1"
[[ "$DEBUG_SHARDS" == "0" || "$DEBUG_SHARDS" == "1" ]] || die "--debug_shards must be 0|1"
(( BUILD_INNER == 1 || BUILD_OUTER == 1 )) || die "Nothing to build."

REPO_ROOT="$(pwd)"
UNIMODAL_ROOT="$(to_abs "$UNIMODAL_ROOT")"
OUT_ROOT="$(to_abs "$OUT_ROOT")"

ILL_DIR="${UNIMODAL_ROOT}/${DATASET_ID}/illumina_wgs/${MODE}"
ONT_DIR="${UNIMODAL_ROOT}/${DATASET_ID}/ont_r104/${MODE}"
[[ -d "$ILL_DIR" ]] || die "Illumina dir not found: $ILL_DIR"
[[ -d "$ONT_DIR" ]] || die "ONT dir not found: $ONT_DIR"
[[ -f "${ILL_DIR}/SUCCESS" ]] || warn "SUCCESS not found in ${ILL_DIR}; proceeding anyway"
[[ -f "${ONT_DIR}/SUCCESS" ]] || warn "SUCCESS not found in ${ONT_DIR}; proceeding anyway"

ILL_GLOB="${ILL_DIR}/${DATASET_ID}.illumina_wgs.${MODE}_*.npz"
ONT_GLOB="${ONT_DIR}/${DATASET_ID}.ont_r104.${MODE}_*.npz"
compgen -G "$ILL_GLOB" > /dev/null || die "No Illumina NPZ matched: $ILL_GLOB"
compgen -G "$ONT_GLOB" > /dev/null || die "No ONT NPZ matched: $ONT_GLOB"

RUN_ROOT="${OUT_ROOT}/${DATASET_ID}"
INNER_DIR="${RUN_ROOT}/inner"
OUTER_DIR="${RUN_ROOT}/outer"
LOG_DIR="${RUN_ROOT}/logs"
PREFIX="${DATASET_ID}"
MANIFEST="${RUN_ROOT}/join_manifest.json"
SUCCESS_MARK="${RUN_ROOT}/SUCCESS"
JOIN_LOG="${LOG_DIR}/build_join_datasets.log"

mkdir -p "$LOG_DIR"
rm -f "$SUCCESS_MARK"

export DATASET_ID MODE UNIMODAL_ROOT OUT_ROOT ILL_GLOB ONT_GLOB INNER_DIR OUTER_DIR PREFIX
export BUILD_INNER BUILD_OUTER CHUNK_SIZE DROP_DISAGREE INCLUDE_TEACHER WRITE_META_CSV PROGRESS_EVERY DEBUG_SHARDS

write_manifest_json

cmd=(
  python3 -u data/5_scripts/hybrid_dv/build_join_datasets.py
  --ill_glob "$ILL_GLOB"
  --ont_glob "$ONT_GLOB"
  --out_inner_dir "$INNER_DIR"
  --out_outer_dir "$OUTER_DIR"
  --prefix "$PREFIX"
  --build_inner "$BUILD_INNER"
  --build_outer "$BUILD_OUTER"
  --chunk_size "$CHUNK_SIZE"
  --drop_disagree "$DROP_DISAGREE"
  --include_teacher "$INCLUDE_TEACHER"
  --write_meta_csv "$WRITE_META_CSV"
  --progress_every "$PROGRESS_EVERY"
  --debug_shards "$DEBUG_SHARDS"
)

{
  echo "==> RUN_ROOT: ${RUN_ROOT}"
  echo "==> DATASET_ID: ${DATASET_ID}"
  echo "==> MODE: ${MODE}"
  echo "==> ILL_GLOB: ${ILL_GLOB}"
  echo "==> ONT_GLOB: ${ONT_GLOB}"
  echo "==> BUILD_INNER: ${BUILD_INNER}"
  echo "==> BUILD_OUTER: ${BUILD_OUTER}"
  echo "==> CHUNK_SIZE: ${CHUNK_SIZE}"
  echo "==> INCLUDE_TEACHER: ${INCLUDE_TEACHER}"
  echo "==> WRITE_META_CSV: ${WRITE_META_CSV}"
  printf '==> RUN CMD: '
  printf '%q ' "${cmd[@]}"
  echo
} | tee "$JOIN_LOG"

"${cmd[@]}" 2>&1 | tee -a "$JOIN_LOG"

if [[ "$BUILD_OUTER" == "1" ]]; then
  compgen -G "${OUTER_DIR}/${PREFIX}_outer_*.npz" > /dev/null || die "No OUTER NPZ shards generated under ${OUTER_DIR}"
fi
if [[ "$BUILD_INNER" == "1" ]]; then
  compgen -G "${INNER_DIR}/${PREFIX}_inner_*.npz" > /dev/null || die "No INNER NPZ shards generated under ${INNER_DIR}"
fi

echo "ok" > "$SUCCESS_MARK"

echo "==> SUCCESS" | tee -a "$JOIN_LOG"
echo "Artifacts frozen at: ${RUN_ROOT}" | tee -a "$JOIN_LOG"
