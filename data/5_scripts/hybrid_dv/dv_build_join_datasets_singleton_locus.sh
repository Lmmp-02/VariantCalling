#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Canonical singleton-locus multimodal join runner for the VariantCalling repo.
#
# Join policy:
#   join_policy_id = singleton_locus
#   dataset_subdir = outer
#   unit_of_fusion = locus
#
# Canonical unimodal inputs:
#   data/4_out/datasets/unimodal/<dataset_id>/illumina_wgs/<mode>/
#   data/4_out/datasets/unimodal/<dataset_id>/ont_r104/<mode>/
#
# Canonical multimodal outputs:
#   data/4_out/datasets/multimodal/by_subject/<dataset_id>/inner/
#   data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/
#
# Condition-local manifests:
#   data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/join_manifest.json
#   data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/join_policy.json
#
# Note:
# - Despite the folder name `outer/`, this is not a full candidate-preserving
#   outer join.
# - A locus is retained only if it has at most one Illumina candidate and at most
#   one ONT candidate.
# - Non-singleton loci are excluded before training/evaluation.
# -----------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

usage() {
  cat >&2 <<'EOF_USAGE'
Canonical singleton-locus multimodal join runner.

Required:
  --dataset_id ID              e.g. hg002_chr20_chr21_harmonized_40x

Optional:
  --mode training|calling      default: training
  --unimodal_root DIR          default: data/4_out/datasets/unimodal
  --out_root DIR               default: data/4_out/datasets/multimodal/by_subject

Join settings:
  --build_inner 0|1            default: 0
  --build_outer 0|1            default: 1
  --chunk_size N               default: 40000
  --drop_ambiguous_bimodal 0|1 default: 1
  --include_teacher 0|1        default: 1
  --write_meta_csv 0|1         default: 0
  --progress_every N           default: 2000
  --debug_shards 0|1           default: 0
  --include_vcf_meta 0|1       default: 1
                               if 1, propagates VCF metadata when available
  --vcf_meta_source POLICY     default: prefer_ill
                               one of: prefer_ill|prefer_ont|ill|ont
  --drop_vcf_key_mismatch 0|1  default: 0
                               if 1, drops bimodal rows where Illumina/ONT variant_key differ

Execution control:
  --force 0|1                  default: 0
                               if 1, removes existing multimodal output dir first

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

write_build_manifest_json() {
  local manifest_path="$1"
  local dataset_subdir="$2"
  local condition_dir="$3"

  MANIFEST_OUT="$manifest_path" \
  MANIFEST_DATASET_SUBDIR="$dataset_subdir" \
  MANIFEST_CONDITION_DIR="$condition_dir" \
  python3 - <<'PY'
import json
import os
from datetime import datetime, timezone

out = os.environ["MANIFEST_OUT"]
dataset_subdir = os.environ["MANIFEST_DATASET_SUBDIR"]
condition_dir = os.environ["MANIFEST_CONDITION_DIR"]

obj = {
    "manifest_type": "multimodal_join_build_manifest",
    "manifest_version": "2.0",
    "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),

    "join_policy_id": "singleton_locus",
    "dataset_subdir": dataset_subdir,

    "dataset_id": os.environ["DATASET_ID"],
    "mode": os.environ["MODE"],

    "unimodal_root": os.environ["UNIMODAL_ROOT"],
    "out_root": os.environ["OUT_ROOT"],
    "condition_dir": condition_dir,

    "ill_glob": os.environ["ILL_GLOB"],
    "ont_glob": os.environ["ONT_GLOB"],
    "inner_dir": os.environ["INNER_DIR"],
    "outer_dir": os.environ["OUTER_DIR"],
    "prefix": os.environ["PREFIX"],

    "build_inner": os.environ["BUILD_INNER"] == "1",
    "build_outer": os.environ["BUILD_OUTER"] == "1",
    "chunk_size": int(os.environ["CHUNK_SIZE"]),
    "drop_ambiguous_bimodal": os.environ["DROP_AMBIGUOUS_BIMODAL"] == "1",
    "include_teacher": os.environ["INCLUDE_TEACHER"] == "1",
    "write_meta_csv": os.environ["WRITE_META_CSV"] == "1",
    "progress_every": int(os.environ["PROGRESS_EVERY"]),
    "debug_shards": os.environ["DEBUG_SHARDS"] == "1",
    "include_vcf_meta": os.environ["INCLUDE_VCF_META"] == "1",
    "vcf_meta_source": os.environ["VCF_META_SOURCE"],
    "drop_vcf_key_mismatch": os.environ["DROP_VCF_KEY_MISMATCH"] == "1",
    "force": os.environ["FORCE"] == "1",

    "git_head": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or None,

    "interpretation_note": (
        "This build manifest records execution parameters. "
        "For methodological interpretation, see join_policy.json in the same folder."
    ),
}

os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
PY
}

write_join_policy_json() {
  local policy_path="$1"
  local dataset_subdir="$2"

  POLICY_OUT="$policy_path" \
  POLICY_DATASET_SUBDIR="$dataset_subdir" \
  python3 - <<'PY'
import json
import os
from datetime import datetime, timezone

out = os.environ["POLICY_OUT"]
dataset_subdir = os.environ["POLICY_DATASET_SUBDIR"]
is_outer = dataset_subdir == "outer"

obj = {
    "manifest_type": "multimodal_join_policy",
    "manifest_version": "1.0",
    "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),

    "dataset_id": os.environ["DATASET_ID"],
    "dataset_family": "multimodal_by_subject",
    "dataset_subdir": dataset_subdir,

    "join_policy_id": "singleton_locus",
    "join_policy_label": "Singleton-locus multimodal join",
    "status": "active_baseline_condition",

    "unit_of_fusion": "locus",
    "join_type": "outer_over_singleton_loci" if is_outer else "inner_over_singleton_loci",

    "retention_policy": {
        "keeps_illumina_only": bool(is_outer),
        "keeps_ont_only": bool(is_outer),
        "keeps_bimodal": True,
        "keeps_non_singleton_loci": False,
        "singleton_definition": (
            "A locus is retained only if it has at most one Illumina candidate "
            "and at most one ONT candidate."
        ),
        "non_singleton_locus_action": "drop_entire_locus",
    },

    "matching_policy": {
        "primary_key": "locus",
        "variant_key_used_for_fusion": False,
        "multi_candidate_loci_supported": False,
        "multi_alt_candidates_supported": False,
    },

    "ambiguity_policy": {
        "drop_ambiguous_bimodal": os.environ["DROP_AMBIGUOUS_BIMODAL"] == "1",
        "ambiguous_if_label_mismatch": os.environ["DROP_AMBIGUOUS_BIMODAL"] == "1",
        "ambiguous_if_variant_type_mismatch": os.environ["DROP_AMBIGUOUS_BIMODAL"] == "1",
        "action": (
            "drop_ambiguous_bimodal_rows"
            if os.environ["DROP_AMBIGUOUS_BIMODAL"] == "1"
            else "retain_ambiguous_bimodal_rows"
        ),
    },

    "vcf_policy": {
        "include_vcf_meta": os.environ["INCLUDE_VCF_META"] == "1",
        "vcf_meta_source": os.environ["VCF_META_SOURCE"],
        "drop_vcf_key_mismatch": os.environ["DROP_VCF_KEY_MISMATCH"] == "1",
        "variant_key_used_for_join": False,
        "variant_key_used_only_as_metadata": os.environ["INCLUDE_VCF_META"] == "1",
    },

    "build_manifest": {
        "path": "join_manifest.json",
        "location": "condition_subdir",
    },

    "intended_use": [
        "controlled head-level evaluation",
        "transfer learning validation on a restricted candidate universe",
        "late-fusion multimodal benchmark under singleton-locus constraints",
        "singleton-constrained VCF/hap.py evaluation",
    ],

    "not_intended_use": [
        "claiming full candidate-universe recall",
        "claiming complete biological variant coverage",
        "directly interpreting hap.py recall as head-only performance",
    ],

    "interpretation_note": (
        "This dataset is valid as a controlled singleton-locus benchmark. "
        "Despite the folder name outer/, it should not be interpreted as preserving "
        "the full exported DeepVariant candidate universe. Loci with multiple candidates "
        "in either modality are excluded before training/evaluation."
        if is_outer
        else
        "This inner dataset is restricted to loci present as singleton candidates in both modalities. "
        "It is useful for controlled bimodal analysis, but not for candidate-universe recall claims."
    ),
}

os.makedirs(os.path.dirname(out), exist_ok=True)
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
CHUNK_SIZE=40000
DROP_AMBIGUOUS_BIMODAL=1
INCLUDE_TEACHER=1
WRITE_META_CSV=0
PROGRESS_EVERY=2000
DEBUG_SHARDS=0
FORCE=0
INCLUDE_VCF_META=1
VCF_META_SOURCE="prefer_ill"
DROP_VCF_KEY_MISMATCH=0

JOIN_POLICY_ID="singleton_locus"
OUTER_DATASET_SUBDIR="outer"
INNER_DATASET_SUBDIR="inner"

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
    --drop_ambiguous_bimodal) DROP_AMBIGUOUS_BIMODAL="$2"; shift 2 ;;
    --include_teacher) INCLUDE_TEACHER="$2"; shift 2 ;;
    --write_meta_csv) WRITE_META_CSV="$2"; shift 2 ;;
    --progress_every) PROGRESS_EVERY="$2"; shift 2 ;;
    --debug_shards) DEBUG_SHARDS="$2"; shift 2 ;;
    --include_vcf_meta) INCLUDE_VCF_META="$2"; shift 2 ;;
    --vcf_meta_source) VCF_META_SOURCE="$2"; shift 2 ;;
    --drop_vcf_key_mismatch) DROP_VCF_KEY_MISMATCH="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$DATASET_ID" ]] || die "Missing --dataset_id"
[[ "$MODE" == "training" || "$MODE" == "calling" ]] || die "--mode must be training|calling"
[[ "$BUILD_INNER" == "0" || "$BUILD_INNER" == "1" ]] || die "--build_inner must be 0|1"
[[ "$BUILD_OUTER" == "0" || "$BUILD_OUTER" == "1" ]] || die "--build_outer must be 0|1"
[[ "$DROP_AMBIGUOUS_BIMODAL" == "0" || "$DROP_AMBIGUOUS_BIMODAL" == "1" ]] || die "--drop_ambiguous_bimodal must be 0|1"
[[ "$INCLUDE_TEACHER" == "0" || "$INCLUDE_TEACHER" == "1" ]] || die "--include_teacher must be 0|1"
[[ "$WRITE_META_CSV" == "0" || "$WRITE_META_CSV" == "1" ]] || die "--write_meta_csv must be 0|1"
[[ "$DEBUG_SHARDS" == "0" || "$DEBUG_SHARDS" == "1" ]] || die "--debug_shards must be 0|1"
[[ "$FORCE" == "0" || "$FORCE" == "1" ]] || die "--force must be 0|1"
(( BUILD_INNER == 1 || BUILD_OUTER == 1 )) || die "Nothing to build."
[[ "$INCLUDE_VCF_META" == "0" || "$INCLUDE_VCF_META" == "1" ]] || die "--include_vcf_meta must be 0|1"
[[ "$VCF_META_SOURCE" == "prefer_ill" || "$VCF_META_SOURCE" == "prefer_ont" || "$VCF_META_SOURCE" == "ill" || "$VCF_META_SOURCE" == "ont" ]] || die "--vcf_meta_source must be prefer_ill|prefer_ont|ill|ont"
[[ "$DROP_VCF_KEY_MISMATCH" == "0" || "$DROP_VCF_KEY_MISMATCH" == "1" ]] || die "--drop_vcf_key_mismatch must be 0|1"

REPO_ROOT="$(pwd)"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
else
  warn ".venv/bin/activate not found under repo root; using system python."
fi

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
INNER_DIR="${RUN_ROOT}/${INNER_DATASET_SUBDIR}"
OUTER_DIR="${RUN_ROOT}/${OUTER_DATASET_SUBDIR}"
LOG_DIR="${RUN_ROOT}/logs"
PREFIX="${DATASET_ID}"

INNER_MANIFEST="${INNER_DIR}/join_manifest.json"
OUTER_MANIFEST="${OUTER_DIR}/join_manifest.json"
INNER_POLICY="${INNER_DIR}/join_policy.json"
OUTER_POLICY="${OUTER_DIR}/join_policy.json"

SUCCESS_MARK="${RUN_ROOT}/SUCCESS"
JOIN_LOG="${LOG_DIR}/build_join_datasets_${JOIN_POLICY_ID}.log"

if [[ "$FORCE" == "1" && -d "$RUN_ROOT" ]]; then
  echo "==> Removing existing multimodal run dir due to --force 1: ${RUN_ROOT}"
  rm -rf "$RUN_ROOT"
fi

mkdir -p "$LOG_DIR"
rm -f "$SUCCESS_MARK"

export DATASET_ID MODE UNIMODAL_ROOT OUT_ROOT ILL_GLOB ONT_GLOB INNER_DIR OUTER_DIR PREFIX
export BUILD_INNER BUILD_OUTER CHUNK_SIZE DROP_AMBIGUOUS_BIMODAL INCLUDE_TEACHER WRITE_META_CSV PROGRESS_EVERY DEBUG_SHARDS FORCE
export INCLUDE_VCF_META VCF_META_SOURCE DROP_VCF_KEY_MISMATCH
export JOIN_POLICY_ID OUTER_DATASET_SUBDIR INNER_DATASET_SUBDIR

# Write build manifests before running the Python builder.
# If the build fails, the root-level SUCCESS marker is not written.
if [[ "$BUILD_INNER" == "1" ]]; then
  write_build_manifest_json "$INNER_MANIFEST" "$INNER_DATASET_SUBDIR" "$INNER_DIR"
fi
if [[ "$BUILD_OUTER" == "1" ]]; then
  write_build_manifest_json "$OUTER_MANIFEST" "$OUTER_DATASET_SUBDIR" "$OUTER_DIR"
fi

cmd=(
  python3 -u data/5_scripts/hybrid_dv/build_join_datasets_singleton_locus.py
  --ill_glob "$ILL_GLOB"
  --ont_glob "$ONT_GLOB"
  --out_inner_dir "$INNER_DIR"
  --out_outer_dir "$OUTER_DIR"
  --prefix "$PREFIX"
  --build_inner "$BUILD_INNER"
  --build_outer "$BUILD_OUTER"
  --chunk_size "$CHUNK_SIZE"
  --drop_ambiguous_bimodal "$DROP_AMBIGUOUS_BIMODAL"
  --include_teacher "$INCLUDE_TEACHER"
  --write_meta_csv "$WRITE_META_CSV"
  --progress_every "$PROGRESS_EVERY"
  --debug_shards "$DEBUG_SHARDS"
  --include_vcf_meta "$INCLUDE_VCF_META"
  --vcf_meta_source "$VCF_META_SOURCE"
  --drop_vcf_key_mismatch "$DROP_VCF_KEY_MISMATCH"
)

{
  echo "==> RUN_ROOT: ${RUN_ROOT}"
  echo "==> JOIN_POLICY_ID: ${JOIN_POLICY_ID}"
  echo "==> OUTER_DATASET_SUBDIR: ${OUTER_DATASET_SUBDIR}"
  echo "==> INNER_DATASET_SUBDIR: ${INNER_DATASET_SUBDIR}"
  echo "==> DATASET_ID: ${DATASET_ID}"
  echo "==> MODE: ${MODE}"
  echo "==> ILL_GLOB: ${ILL_GLOB}"
  echo "==> ONT_GLOB: ${ONT_GLOB}"
  echo "==> BUILD_INNER: ${BUILD_INNER}"
  echo "==> BUILD_OUTER: ${BUILD_OUTER}"
  echo "==> CHUNK_SIZE: ${CHUNK_SIZE}"
  echo "==> DROP_AMBIGUOUS_BIMODAL: ${DROP_AMBIGUOUS_BIMODAL}"
  echo "==> INCLUDE_TEACHER: ${INCLUDE_TEACHER}"
  echo "==> WRITE_META_CSV: ${WRITE_META_CSV}"
  echo "==> INCLUDE_VCF_META: ${INCLUDE_VCF_META}"
  echo "==> VCF_META_SOURCE: ${VCF_META_SOURCE}"
  echo "==> DROP_VCF_KEY_MISMATCH: ${DROP_VCF_KEY_MISMATCH}"
  echo "==> FORCE: ${FORCE}"
  if [[ "$BUILD_INNER" == "1" ]]; then
    echo "==> INNER_MANIFEST: ${INNER_MANIFEST}"
    echo "==> INNER_POLICY: ${INNER_POLICY}"
  fi
  if [[ "$BUILD_OUTER" == "1" ]]; then
    echo "==> OUTER_MANIFEST: ${OUTER_MANIFEST}"
    echo "==> OUTER_POLICY: ${OUTER_POLICY}"
  fi
  printf '==> RUN CMD: '
  printf '%q ' "${cmd[@]}"
  echo
} | tee "$JOIN_LOG"

"${cmd[@]}" 2>&1 | tee -a "$JOIN_LOG"

if [[ "$BUILD_OUTER" == "1" ]]; then
  compgen -G "${OUTER_DIR}/${PREFIX}_outer_*.npz" > /dev/null || die "No OUTER NPZ shards generated under ${OUTER_DIR}"
  [[ -f "${OUTER_DIR}/${PREFIX}_outer_report.json" ]] || die "Missing OUTER report JSON under ${OUTER_DIR}"
  [[ -f "${OUTER_MANIFEST}" ]] || die "Missing OUTER join_manifest.json under ${OUTER_DIR}"

  write_join_policy_json "$OUTER_POLICY" "$OUTER_DATASET_SUBDIR"
  [[ -f "${OUTER_POLICY}" ]] || die "Missing OUTER join_policy.json under ${OUTER_DIR}"
fi

if [[ "$BUILD_INNER" == "1" ]]; then
  compgen -G "${INNER_DIR}/${PREFIX}_inner_*.npz" > /dev/null || die "No INNER NPZ shards generated under ${INNER_DIR}"
  [[ -f "${INNER_DIR}/${PREFIX}_inner_report.json" ]] || die "Missing INNER report JSON under ${INNER_DIR}"
  [[ -f "${INNER_MANIFEST}" ]] || die "Missing INNER join_manifest.json under ${INNER_DIR}"

  write_join_policy_json "$INNER_POLICY" "$INNER_DATASET_SUBDIR"
  [[ -f "${INNER_POLICY}" ]] || die "Missing INNER join_policy.json under ${INNER_DIR}"
fi

echo "ok" > "$SUCCESS_MARK"

echo "==> SUCCESS" | tee -a "$JOIN_LOG"
echo "Artifacts frozen at: ${RUN_ROOT}" | tee -a "$JOIN_LOG"

if [[ "$BUILD_OUTER" == "1" ]]; then
  echo "Outer manifests:" | tee -a "$JOIN_LOG"
  echo "  ${OUTER_MANIFEST}" | tee -a "$JOIN_LOG"
  echo "  ${OUTER_POLICY}" | tee -a "$JOIN_LOG"
fi
if [[ "$BUILD_INNER" == "1" ]]; then
  echo "Inner manifests:" | tee -a "$JOIN_LOG"
  echo "  ${INNER_MANIFEST}" | tee -a "$JOIN_LOG"
  echo "  ${INNER_POLICY}" | tee -a "$JOIN_LOG"
fi