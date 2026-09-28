#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

usage() {
  cat >&2 <<'EOF'
Run full DeepVariant pipeline and produce VCF/gVCF.

Required:
  --dataset_id ID
  --tech illumina|ont
  --bam PATH

Optional:
  --sample HG005
  --scope_id chr20_chr21
  --ref PATH
  --runtime docker|udocker        default: docker
  --image IMAGE                  default: google/deepvariant:1.9.0
  --threads N                    default: nproc
  --out_root DIR                 default: data/4_out/deepvariant/full_calls
  --force 0|1                    default: 0
  --clean_intermediate 0|1       default: 1
  --extra_args "..."             extra args passed to run_deepvariant

Output:
  data/4_out/deepvariant/full_calls/<dataset_id>/<dv_preset>/output.vcf.gz
  data/4_out/deepvariant/full_calls/<dataset_id>/<dv_preset>/output.g.vcf.gz
EOF
}

to_abs() {
  local p="$1"
  [[ "$p" == /* ]] && { echo "$p"; return; }
  echo "$(pwd)/$p"
}

infer_sample_from_bam() {
  local base
  base="$(basename "$1")"
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
      if [[ "$PULL_IMAGE" == "1" ]]; then
        docker pull "$IMAGE"
      fi
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

write_config_json() {
  python3 - "$CONFIG_JSON" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

out = sys.argv[1]
obj = {
    "tool": "deepvariant",
    "mode": "full_run_deepvariant",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_id": os.environ["DATASET_ID"],
    "sample": os.environ["SAMPLE"],
    "scope_id": os.environ.get("SCOPE_ID") or None,
    "tech": os.environ["TECH"],
    "dv_preset": os.environ["DV_PRESET"],
    "model_type": os.environ["MODEL_TYPE"],
    "bam": os.environ["BAM"],
    "ref": os.environ["REF"],
    "runtime": os.environ["RUNTIME"],
    "image": os.environ["IMAGE"],
    "threads": int(os.environ["THREADS"]),
    "out_dir": os.environ["RUN_DIR"],
    "out_vcf": os.environ["OUT_VCF"],
    "out_gvcf": os.environ["OUT_GVCF"],
    "intermediate_results_dir": os.environ["INT_DIR"],
    "logging_dir": os.environ["LOG_DIR"],
    "extra_args": os.environ.get("EXTRA_ARGS", ""),
    "git_head": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or None,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
PY
}

DATASET_ID=""
TECH=""
BAM=""
SAMPLE=""
SCOPE_ID="chr20_chr21"
REF=""
RUNTIME="docker"
IMAGE="google/deepvariant:1.9.0"
PULL_IMAGE=0
THREADS="$(nproc)"
OUT_ROOT="data/4_out/deepvariant/full_calls"
FORCE=0
CLEAN_INTERMEDIATE=1
EXTRA_ARGS=""

[[ $# -eq 0 ]] && { usage; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_id) DATASET_ID="$2"; shift 2 ;;
    --tech) TECH="$2"; shift 2 ;;
    --bam) BAM="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --scope_id|--regions_id) SCOPE_ID="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --pull_image) PULL_IMAGE="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    --clean_intermediate) CLEAN_INTERMEDIATE="$2"; shift 2 ;;
    --extra_args) EXTRA_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$DATASET_ID" ]] || die "Missing --dataset_id"
[[ "$TECH" == "illumina" || "$TECH" == "ont" ]] || die "--tech must be illumina|ont"
[[ -n "$BAM" ]] || die "Missing --bam"
[[ "$RUNTIME" == "docker" || "$RUNTIME" == "udocker" ]] || die "--runtime must be docker|udocker"
[[ "$FORCE" == "0" || "$FORCE" == "1" ]] || die "--force must be 0|1"
[[ "$CLEAN_INTERMEDIATE" == "0" || "$CLEAN_INTERMEDIATE" == "1" ]] || die "--clean_intermediate must be 0|1"
[[ "$PULL_IMAGE" == "0" || "$PULL_IMAGE" == "1" ]] || die "--pull_image must be 0|1"

container_exists || die "${RUNTIME} not found in PATH"

REPO_ROOT="$(pwd)"
BAM="$(to_abs "$BAM")"
OUT_ROOT="$(to_abs "$OUT_ROOT")"

if [[ -z "$SAMPLE" ]]; then
  SAMPLE="$(infer_sample_from_bam "$BAM" || true)"
fi
[[ -n "$SAMPLE" ]] || die "Could not infer sample from BAM. Please pass --sample."

if [[ -z "$REF" ]]; then
  REF="data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.${SCOPE_ID}.fa"
fi
REF="$(to_abs "$REF")"

[[ -f "$BAM" ]] || die "BAM not found: $BAM"
[[ -f "${BAM}.bai" || -f "${BAM%.bam}.bai" ]] || die "BAM index not found next to: $BAM"
[[ -f "$REF" ]] || die "Reference FASTA not found: $REF"
[[ -f "${REF}.fai" ]] || die "Reference FASTA index not found: ${REF}.fai"

if [[ "$TECH" == "illumina" ]]; then
  DV_PRESET="illumina_wgs"
  MODEL_TYPE="WGS"
else
  DV_PRESET="ont_r104"
  MODEL_TYPE="ONT_R104"
fi

RUN_DIR="${OUT_ROOT}/${DATASET_ID}/${DV_PRESET}"
LOG_DIR="${RUN_DIR}/logs"
INT_DIR="${RUN_DIR}/intermediate"
OUT_VCF="${RUN_DIR}/output.vcf.gz"
OUT_GVCF="${RUN_DIR}/output.g.vcf.gz"
CONFIG_JSON="${RUN_DIR}/run_config.json"
SUCCESS_MARK="${RUN_DIR}/SUCCESS"

if [[ "$FORCE" == "1" && -d "$RUN_DIR" ]]; then
  echo "==> Removing existing run dir due to FORCE=1: $RUN_DIR"
  rm -rf "$RUN_DIR"
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"

if [[ -f "$SUCCESS_MARK" && -s "$OUT_VCF" && -s "$OUT_GVCF" && "$FORCE" == "0" ]]; then
  echo "==> SUCCESS already present: $RUN_DIR"
  echo "Nothing to do. Use --force 1 to rebuild."
  exit 0
fi

if [[ "$CLEAN_INTERMEDIATE" == "1" ]]; then
  rm -rf "$INT_DIR"
fi
mkdir -p "$INT_DIR"

export DATASET_ID SAMPLE SCOPE_ID TECH DV_PRESET MODEL_TYPE
export BAM REF RUNTIME IMAGE THREADS RUN_DIR OUT_VCF OUT_GVCF INT_DIR LOG_DIR EXTRA_ARGS

write_config_json

echo "=== DeepVariant full run ==="
echo "DATASET_ID=$DATASET_ID"
echo "TECH=$TECH"
echo "DV_PRESET=$DV_PRESET"
echo "MODEL_TYPE=$MODEL_TYPE"
echo "SAMPLE=$SAMPLE"
echo "SCOPE_ID=$SCOPE_ID"
echo "BAM=$BAM"
echo "REF=$REF"
echo "THREADS=$THREADS"
echo "RUNTIME=$RUNTIME"
echo "IMAGE=$IMAGE"
echo "RUN_DIR=$RUN_DIR"
echo "OUT_VCF=$OUT_VCF"
echo "OUT_GVCF=$OUT_GVCF"
echo "============================"

cmd=(
  /opt/deepvariant/bin/run_deepvariant
  --model_type="${MODEL_TYPE}"
  --ref="${REF}"
  --reads="${BAM}"
  --output_vcf="${OUT_VCF}"
  --output_gvcf="${OUT_GVCF}"
  --num_shards="${THREADS}"
  --intermediate_results_dir="${INT_DIR}"
  --logging_dir="${LOG_DIR}"
  --vcf_stats_report=true
)

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra=( $EXTRA_ARGS )
  cmd+=("${extra[@]}")
fi

printf 'RUN CMD: '
printf '%q ' "${cmd[@]}"
echo

container_exec "${cmd[@]}"

[[ -s "$OUT_VCF" ]] || die "output VCF was not created or is empty: $OUT_VCF"
[[ -s "$OUT_GVCF" ]] || die "output gVCF was not created or is empty: $OUT_GVCF"

gzip -t "$OUT_VCF"
gzip -t "$OUT_GVCF"

echo "ok" > "$SUCCESS_MARK"

echo "==> SUCCESS"
echo "VCF : $OUT_VCF"
echo "gVCF: $OUT_GVCF"
