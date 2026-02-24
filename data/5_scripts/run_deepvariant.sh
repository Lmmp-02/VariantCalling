#!/usr/bin/env bash
set -euo pipefail

cd ~/VariantCalling

# -----------------------
# Defaults (editables)
# -----------------------
DV_VERSION_DEFAULT="1.9.0"
DV_IMAGE="google/deepvariant"
THREADS_DEFAULT="$(nproc)"
SAMPLE_DEFAULT="HG003"
REGION_DEFAULT="chr20"

REF_DEFAULT="data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa"

# Originals
ILLUMINA_ORIG_BAM_DEFAULT="data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam"
ONT_ORIG_BAM_DEFAULT="data/1_input_bams/HG003/ONT/HG003.GRCh38.ONT_R104_sup_PAY87794.chr20.bam"

# Downsample base dir
DOWNSAMPLED_BASE_DEFAULT="data/1_input_bams/HG003/downsampled"

# Tech folder label for ONT outputs (as per your convention)
ONT_TECH_LABEL_DEFAULT="ONT_R104_sup_PAY87794"

# -----------------------
# Args
# -----------------------
TECH=""          # Illumina | ONT
COV="orig"       # orig | 30 | 15 | 10 | 5
SEED="orig"      # orig | 42 | ...
REGION="$REGION_DEFAULT"
DV_VERSION="$DV_VERSION_DEFAULT"
THREADS="$THREADS_DEFAULT"
SAMPLE="$SAMPLE_DEFAULT"
REF="$REF_DEFAULT"
ONT_TECH_LABEL="$ONT_TECH_LABEL_DEFAULT"

usage() {
  cat <<EOF
Usage:
  $0 --tech {Illumina|ONT} [--cov orig|30|15|10|5] [--seed orig|42|...] [--region chr20]
     [--dv_version 1.9.0] [--threads N] [--ref path.fa] [--sample HG003]
     [--ont_label ONT_R104_sup_PAY87794]

Examples:
  $0 --tech Illumina --cov orig
  $0 --tech Illumina --cov 10 --seed 42
  $0 --tech ONT --cov 5 --seed 2026
EOF
}

die(){ echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tech) TECH="$2"; shift 2;;
    --cov) COV="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --dv_version) DV_VERSION="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --ref) REF="$2"; shift 2;;
    --sample) SAMPLE="$2"; shift 2;;
    --ont_label) ONT_TECH_LABEL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "$TECH" ]] || die "Missing --tech"

# -----------------------
# Resolve model + BAM + output tech folder
# -----------------------
MODEL_TYPE=""
BAM=""
OUT_TECH_FOLDER=""

case "$TECH" in
  Illumina)
    MODEL_TYPE="WGS"
    OUT_TECH_FOLDER="Illumina"
    if [[ "$COV" == "orig" ]]; then
      BAM="$ILLUMINA_ORIG_BAM_DEFAULT"
    else
      [[ "$SEED" != "orig" ]] || die "Downsampled cov=$COV requires --seed"
      BAM="${DOWNSAMPLED_BASE_DEFAULT}/ILLUMINA_chr20/HG003.ILLUMINA.chr20.${COV}x.s${SEED}.bam"
    fi
    ;;
  ONT)
    MODEL_TYPE="ONT_R104"
    OUT_TECH_FOLDER="$ONT_TECH_LABEL"
    if [[ "$COV" == "orig" ]]; then
      BAM="$ONT_ORIG_BAM_DEFAULT"
    else
      [[ "$SEED" != "orig" ]] || die "Downsampled cov=$COV requires --seed"
      BAM="${DOWNSAMPLED_BASE_DEFAULT}/ONT_partial_chr20/HG003.ONT_partial.chr20.${COV}x.s${SEED}.bam"
    fi
    ;;
  *) die "Unsupported tech: $TECH (use Illumina or ONT)";;
esac

[[ -f "$BAM" ]] || die "BAM not found: $BAM"
[[ -f "${BAM}.bai" ]] || die "BAI not found: ${BAM}.bai"
[[ -f "$REF" ]] || die "REF not found: $REF"
[[ -f "${REF}.fai" ]] || die "REF fai not found: ${REF}.fai"

# -----------------------
# Outputs (your convention)
# -----------------------
COV_TAG=$([[ "$COV" == "orig" ]] && echo "covorig" || echo "cov${COV}x")
SEED_TAG=$([[ "$SEED" == "orig" ]] && echo "sorig" || echo "s${SEED}")
RUN_TAG="${COV_TAG}_${SEED_TAG}"

OUT_DIR="data/4_out/deepvariant/${SAMPLE}/${OUT_TECH_FOLDER}/${REGION}/${RUN_TAG}"
LOG_DIR="${OUT_DIR}/logs"
INT_DIR="${OUT_DIR}/intermediate"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$INT_DIR"

OUT_VCF="${OUT_DIR}/output.vcf.gz"
OUT_GVCF="${OUT_DIR}/output.g.vcf.gz"
CONFIG_JSON="${OUT_DIR}/run_config.json"

write_config_json () {
  cat > "$CONFIG_JSON" <<EOF
{
  "tool": "deepvariant",
  "dv_version": "${DV_VERSION}",
  "docker_image": "${DV_IMAGE}:${DV_VERSION}",
  "sample": "${SAMPLE}",
  "tech_arg": "${TECH}",
  "out_tech_folder": "${OUT_TECH_FOLDER}",
  "model_type": "${MODEL_TYPE}",
  "region": "${REGION}",
  "coverage": "${COV}",
  "seed": "${SEED}",
  "threads": ${THREADS},
  "bam": "${BAM}",
  "ref": "${REF}",
  "out_dir": "${OUT_DIR}",
  "out_vcf": "${OUT_VCF}",
  "out_gvcf": "${OUT_GVCF}",
  "intermediate_results_dir": "${INT_DIR}",
  "logging_dir": "${LOG_DIR}"
}
EOF
}

write_config_json

echo "=== DeepVariant ==="
echo "TECH=$TECH  MODEL=$MODEL_TYPE  DV=$DV_VERSION  REGION=$REGION  RUN=$RUN_TAG"
echo "BAM=$BAM"
echo "REF=$REF"
echo "OUT_DIR=$OUT_DIR"

if [[ -f "$OUT_VCF" ]]; then
  echo ">> (skip) VCF already exists: $OUT_VCF"
  exit 0
fi

docker pull "${DV_IMAGE}:${DV_VERSION}"

docker run --rm \
  -v "$PWD":/work \
  "${DV_IMAGE}:${DV_VERSION}" \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type="${MODEL_TYPE}" \
    --ref="/work/${REF}" \
    --reads="/work/${BAM}" \
    --output_vcf="/work/${OUT_VCF}" \
    --output_gvcf="/work/${OUT_GVCF}" \
    --num_shards="${THREADS}" \
    --intermediate_results_dir="/work/${INT_DIR}" \
    --logging_dir="/work/${LOG_DIR}" \
    --vcf_stats_report=true