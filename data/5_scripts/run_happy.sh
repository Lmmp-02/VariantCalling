#!/usr/bin/env bash
set -euo pipefail

cd ~/VariantCalling

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate happy

# -----------------------
# Defaults (editables)
# -----------------------
THREADS_DEFAULT="$(nproc)"
DV_VERSION_DEFAULT="1.9.0"
SAMPLE_DEFAULT="HG003"
REGION_DEFAULT="chr20"

REF_DEFAULT="data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa"
TRUTH_VCF_DEFAULT="data/3_truth/HG003/HG003_chr20.truth.vcf.gz"
TRUTH_BED_DEFAULT="data/3_truth/HG003/HG003_chr20.confident.bed"

ONT_TECH_LABEL_DEFAULT="ONT_R104_sup_PAY87794"

# -----------------------
# Args
# -----------------------
TECH=""          # Illumina | ONT
COV="orig"
SEED="orig"
REGION="$REGION_DEFAULT"
DV_VERSION="$DV_VERSION_DEFAULT"
THREADS="$THREADS_DEFAULT"
SAMPLE="$SAMPLE_DEFAULT"
REF="$REF_DEFAULT"
TRUTH_VCF="$TRUTH_VCF_DEFAULT"
TRUTH_BED="$TRUTH_BED_DEFAULT"
ONT_TECH_LABEL="$ONT_TECH_LABEL_DEFAULT"

usage() {
  cat <<EOF
Usage:
  $0 --tech {Illumina|ONT} [--cov orig|30|15|10|5] [--seed orig|42|...] [--region chr20]
     [--dv_version 1.9.0] [--threads N] [--ref path.fa]
     [--truth_vcf path.vcf.gz] [--truth_bed path.bed]
     [--sample HG003] [--ont_label ONT_R104_sup_PAY87794]

Examples:
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
    --truth_vcf) TRUTH_VCF="$2"; shift 2;;
    --truth_bed) TRUTH_BED="$2"; shift 2;;
    --sample) SAMPLE="$2"; shift 2;;
    --ont_label) ONT_TECH_LABEL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "$TECH" ]] || die "Missing --tech"
[[ -f "$REF" ]] || die "REF not found: $REF"
[[ -f "$TRUTH_VCF" ]] || die "Truth VCF not found: $TRUTH_VCF"
[[ -f "$TRUTH_BED" ]] || die "Truth BED not found: $TRUTH_BED"

# Match DV output folder naming
OUT_TECH_FOLDER=$([[ "$TECH" == "Illumina" ]] && echo "Illumina" || echo "$ONT_TECH_LABEL")

COV_TAG=$([[ "$COV" == "orig" ]] && echo "covorig" || echo "cov${COV}x")
SEED_TAG=$([[ "$SEED" == "orig" ]] && echo "sorig" || echo "s${SEED}")
RUN_TAG="${COV_TAG}_${SEED_TAG}"

QUERY_VCF="data/4_out/deepvariant/${SAMPLE}/${OUT_TECH_FOLDER}/${REGION}/${RUN_TAG}/output.vcf.gz"
[[ -f "$QUERY_VCF" ]] || die "Query VCF not found (run DeepVariant first?): $QUERY_VCF"

OUT_DIR="data/4_out/happy/${SAMPLE}/${OUT_TECH_FOLDER}/${REGION}/${RUN_TAG}"
mkdir -p "$OUT_DIR"
OUT_PREFIX="${OUT_DIR}/happy"

SUMMARY="${OUT_DIR}/summary.csv"
EXTENDED="${OUT_DIR}/extended.csv"
CONFIG_JSON="${OUT_DIR}/run_config.json"

write_config_json () {
  cat > "$CONFIG_JSON" <<EOF
{
  "tool": "hap.py",
  "sample": "${SAMPLE}",
  "tech_arg": "${TECH}",
  "out_tech_folder": "${OUT_TECH_FOLDER}",
  "region": "${REGION}",
  "coverage": "${COV}",
  "seed": "${SEED}",
  "threads": ${THREADS},
  "ref": "${REF}",
  "truth_vcf": "${TRUTH_VCF}",
  "truth_bed": "${TRUTH_BED}",
  "query_vcf": "${QUERY_VCF}",
  "out_dir": "${OUT_DIR}",
  "out_prefix": "${OUT_PREFIX}"
}
EOF
}

write_config_json

echo "=== hap.py ==="
echo "TECH=$TECH REGION=$REGION RUN=$RUN_TAG"
echo "TRUTH_VCF=$TRUTH_VCF"
echo "QUERY_VCF=$QUERY_VCF"
echo "OUT_DIR=$OUT_DIR"

if [[ -f "$SUMMARY" ]]; then
  echo ">> (skip) summary already exists: $SUMMARY"
  exit 0
fi

hap.py \
  "$TRUTH_VCF" \
  "$QUERY_VCF" \
  -f "$TRUTH_BED" \
  -r "$REF" \
  -o "$OUT_PREFIX" \
  --threads "$THREADS"

# Normalize to fixed names
if [[ -f "${OUT_PREFIX}.summary.csv" ]]; then cp "${OUT_PREFIX}.summary.csv" "$SUMMARY"; fi
if [[ -f "${OUT_PREFIX}.extended.csv" ]]; then cp "${OUT_PREFIX}.extended.csv" "$EXTENDED"; fi