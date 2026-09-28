#!/usr/bin/env bash
set -euo pipefail

# Validate all regional BAMs under data/1_input_bams.
# Cheap checks only:
# - file exists
# - index exists
# - samtools quickcheck
# - samtools idxstats
# - expected regions present with mapped reads > 0
# - metadata json exists
#
# Output:
# - human-readable summary on stdout
# - TSV report on disk
#
# Usage:
#   bash validate_input_bams.sh
#   bash validate_input_bams.sh --root data/1_input_bams
#   bash validate_input_bams.sh --regions chr20 chr21
#   bash validate_input_bams.sh --root data/1_input_bams --out reports/input_bam_validation.tsv

ROOT="data/1_input_bams"
OUT="reports/input_bam_validation.tsv"
REGIONS=("chr20" "chr21")

usage() {
  cat <<USAGE
Usage:
  $0 [--root PATH] [--out FILE] [--regions chr20 chr21]

Options:
  --root     Root directory containing sample/tech BAMs [default: data/1_input_bams]
  --out      Output TSV report [default: reports/input_bam_validation.tsv]
  --regions  Expected regions with mapped reads > 0 [default: chr20 chr21]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --regions)
      REGIONS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do REGIONS+=("$1"); shift; done
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

command -v samtools >/dev/null || { echo "ERROR: samtools not found"; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 1; }

mkdir -p "$(dirname "$OUT")"

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

printf "bam_path\tstatus\tquickcheck_ok\tindex_ok\tmetadata_ok\texpected_regions_ok\tmapped_regions\tmissing_regions\tnotes\n" > "$OUT"

mapfile -t BAMS < <(find "$ROOT" -type f -name "*.bam" | sort)

if [[ ${#BAMS[@]} -eq 0 ]]; then
  echo "No BAMs found under $ROOT"
  exit 1
fi

PASS=0
FAIL=0

echo "Validating ${#BAMS[@]} BAM(s) under: $ROOT"
echo "Expected regions: ${REGIONS[*]}"
echo

for BAM in "${BAMS[@]}"; do
  STATUS="PASS"
  QUICKCHECK_OK="yes"
  INDEX_OK="yes"
  METADATA_OK="yes"
  EXPECTED_OK="yes"
  NOTES=()

  BAI1="${BAM}.bai"
  BAI2="${BAM%.bam}.bai"
  META="${BAM%.bam}.metadata.json"

  if ! [[ -f "$BAI1" || -f "$BAI2" ]]; then
    INDEX_OK="no"
    STATUS="FAIL"
    NOTES+=("missing_index")
  fi

  if ! [[ -f "$META" ]]; then
    METADATA_OK="no"
    STATUS="FAIL"
    NOTES+=("missing_metadata")
  fi

  QC_OUT="${TMPDIR_LOCAL}/quickcheck.txt"
  if ! samtools quickcheck -v "$BAM" >"$QC_OUT" 2>&1; then
    QUICKCHECK_OK="no"
    STATUS="FAIL"
    NOTES+=("quickcheck_failed")
  fi

  IDX_OUT="${TMPDIR_LOCAL}/idxstats.txt"
  if ! samtools idxstats "$BAM" >"$IDX_OUT" 2>"${TMPDIR_LOCAL}/idxstats.err"; then
    EXPECTED_OK="no"
    STATUS="FAIL"
    NOTES+=("idxstats_failed")
    MAPPED_REGIONS=""
    MISSING_REGIONS="$(IFS=,; echo "${REGIONS[*]}")"
  else
    MAPPED_REGIONS=$(awk '$3>0 {print $1}' "$IDX_OUT" | paste -sd "," -)
    if [[ -z "$MAPPED_REGIONS" ]]; then
      NOTES+=("no_mapped_reads")
      STATUS="FAIL"
      EXPECTED_OK="no"
    fi

    MISSING=()
    for R in "${REGIONS[@]}"; do
      COUNT=$(awk -v r="$R" '$1==r {print $3}' "$IDX_OUT")
      if [[ -z "${COUNT:-}" || "$COUNT" == "0" ]]; then
        MISSING+=("$R")
      fi
    done

    if [[ ${#MISSING[@]} -gt 0 ]]; then
      EXPECTED_OK="no"
      STATUS="FAIL"
      NOTES+=("missing_expected_regions")
      MISSING_REGIONS=$(IFS=,; echo "${MISSING[*]}")
    else
      MISSING_REGIONS=""
    fi
  fi

  if [[ "$STATUS" == "PASS" ]]; then
    ((PASS+=1))
    echo "[PASS] $BAM"
  else
    ((FAIL+=1))
    echo "[FAIL] $BAM"
    echo "       quickcheck_ok=$QUICKCHECK_OK index_ok=$INDEX_OK metadata_ok=$METADATA_OK expected_regions_ok=$EXPECTED_OK"
    echo "       mapped_regions=${MAPPED_REGIONS:-}"
    echo "       missing_regions=${MISSING_REGIONS:-}"
    echo "       notes=$(IFS=,; echo "${NOTES[*]}")"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$BAM" \
    "$STATUS" \
    "$QUICKCHECK_OK" \
    "$INDEX_OK" \
    "$METADATA_OK" \
    "$EXPECTED_OK" \
    "${MAPPED_REGIONS:-}" \
    "${MISSING_REGIONS:-}" \
    "$(IFS=,; echo "${NOTES[*]}")" >> "$OUT"
done

echo
echo "Validation finished."
echo "PASS: $PASS"
echo "FAIL: $FAIL"
echo "TSV report: $OUT"
