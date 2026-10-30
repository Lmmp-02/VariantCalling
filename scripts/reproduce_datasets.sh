#!/usr/bin/env bash
set -euo pipefail

# Reproduce the seven singleton-locus multimodal datasets used by the paper.
# This is orchestration only: it filters the existing downsampling manifest and
# chains the existing downsampling, DeepVariant and multimodal-join scripts.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUNTIME="docker"
ONLY=""
DRY_RUN=0
MANIFEST="scripts/data/config/downsample_multimodal_benchmark.tsv"

DATASETS=(
  hg002_chr20_chr21_harmonized_40x
  hg003_chr20_chr21_harmonized_40x
  hg004_chr20_chr21_harmonized_40x
  hg005_chr20_chr21_harmonized_40x
  hg005_chr20_chr21_coverage_study_20x
  hg005_chr20_chr21_coverage_study_10x
  hg005_chr20_chr21_coverage_study_5x
)

usage() {
  cat <<'USAGE'
Reproduce the seven paper multimodal datasets by chaining existing scripts.

Usage:
  scripts/reproduce_datasets.sh [options]

Options:
  --runtime docker|udocker   DeepVariant container runtime (default: docker)
  --only DATASET_ID         Reproduce one of the seven paper datasets only
  --dry-run                 Print the complete plan without executing or writing files
  -h, --help                Show this help

The downsampling step uses scripts/data/downsample_multimodal_benchmark.sh and
therefore inherits its MICROMAMBA_BIN, CONDA_ENV, THREADS and FORCE settings.
See docs/reproduce_datasets.md for required inputs and the Level 3 scope.
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

contains_dataset() {
  local wanted="$1"
  local item
  for item in "${DATASETS[@]}"; do
    [[ "$item" == "$wanted" ]] && return 0
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime)
      [[ $# -ge 2 ]] || die "--runtime requires a value"
      RUNTIME="$2"
      shift 2
      ;;
    --only)
      [[ $# -ge 2 ]] || die "--only requires a dataset id"
      ONLY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ "$RUNTIME" == "docker" || "$RUNTIME" == "udocker" ]] || die "--runtime must be docker|udocker"
[[ -z "$ONLY" ]] || contains_dataset "$ONLY" || die "Unknown paper dataset id: $ONLY"
[[ -f "$MANIFEST" ]] || die "Downsampling manifest not found: $MANIFEST"

selected_datasets=()
if [[ -n "$ONLY" ]]; then
  selected_datasets+=("$ONLY")
else
  selected_datasets=("${DATASETS[@]}")
fi

# dataset_spec prints: sample<TAB>output_group<TAB>target_cov<TAB>coverage_tag
# The mapping mirrors the seven dataset ids frozen for the paper.
dataset_spec() {
  case "$1" in
    hg002_chr20_chr21_harmonized_40x) printf 'HG002\tharmonized_40x\t40\tharmonized_40x\n' ;;
    hg003_chr20_chr21_harmonized_40x) printf 'HG003\tharmonized_40x\t40\tharmonized_40x\n' ;;
    hg004_chr20_chr21_harmonized_40x) printf 'HG004\tharmonized_40x\t40\tharmonized_40x\n' ;;
    hg005_chr20_chr21_harmonized_40x) printf 'HG005\tharmonized_40x\t40\tharmonized_40x\n' ;;
    hg005_chr20_chr21_coverage_study_20x) printf 'HG005\tcoverage_study\t20\tcoverage_study_20x\n' ;;
    hg005_chr20_chr21_coverage_study_10x) printf 'HG005\tcoverage_study\t10\tcoverage_study_10x\n' ;;
    hg005_chr20_chr21_coverage_study_5x) printf 'HG005\tcoverage_study\t5\tcoverage_study_5x\n' ;;
    *) return 1 ;;
  esac
}

manifest_rows_for_dataset() {
  local dataset_id="$1"
  local sample output_group target_cov coverage_tag
  IFS=$'\t' read -r sample output_group target_cov coverage_tag < <(dataset_spec "$dataset_id")
  awk -F '\t' -v sample="$sample" -v group="$output_group" -v cov="$target_cov" '
    NR == 1 { print; next }
    $1 == "1" && $2 == sample && $7 == group && $8 == cov { print }
  ' "$MANIFEST"
}

manifest_row_count_for_dataset() {
  local dataset_id="$1"
  manifest_rows_for_dataset "$dataset_id" | awk 'END { print NR - 1 }'
}

derived_bam_for_tech() {
  local dataset_id="$1"
  local wanted_tech="$2"
  local sample output_group target_cov coverage_tag
  IFS=$'\t' read -r sample output_group target_cov coverage_tag < <(dataset_spec "$dataset_id")

  local row enabled row_sample tech source_policy regions input_bam row_group row_cov seed notes
  row="$(awk -F '\t' -v sample="$sample" -v group="$output_group" -v cov="$target_cov" -v tech="$wanted_tech" '
    NR > 1 && $1 == "1" && $2 == sample && $3 == tech && $7 == group && $8 == cov { print; exit }
  ' "$MANIFEST")"
  [[ -n "$row" ]] || die "No manifest row for ${dataset_id} / ${wanted_tech}"

  IFS=$'\t' read -r enabled row_sample tech source_policy regions input_bam row_group row_cov seed notes <<< "$row"
  local base stem
  base="$(basename "$input_bam")"
  stem="${base%.bam}"
  printf 'data/1_input_bams/%s/derived/%s/%s/%s/%s.cov%sx.s%s.bam\n' \
    "$row_sample" "$regions" "$row_group" "$tech" "$stem" "$row_cov" "$seed"
}

print_command() {
  printf '[DRY-RUN]'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

run_downsampling() {
  local dataset_id="$1"
  local rows
  rows="$(manifest_row_count_for_dataset "$dataset_id")"
  [[ "$rows" == "2" ]] || die "Expected exactly 2 enabled downsampling rows for ${dataset_id}; found ${rows}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[Downsample] ${dataset_id} (2 filtered rows from ${MANIFEST})"
    print_command bash scripts/data/downsample_multimodal_benchmark.sh "<filtered-manifest:${dataset_id}>"
    return 0
  fi

  local tmp_manifest rc
  tmp_manifest="$(mktemp "${TMPDIR:-/tmp}/variantcalling_${dataset_id}.XXXXXX.tsv")"
  manifest_rows_for_dataset "$dataset_id" > "$tmp_manifest"
  if bash scripts/data/downsample_multimodal_benchmark.sh "$tmp_manifest"; then
    rm -f -- "$tmp_manifest"
  else
    rc=$?
    rm -f -- "$tmp_manifest"
    return "$rc"
  fi
}

for dataset_id in "${selected_datasets[@]}"; do
  IFS=$'\t' read -r sample output_group target_cov coverage_tag < <(dataset_spec "$dataset_id")
  ill_bam="$(derived_bam_for_tech "$dataset_id" Illumina)"
  ont_bam="$(derived_bam_for_tech "$dataset_id" ONT)"

  echo
  echo "================================================================"
  echo "[Dataset] ${dataset_id}"
  echo "[Sample ] ${sample}"
  echo "[Target ] ${target_cov}x (${output_group})"
  echo "================================================================"

  run_downsampling "$dataset_id"

  run_command bash scripts/deepvariant/dv_make_examples.sh \
    --tech illumina \
    --bam "$ill_bam" \
    --sample "$sample" \
    --scope_id chr20_chr21 \
    --coverage_tag "$coverage_tag" \
    --dataset_id "$dataset_id" \
    --mode training \
    --runtime "$RUNTIME"

  run_command bash scripts/deepvariant/dv_make_examples.sh \
    --tech ont \
    --bam "$ont_bam" \
    --sample "$sample" \
    --scope_id chr20_chr21 \
    --coverage_tag "$coverage_tag" \
    --dataset_id "$dataset_id" \
    --mode training \
    --runtime "$RUNTIME"

  run_command bash scripts/deepvariant/dv_export_features.sh \
    --dataset_id "$dataset_id" \
    --tech illumina \
    --mode training \
    --runtime "$RUNTIME"

  run_command bash scripts/deepvariant/dv_export_features.sh \
    --dataset_id "$dataset_id" \
    --tech ont \
    --mode training \
    --runtime "$RUNTIME"

  run_command bash scripts/multimodal/dv_build_join_datasets_singleton_locus.sh \
    --dataset_id "$dataset_id" \
    --mode training

done

cat <<'EOF'

Dataset recipe complete.
Run the group-count check after all requested joins exist:
  python scripts/multimodal/check_group_counts.py

EOF
